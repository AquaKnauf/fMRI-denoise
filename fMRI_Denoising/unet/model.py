class UNet3DfMRI(nn.Module):
    def __init__(self, input_channels=1, conv_filt=32, kernel_size=5, activation="relu", pool_size=2):
        super(UNet3DfMRI, self).__init__()
        padding = kernel_size // 2
        act_fn = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2)

        self.pool_size = pool_size
        self.total_down_factor = pool_size ** 4  # 4 pooling layers → factor 16

        # Encoder
        self.enc1 = self._block(input_channels, conv_filt, kernel_size, padding, act_fn)
        self.pool1 = nn.MaxPool3d(pool_size)

        self.enc2 = self._block(conv_filt, 2*conv_filt, kernel_size, padding, act_fn)
        self.pool2 = nn.MaxPool3d(pool_size)

        self.enc3 = self._block(2*conv_filt, 4*conv_filt, kernel_size, padding, act_fn)
        self.pool3 = nn.MaxPool3d(pool_size)

        self.enc4 = self._block(4*conv_filt, 8*conv_filt, kernel_size, padding, act_fn)
        self.pool4 = nn.MaxPool3d(pool_size)

        # Bottleneck
        self.bottleneck = self._block(8*conv_filt, 16*conv_filt, kernel_size, padding, act_fn)

        # Decoder
        self.up4 = nn.ConvTranspose3d(16*conv_filt, 8*conv_filt, kernel_size=2, stride=2)
        self.dec4 = self._block(16*conv_filt, 8*conv_filt, kernel_size, padding, act_fn)

        self.up3 = nn.ConvTranspose3d(8*conv_filt, 4*conv_filt, kernel_size=2, stride=2)
        self.dec3 = self._block(8*conv_filt, 4*conv_filt, kernel_size, padding, act_fn)

        self.up2 = nn.ConvTranspose3d(4*conv_filt, 2*conv_filt, kernel_size=2, stride=2)
        self.dec2 = self._block(4*conv_filt, 2*conv_filt, kernel_size, padding, act_fn)

        self.up1 = nn.ConvTranspose3d(2*conv_filt, conv_filt, kernel_size=2, stride=2)
        self.dec1 = self._block(2*conv_filt, conv_filt, kernel_size, padding, act_fn)

        self.output_conv = nn.Conv3d(conv_filt, 1, kernel_size=1)
        self.apply(self._initialize_weights)

    def _block(self, in_channels, out_channels, kernel_size, padding, act_fn):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            act_fn,
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            act_fn
        )

    def _initialize_weights(self, module):
        import torch.nn.init as init
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm3d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def pad_to_divisible(self, x):
        """
        Pad x with zeros so that its spatial dims are divisible by total_down_factor.
        Returns the padded tensor and the padding sizes used.
        """
        _, _, d, h, w = x.shape
        factor = self.total_down_factor

        pad_d = (factor - d % factor) % factor
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor

        padding = (0, pad_w, 0, pad_h, 0, pad_d)  # (w_left, w_right, h_top, h_bottom, d_front, d_back)
        x_padded = F.pad(x, padding)

        return x_padded, padding

    def remove_padding(self, x, padding):
        """
        Remove padding from the tensor x. Inverse of pad_to_divisible.
        """
        _, _, d, h, w = x.shape
        pd, ph, pw = padding[5], padding[3], padding[1]  # back, bottom, right

        return x[:, :, :d-pd, :h-ph, :w-pw]

    def forward(self, x):
        # Store original shape and pad
        x, padding = self.pad_to_divisible(x)

        # Encoder
        conv1 = self.enc1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.enc2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.enc3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.enc4(pool3)
        pool4 = self.pool4(conv4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Decoder
        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([conv4, up4], dim=1))

        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([conv3, up3], dim=1))

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([conv2, up2], dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([conv1, up1], dim=1))

        out = self.output_conv(dec1)

        # Remove padding to get back to original shape
        out = self.remove_padding(out, padding)

        return out
