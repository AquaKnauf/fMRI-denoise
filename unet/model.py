import torch
import torch.nn as nn
import torch.nn.functional as F



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
        Returns the padded tensor and the original shape for proper unpadding.
        """
        _, _, d, h, w = x.shape
        factor = self.total_down_factor

        pad_d = (factor - d % factor) % factor
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor

        # Distribute padding as evenly as possible
        pad_d_front = pad_d // 2
        pad_d_back = pad_d - pad_d_front
        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        padding = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back)
        x_padded = F.pad(x, padding)

        # Store original shape for unpadding
        original_shape = (d, h, w)
        return x_padded, (original_shape, padding)

    def remove_padding(self, x, padding_info):
        """
        Remove padding from the tensor x using the original shape.
        """
        original_shape, padding = padding_info
        orig_d, orig_h, orig_w = original_shape
        
        # Simply crop to original dimensions from the center
        _, _, curr_d, curr_h, curr_w = x.shape
        
        # Calculate start indices (should be the padding we added)
        start_d = (curr_d - orig_d) // 2
        start_h = (curr_h - orig_h) // 2
        start_w = (curr_w - orig_w) // 2
        
        return x[:, :, 
                start_d:start_d + orig_d,
                start_h:start_h + orig_h, 
                start_w:start_w + orig_w]

    def forward(self, x):
        # Store original shape and pad
        original_shape = x.shape[2:]  # (D, H, W)
        x, padding_info = self.pad_to_divisible(x)
        
        # Debug prints (remove these once working)
        

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
        out = self.remove_padding(out, padding_info)


        return out
