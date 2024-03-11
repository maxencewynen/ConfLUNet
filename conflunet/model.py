from torch import nn
import torch
from typing import Tuple


class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Desired number of output channels.
        bottleneck (bool, optional): Specifies the bottleneck block. Defaults to False.

    Returns:
        Tensor: Output tensor after convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, bottleneck: bool = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels // 2)
        self.conv2 = nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.relu(self.bn2(self.conv2(res)))
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path

    Args:
        in_channels (int): Number of input channels.
        res_channels (int, optional): Number of residual connections' channels to be concatenated. Defaults to 0.
        last_layer (bool, optional): Specifies the last output layer. Defaults to False.
        out_channels (int, optional): Number of output channels for disparate classes. Defaults to None.

    Returns:
        torch.Tensor: Output Tensor.
    """

    def __init__(self, in_channels: int, res_channels: int = 0, last_layer: bool = False, out_channels: int = None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (not last_layer and out_channels is None) or (last_layer and out_channels is not None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels + res_channels, out_channels=in_channels // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels // 2, out_channels=out_channels, kernel_size=(1, 1, 1))

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input Tensor.
            residual (torch.Tensor, optional): Residual connection to be concatenated with input. Defaults to None.

        Returns:
            torch.Tensor: Output Tensor.
        """
        out = self.upconv1(x)
        if residual is not None:
            out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer:
            out = self.conv3(out)
        return out


class ConfLUNet(nn.Module):
    """
    The ConfLUNet model architecture, based on the exact same architecture as UNet3D.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        level_channels (Tuple[int, int, int]): Number of channels at each level of the analysis path. Default is (64, 128, 256).
        bottleneck_channel (int): Number of channels for the bottleneck block. Default is 512.
        separate_decoders (bool): Whether to use separate decoders for semantic and object center prediction. Default is False.
        scale_offsets (int): Scale factor for the predicted offsets. Output offsets will be multiplied by this value. Default is 1.

    Returns:
        torch.Tensor: Semantic segmentation output.
        torch.Tensor: Object center prediction output.
        torch.Tensor: Offset prediction output.
    """

    def __init__(self, in_channels: int, num_classes: int, level_channels: Tuple[int, int, int] = (64, 128, 256),
                 bottleneck_channel: int = 512, separate_decoders: bool = False, scale_offsets: int = 1) -> None:
        super(ConfLUNet, self).__init__()

        self.scale_offsets = scale_offsets

        # Analysis Path
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_channels[0])
        self.a_block2 = Conv3DBlock(in_channels=level_channels[0], out_channels=level_channels[1])
        self.a_block3 = Conv3DBlock(in_channels=level_channels[1], out_channels=level_channels[2])
        self.bottleNeck = Conv3DBlock(in_channels=level_channels[2], out_channels=bottleneck_channel, bottleneck=True)

        # Semantic Decoding Path
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_channels[2])
        self.s_block2 = UpConv3DBlock(in_channels=level_channels[2], res_channels=level_channels[1])

        self.separate_decoders = separate_decoders
        if not self.separate_decoders:
            self.s_block1 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0],
                                          out_channels=num_classes + 4, last_layer=True)
        else:
            self.s_block1 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0],
                                          out_channels=num_classes, last_layer=True)

            self.oc_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_channels[2])
            self.oc_block2 = UpConv3DBlock(in_channels=level_channels[2], res_channels=level_channels[1])
            self.oc_block1 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0],
                                           out_channels=4, last_layer=True)

        self._init_parameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConfLUNet model.

        Args:
            x (torch.Tensor): Input Tensor.

        Returns:
            torch.Tensor: Semantic segmentation output.
            torch.Tensor: Object center prediction output.
            torch.Tensor: Offset prediction output.
        """
        # Analysis Pathway
        out, residual_level1 = self.a_block1(x)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        # Semantic Decoding Pathway
        sem_decoder_out = self.s_block3(out, residual_level3)
        sem_decoder_out = self.s_block2(sem_decoder_out, residual_level2)
        sem_decoder_out = self.s_block1(sem_decoder_out, residual_level1)

        if not self.separate_decoders:
            semantic_out = sem_decoder_out[:, :2]
            center_prediction_out = sem_decoder_out[:, 2:3]
            offsets_out = sem_decoder_out[:, 3:] * self.scale_offsets
        else:
            oc_decoder_out = self.oc_block3(out, residual_level3)
            oc_decoder_out = self.oc_block2(oc_decoder_out, residual_level2)
            oc_decoder_out = self.oc_block1(oc_decoder_out, residual_level1)

            semantic_out = sem_decoder_out
            center_prediction_out = oc_decoder_out[:, :1]
            offsets_out = oc_decoder_out[:, 1:] * self.scale_offsets

        return semantic_out, center_prediction_out, offsets_out

    def _init_parameters(self) -> None:
        """
        Initialize parameters for the model.
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                # torch.nn.init.normal_(m.weight, std=0.001)
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


class UNet3D(nn.Module):
    """
    The UNet3D model architecture.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        level_channels (Tuple[int, int, int]): Number of channels at each level of the analysis path. Default is (64, 128, 256).
        bottleneck_channel (int): Number of channels for the bottleneck block. Default is 512.
        separate_decoders (bool): Whether to use separate decoders for semantic and object center prediction. Default is False.
        scale_offsets (int): Scale factor for the predicted offsets. Output offsets will be multiplied by this value. Default is 1.

    Returns:
        torch.Tensor: Semantic segmentation output.
        torch.Tensor: Object center prediction output.
        torch.Tensor: Offset prediction output.
    """
    def __init__(self, in_channels: int, num_classes: int, level_channels: Tuple[int, int, int] = (64, 128, 256),
                 bottleneck_channel: int = 512) -> None:
        super(UNet3D, self).__init__()

        # Analysis Path
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_channels[0])
        self.a_block2 = Conv3DBlock(in_channels=level_channels[0], out_channels=level_channels[1])
        self.a_block3 = Conv3DBlock(in_channels=level_channels[1], out_channels=level_channels[2])
        self.bottleNeck = Conv3DBlock(in_channels=level_channels[2], out_channels=bottleneck_channel, bottleneck=True)

        # Semantic Decoding Path
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_channels[2])
        self.s_block2 = UpConv3DBlock(in_channels=level_channels[2], res_channels=level_channels[1])
        self.s_block1 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0],
                                      out_channels=num_classes, last_layer=True)

        self._init_parameters()

    def forward(self, input):
        # Analysis Pathway
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        # Semantic Decoding Pathway
        decoder_out = self.s_block3(out, residual_level3)
        decoder_out = self.s_block2(decoder_out, residual_level2)
        decoder_out = self.s_block1(decoder_out, residual_level1)

        semantic_out = decoder_out

        return semantic_out

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                # torch.nn.init.normal_(m.weight, std=0.001)
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


def get_pretrained_model(model_path, in_channels, num_classes=2):
    model = UNet3D(in_channels=1, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    if in_channels == 1:
        return model

    sd_layer1 = model.a_block1.conv1.state_dict()

    duplicated_weight = sd_layer1["weight"].repeat(1, in_channels, 1, 1, 1)
    bias = sd_layer1["bias"]

    out_channels = model.a_block1.conv1.out_channels
    new_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
    new_layer.load_state_dict({"weight": duplicated_weight, "bias": bias})
    model.a_block1.conv1 = new_layer

    return model


if __name__ == '__main__':
    model = ConfLUNet(in_channels=1, num_classes=2).cuda()
    a = torch.rand(2, 1, 96, 96, 96).cuda()
    output = model(a)
