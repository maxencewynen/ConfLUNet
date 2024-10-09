from typing import Union, Type, List, Tuple

import numpy as np
import torch
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
import pydoc
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


class ConfLUNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision: bool,
                 nonlin_first: bool = False,):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias
        norm_op = encoder.norm_op
        norm_op_kwargs = encoder.norm_op_kwargs
        dropout_op = encoder.dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs
        nonlin = encoder.nonlin
        nonlin_kwargs = encoder.nonlin_kwargs

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))

            # if last stage
            if s == n_stages_encoder - 1:
                seg_output = StackedConvBlocks(
                    n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first
                )

                centers = StackedConvBlocks(
                    n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first
                )

                offsets = StackedConvBlocks(
                    n_conv_per_stage[s - 1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first
                )

                self.last_stage_seg_output = seg_output
                self.last_stage_centers_output = centers
                self.last_stage_offsets_output = offsets
                self.last_stage = (self.last_stage_seg_output,
                                   self.last_stage_centers_output,
                                   self.last_stage_offsets_output)
                seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
                self.center_output_layer = encoder.conv_op(input_features_skip, 1, 1, 1, 0, bias=True)
                self.offsets_output_layer = encoder.conv_op(input_features_skip, 3, 1, 1, 0, bias=True)

            else:
                # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
                stages.append(StackedConvBlocks(
                    n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                    encoder.kernel_sizes[-(s + 1)], 1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first
                ))

                # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
                # then a model trained with deep_supervision=True could not easily be loaded at inference time where
                # deep supervision is not needed. It's just a convenience thing
                seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        s = -1
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)

            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            lres_input = x

        x = self.transpconvs[s+1](lres_input)
        x = torch.cat((x, skips[-(s + 1 + 2)]), 1)
        seg_output, centers, offsets = self.last_stage
        seg = seg_output(x)
        seg_outputs.append(self.seg_layers[s+1](seg))

        center_output = self.center_output_layer(centers(x))
        offsets_output = self.offsets_output_layer(offsets(x))

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = (seg_outputs[0], center_output, offsets_output)
        else:
            r = (seg_outputs, center_output, offsets_output)
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


class nnConfLUNet(PlainConvUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage,
                 num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                 dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first)

        self.decoder = ConfLUNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                        nonlin_first=nonlin_first)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    """
    Adapted from nnUNetv2.utilities.get_network_from_plans
    """
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)

    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)

    if nw_class is None:
        raise RuntimeError(f"network class {network_class} not found")

    if deep_supervision is not None and 'deep_supervision' not in arch_kwargs.keys():
        arch_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


if __name__ == '__main__':
    model = nnConfLUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=torch.nn.modules.conv.Conv3d,
        kernel_sizes=[[3, 3, 3]]*6,
        strides=[[1,1,1]] + [[2,2,2]]*4 + [[1,2,2]],
        n_conv_per_stage=[2]*6,
        num_classes=2,
        n_conv_per_stage_decoder=[2]*5,
        conv_bias=True,
        norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
        # nonlin_first=False
    )

    model = nnConfLUNet(
        input_channels=1,
        n_stages=3,
        features_per_stage=(32, 64, 128),
        conv_op=torch.nn.modules.conv.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 3,
        strides=[[1, 1, 1]] + [[2,2,2]] * 1 + [[1, 2, 2]],
        n_conv_per_stage=[2] * 3,
        num_classes=2,
        n_conv_per_stage_decoder=[2] * 2,
        conv_bias=True,
        norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
        # nonlin_first=False
    ).to('cuda')

    # print(model.decoder)
    # print("hello")
    #
    # data = torch.randn(1, 1, 96, 96, 96).to('cuda')
    # out = model(data)
    # print()
    #
    # from torchviz import make_dot
    #
    # if isinstance(out, list) or isinstance(out, tuple):
    #     seg = out[0] if isinstance(out[0], torch.Tensor) else out[0][0]
    #     out = torch.cat([seg, out[1], out[2]], dim=1)  # Combine into a single tensor
    #
    # # Visualize the model
    # make_dot(out, params=dict(model.named_parameters())).render("model_torchviz", format="png")

    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))