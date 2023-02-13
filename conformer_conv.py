import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from conformer_activations import Swish, GLU


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, inputs):
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.
    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            padding=0,
            bias=True,
    ):
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs):
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.
    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
        device (torch.device): torch device (cuda or cpu)
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """

    def __init__(self,in_channels, kernel_size=31, expansion_factor=2, dropout=0.1):
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.layernorm = nn.LayerNorm(in_channels)
        self.pointconv1 = PointwiseConv1d(in_channels, in_channels*expansion_factor, stride=1, padding=0, bias=True)
        self.glu = GLU(dim=1)
        self.depthconv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.batchnorm = nn.BatchNorm1d(in_channels)
        self.swish = Swish()
        self.pointconv2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        residual = x
        output = self.layernorm(x)
        output = output.transpose(1,2)
        output = self.pointconv1(output)
        output = self.glu(output)
        output = self.depthconv(output)
        output = self.batchnorm(output)
        output = self.swish(output)
        output = self.pointconv2(output)
        output = self.dropout(output).transpose(1,2)
        output = output + residual
        return output
