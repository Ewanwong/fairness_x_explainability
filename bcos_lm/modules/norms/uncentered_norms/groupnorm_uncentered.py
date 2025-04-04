"""
Group norm without centering.
Also contains GN-based LayerNorm and InstanceNorm wrappers,
`GNLayerNorm` and `GNInstanceNorm`.

Code partially taken from PyTorch
"""
from typing import Optional

import torch.nn as nn
from torch import Tensor

from bcos_lm.modules.common import DynamicMultiplication

__all__ = [
    "group_norm_uncentered",
    "GroupNormUncentered2d",
    "GNInstanceNormUncentered2d",
    "GNLayerNormUncentered2d",
]


def group_norm_uncentered(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    dynamic_multiplication: Optional[DynamicMultiplication] = None,
):
    """
    Un-centered group norm on batched input
    """
    assert input.shape[1] % num_groups == 0, (
        "Number of channels in input should be divisible by num_groups, "
        f"but got input of shape {input.shape} and num_groups={num_groups}"
    )

    # separate the groups
    # (N, C, *) -> (N, G, C // G, *)
    N, C = input.shape[:2]
    x = input.reshape(N, num_groups, C // num_groups, *input.shape[2:])

    # calc stats
    var = x.var(dim=tuple(range(2, x.dim())), unbiased=False, keepdim=True)
    std = (var + eps).sqrt()

    # normalize
    if dynamic_multiplication:
        x = dynamic_multiplication(input=x, weight=1 / std)
    else:
        x = x / std

    # reshape back
    x = x.reshape(input.shape)

    # affine transformation
    if weight is not None:
        x = weight[None, ..., None, None] * x

    if bias is not None:
        x = x + bias[None, ..., None, None]

    return x


class GroupNormUncentered2d(nn.GroupNorm):
    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        super().__init__(num_groups, num_channels, eps, affine)
        self.dynamic_multiplication = DynamicMultiplication()

    def forward(self, input: Tensor) -> Tensor:
        return group_norm_uncentered(
            input,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
            dynamic_multiplication=self.dynamic_multiplication,
        )


class GNInstanceNormUncentered2d(GroupNormUncentered2d):
    def __init__(self, num_channels: int, *args, **kwargs):
        super().__init__(
            num_groups=num_channels,
            num_channels=num_channels,
            *args,
            **kwargs,
        )


class GNLayerNormUncentered2d(GroupNormUncentered2d):
    def __init__(self, num_channels: int, *args, **kwargs):
        super().__init__(
            num_groups=1,
            num_channels=num_channels,
            *args,
            **kwargs,
        )