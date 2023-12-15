from typing import Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from lutils.configuration import Configuration
from model.layers.utils import SequenceConverter
from model.layers import build_position_encoding


class FlowRepresentationNetwork(nn.Module):
    """
    Model that encodes an optical flow into a state

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            tile_size: Tuple[int, int],
            out_res: Tuple[int, int],
            depth: int = 4):
        super(FlowRepresentationNetwork, self).__init__()

        self.tile_size = tile_size

        self.tiling = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=self.tile_size, stride=self.tile_size, padding=(0, 0)),
            nn.GELU()
        )

        self.encoding_layers = nn.ModuleList()
        for _ in range(depth):
            self.encoding_layers.append(nn.Sequential(
                Rearrange("b c h w -> b c (h w)"),
                nn.BatchNorm1d(256),
                Rearrange("b c (h w) -> b c h w", h=out_res[0]),
                nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                nn.GELU()
            ))

        self.out = nn.Sequential(
            Rearrange("b c h w -> b c (h w)"),
            nn.BatchNorm1d(256),
            Rearrange("b c (h w) -> b c h w", h=out_res[0]),
            nn.Conv2d(256, out_channels, kernel_size=(1, 1), stride=(1, 1),padding=(0, 0))
        )

        self.pos = build_position_encoding(out_channels, position_embedding_name="learned")

    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        """
        Computes the state corresponding to each observation

        :param flows: (bs, 3, height, width) tensor
        :return: (bs, state_features, state_height, state_width) tensor of states
        """

        # Tile flows
        x = self.tiling(flows)

        # Forward residual layers
        for layer in self.encoding_layers:
            x = x + layer(x)

        # Project to out_channels
        x = self.out(x)

        # Add position encodings
        pos = self.pos(x)
        x = x + pos

        return x


def build_flow_representation_network(config: Configuration, convert_to_sequence: bool = False) -> nn.Module:
    backbone = FlowRepresentationNetwork(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        tile_size=config["tile_size"],
        out_res=config["out_res"],
        depth=config["depth"])

    if convert_to_sequence:
        return SequenceConverter(backbone=backbone)
    else:
        return backbone
