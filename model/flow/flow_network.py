import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from lutils.configuration import Configuration


class FlowNetwork(nn.Module):
    def __init__(self, scale: float = 4.0):
        super(FlowNetwork, self).__init__()

        self.scale = scale

        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT)

    @torch.no_grad()
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """

        :param observations: [bs, num_observations, 3, height, width]
        :return: [bs, num_observations - 1, 2, height, width]
        """

        num_observations = observations.size(1)
        height = observations.size(3)
        width = observations.size(4)

        # Enlarge images to feed into RAFT
        flat_observations = rearrange(observations, "b n c h w -> (b n) c h w")
        flat_large_observations = F.interpolate(
            flat_observations, scale_factor=self.scale, mode="bilinear", align_corners=False)
        folded_large_observations = rearrange(flat_large_observations, "(b n) c h w -> b n c h w", n=num_observations)

        # Extract image pairs
        flat_first_images = rearrange(folded_large_observations[:, :-1], "b n c h w -> (b n) c h w")
        flat_second_images = rearrange(folded_large_observations[:, 1:], "b n c h w -> (b n) c h w")

        # Predict flows
        list_of_large_flows = self.raft(flat_first_images, flat_second_images)

        # Fold and resize to original size
        resized_flows = F.interpolate(list_of_large_flows[-1] / self.scale, size=[height, width], mode="nearest")
        folded_flows = rearrange(resized_flows, "(b n) c h w -> b n c h w", n=num_observations - 1)

        return folded_flows


def build_flow_network(config: Configuration):
    return FlowNetwork(
        scale=config["scale"])
