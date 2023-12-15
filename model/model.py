from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torchdiffeq import odeint
from tqdm import tqdm

from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper
from model.vector_field_regressor import build_vector_field_regressor
from model.vqgan.taming.autoencoder import vq_f8_ddconfig, vq_f8_small_ddconfig, vq_f16_ddconfig, VQModelInterface
from model.vqgan.vqvae import build_vqvae
from model.flow import (
    build_flow_network,
    build_flow_representation_network,
    build_sparsification_network
)


class Model(nn.Module):
    def __init__(self, config: Configuration):
        super(Model, self).__init__()

        self.config = config
        self.sigma = config["sigma"]
        self.skip_prob = config["skip_prob"]

        if config["autoencoder"]["type"] == "ours":
            self.ae = build_vqvae(
                config=config["autoencoder"],
                convert_to_sequence=True)
            self.ae.backbone.load_from_ckpt(config["autoencoder"]["ckpt_path"])
        else:
            if config["autoencoder"]["config"] == "f8":
                ae_config = vq_f8_ddconfig
            elif config["autoencoder"]["config"] == "f8_small":
                ae_config = vq_f8_small_ddconfig
            else:
                ae_config = vq_f16_ddconfig
            self.ae = VQModelInterface(ae_config, config["autoencoder"]["ckpt_path"])

        self.vector_field_regressor = build_vector_field_regressor(
            config=self.config["vector_field_regressor"])

        self.flow_network = build_flow_network(
            config=self.config["flow_network"])

        self.flow_representation_network = build_flow_representation_network(
            config=self.config["flow_representation_network"],
            convert_to_sequence=True)

        self.sparsification_network = build_sparsification_network(
            config=self.config["sparsification_network"],
            convert_to_sequence=True)

    def load_from_ckpt(self, ckpt_path: str):
        loaded_state = torch.load(ckpt_path, map_location="cpu")

        is_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_ddp = True
                break
        if is_ddp:
            state = {k.replace("module.", ""): v for k, v in loaded_state["model"].items()}
        else:
            state = {f"module.{k}": v for k, v in loaded_state["model"].items()}

        dmodel = self.module if isinstance(self, torch.nn.parallel.DistributedDataParallel) else self
        dmodel.load_state_dict(state)

    def forward(
            self,
            observations: torch.Tensor,
            flows: torch.Tensor) -> DictWrapper[str, Any]:
        """

        :param observations: [b, num_observations, num_channels, height, width]
        :param flows: [b, num_observations - 1, 2, height, width]
        """

        batch_size = observations.size(0)
        num_observations = observations.size(1)
        assert num_observations > 2

        # Sample target frames and conditioning
        target_frames_indices = torch.randint(low=2, high=num_observations, size=[batch_size])
        target_frames = observations[torch.arange(batch_size), target_frames_indices]
        reference_frames_indices = target_frames_indices - 1
        reference_frames = observations[torch.arange(batch_size), reference_frames_indices]
        conditioning_frames_indices = torch.cat(
            [torch.randint(low=0, high=s - 1, size=[1]) for s in target_frames_indices], dim=0)
        conditioning_frames = observations[torch.arange(batch_size), conditioning_frames_indices]

        # Compute flows
        if flows is None:
            flows = self.flow_network(torch.stack([reference_frames, target_frames], dim=1))
        else:
            flows = flows[torch.arange(batch_size), reference_frames_indices].unsqueeze(1)

        # Sparsify flows
        sparsification_results = self.sparsification_network(flows)
        sparse_flows = sparsification_results.sparse_output

        # Encode flows
        sparse_flow_states = self.flow_representation_network(sparse_flows).squeeze(1)

        # Encode observations to latent codes
        with torch.no_grad():
            self.ae.eval()
            input_frames = torch.stack([target_frames, reference_frames, conditioning_frames], dim=1)
            if self.config["autoencoder"]["type"] == "ours":
                latents = self.ae(input_frames).latents
            else:
                flat_input_frames = rearrange(input_frames, "b n c h w -> (b n) c h w")
                flat_latents = self.ae.encode(flat_input_frames)
                latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=3)
        target_latents = latents[:, 0]
        reference_latents = latents[:, 1]
        conditioning_latents = latents[:, 2]

        # Sample input latents
        noise = torch.randn_like(target_latents).to(target_latents.dtype).to(target_latents.device)
        timestamps = torch.rand(batch_size, 1, 1, 1).to(target_latents.dtype).to(target_latents.device)
        input_latents = (1 - (1 - self.sigma) * timestamps) * noise + timestamps * target_latents

        # Calculate target vectors
        target_vectors = (target_latents - (1 - self.sigma) * input_latents) / (1 - (1 - self.sigma) * timestamps)

        # Calculate time distances
        index_distances = (reference_frames_indices - conditioning_frames_indices).to(input_latents.device)

        # Predict vectors
        reconstructed_vectors = self.vector_field_regressor(
            input_latents=input_latents,
            reference_latents=reference_latents,
            conditioning_latents=conditioning_latents,
            action_latents=sparse_flow_states,
            index_distances=index_distances,
            timestamps=timestamps.squeeze(3).squeeze(2).squeeze(1),
            skip_past=torch.rand(1)[0].item() < self.skip_prob,
            skip_action=torch.rand(1)[0].item() < self.skip_prob)

        return DictWrapper(
            # Inputs
            observations=observations,

            # Data for loss calculation
            reconstructed_vectors=reconstructed_vectors,
            target_vectors=target_vectors)

    @torch.no_grad()
    def calculate_sparse_flows(self, observations: torch.Tensor, flows: torch.Tensor, num_vectors: int = None):
        """
        :param observations: [b, num_observations, num_channels, height, width]
        :param flows: [b, num_observations - 1, 2, height, width]
        :param num_vectors: number of vectors to select
        :return: [b, num_observations - 1, 3, height, width]
        """

        # Compute flows
        if flows is None:
            flows = self.flow_network(observations)

        # Sparsify flows
        sparsification_results = self.sparsification_network(flows, num_vectors=num_vectors)
        sparse_flows = sparsification_results.sparse_output

        return sparse_flows

    @torch.no_grad()
    def generate_frames(
            self,
            observations: torch.Tensor,
            sparse_flows: torch.Tensor,
            num_frames: int = None,
            steps: int = 100,
            warm_start: float = 0.0,
            past_horizon: int = -1,
            skip_past: bool = False,
            verbose: bool = False) -> torch.Tensor:
        """
        Generates num_frames frames conditioned on observations

        :param observations: [b, 1, num_channels, height, width]
        :param sparse_flows: [b, n, 3, height, width], None for no actions
        :param num_frames: number of frames to generate
        :param warm_start: part of the integration path to jump to
        :param steps: number of steps for sampling
        :param past_horizon: number of frames to condition on
        :param skip_past: whether to consider past or not
        :param verbose: whether to display loading bar
        """

        # Encode observations to latents
        self.ae.eval()
        if self.config["autoencoder"]["type"] == "ours":
            latents = self.ae(observations).latents
        else:
            flat_input_frames = rearrange(observations, "b n c h w -> (b n) c h w")
            flat_latents = self.ae.encode(flat_input_frames)
            latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=observations.size(1))

        b, n, c, h, w = latents.shape

        # Encode flows
        if sparse_flows is not None:
            sparse_flow_states = self.flow_representation_network(sparse_flows)
            num_actions = sparse_flow_states.size(1)
        else:
            sparse_flow_states = torch.zeros(
                [b, 1, self.vector_field_regressor.action_state_size, 16, 16]).to(latents.device)
            num_actions = 0

        # Generate future latents
        gen = tqdm(range(num_frames), desc="Generating frames", disable=not verbose, leave=False)
        for i in gen:
            def f(t: torch.Tensor, y: torch.Tensor):
                lower_bound = 0 if past_horizon == -1 else max(0, latents.size(1) - past_horizon)
                if latents.size(1) == 1:
                    higher_bound = 1
                else:
                    higher_bound = latents.size(1) - 1

                # Sample conditioning and reference
                conditioning_latents_indices = torch.randint(low=lower_bound, high=higher_bound, size=[b])
                conditioning_latents = latents[torch.arange(b), conditioning_latents_indices]
                reference_latents = latents[:, -1]

                # Calculate index distances
                index_distances = (higher_bound - conditioning_latents_indices).to(y.device)

                # Calculate vectors
                vectors = self.vector_field_regressor(
                    input_latents=y,
                    reference_latents=reference_latents,
                    conditioning_latents=conditioning_latents,
                    action_latents=sparse_flow_states[:, i] if i < num_actions else sparse_flow_states[:, -1],
                    index_distances=index_distances,
                    timestamps=t * torch.ones(b).to(latents.device),
                    skip_past=skip_past or (n == 1 and i == 0),
                    skip_action=i >= num_actions)

                return vectors

            # Initialize with noise
            noise = torch.randn([b, c, h, w]).to(latents.device)
            y0 = (1 - (1 - self.sigma) * warm_start) * noise + warm_start * latents[:, -1]

            # Solve ODE
            next_latents = odeint(
                f,
                y0,
                t=torch.linspace(warm_start, 1, int((1 - warm_start) * steps)).to(y0.device),
                method="rk4"
            )[-1]
            latents = torch.cat([latents, next_latents.unsqueeze(1)], dim=1)

        # Close loading bar
        gen.close()

        # Decode to image space
        latents = rearrange(latents, "b n c h w -> (b n) c h w")
        if self.config["autoencoder"]["type"] == "ours":
            reconstructed_observations = self.ae.backbone.decode_from_latents(latents)
        else:
            reconstructed_observations = self.ae.decode(latents)
        reconstructed_observations = rearrange(reconstructed_observations, "(b n) c h w -> b n c h w", b=b)

        return reconstructed_observations
