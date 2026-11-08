"""PyTorch architectures adapted from the MIT-licensed SelfOracle release.

See THIRD_PARTY_NOTICES and README.md for provenance and intentional differences.
"""

import torch
from torch import nn


class Reconstructor(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.kind = config["model"]
        size = config["height"] * config["width"] * 3
        if self.kind == "sae":
            self.encoder = nn.Linear(size, 64)
            self.decoder = nn.Linear(64, size)
        elif self.kind == "vae":
            self.encoder = nn.Linear(size, 512)
            self.mean = nn.Linear(512, 2)
            self.log_var = nn.Linear(512, 2)
            self.hidden_decoder = nn.Linear(2, 512)
            self.decoder = nn.Linear(512, size)
        else:
            raise ValueError(f"Unknown model: {self.kind}")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images):
        hidden = torch.relu(self.encoder(images))
        penalty = images.new_zeros(())
        if self.kind == "sae":
            penalty = 1e-8 * hidden.abs().sum(dim=1).mean()
        else:
            mean = self.mean(hidden)
            if self.training:
                latent = mean + torch.exp(0.5 * self.log_var(hidden)) * torch.randn_like(mean)
            else:
                # Deterministic posterior-mean scoring, shared by calibration and testing.
                latent = mean
            hidden = torch.relu(self.hidden_decoder(latent))
        return torch.sigmoid(self.decoder(hidden)), penalty

