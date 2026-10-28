"""Learned metric kinds: channels grouped by their names, and a per-kind table of deviation sizes.

A fault type is a pattern across metric kinds (memory stress: the memory channels move, CPU does
not). Each channel goes to the nearest of K centroids of name embeddings (learned in training by
k-means over the training names, numbers collapsed, and shipped in the export), and its deviation
sizes are summarized per kind: max, mean, and whether the kind has a channel, once over every
channel and once over the channels the question quotes. An MLP adds the two tables to the fused
vector.
"""

import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SIZES = 2  # leading deviation-feature columns that measure how far a channel deviates
STATS = 2 * SIZES + 1  # per kind: max and mean of each size, and whether the kind has a channel
_NUMBER = re.compile(r"\d+")


def kind_pattern(name: str) -> str:
    """``sensor_6636_latency`` -> ``sensor_#_latency``: numbers in names are IDs, never kinds."""
    return _NUMBER.sub("#", name)


class KindSummary(nn.Module):
    """Per-kind deviation tables (all channels, quoted channels) -> a vector added to z."""

    def __init__(self, kinds: int, text_dim: int, hidden_dim: int):
        super().__init__()
        self.kinds = kinds
        self.register_buffer("centroids", torch.zeros(kinds, text_dim))  # normalized, from the export
        self.mlp = nn.Sequential(nn.Linear(2 * kinds * STATS, 128), nn.GELU(), nn.Linear(128, hidden_dim))

    def assign(self, embeddings: Tensor) -> Tensor:
        """The kind of each name, [...]: its nearest centroid."""
        flat = F.normalize(embeddings.float(), dim=-1).reshape(-1, embeddings.shape[-1])
        return torch.cdist(flat, self.centroids.float()).argmin(-1).reshape(embeddings.shape[:-1])

    def forward(self, embeddings: Tensor, features: Tensor, quoted: Tensor) -> Tensor:
        """``embeddings`` [B, C, d_txt] of the channels' patterns, ``features`` [B, C, F], ``quoted`` [B, C]."""
        sizes = features[..., :SIZES].float()
        member = F.one_hot(self.assign(embeddings), self.kinds).to(sizes)  # [B, C, K]
        tables = [self._table(member, sizes), self._table(member * quoted[..., None].to(sizes), sizes)]
        return self.mlp(torch.cat(tables, dim=-1).to(self.mlp[0].weight.dtype))

    def _table(self, member: Tensor, sizes: Tensor) -> Tensor:
        count = member.sum(1)  # [B, K]
        present = (count > 0).to(sizes)
        mean = torch.einsum("bck,bcs->bks", member, sizes) / count.clamp_min(1)[..., None]
        masked = sizes[:, :, None, :].masked_fill(member[..., None] == 0, float("-inf"))
        top = masked.amax(1).nan_to_num(neginf=0.0) * present[..., None]
        return torch.cat([top, mean, present[..., None]], dim=-1).flatten(1)  # [B, K * STATS]
