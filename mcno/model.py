import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator


class AdaptiveMCConv1d(nn.Module):
    """Monte Carlo-based kernel integration layer.

    Approximates the kernel integral by sampling a fixed set of random
    grid points and learning a kernel tensor over those points.
    """

    def __init__(self, in_channels, out_channels, num_samples, grid_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.num_samples = num_samples

        self.coeff_nn = nn.Parameter(torch.empty(in_channels, out_channels, num_samples))
        nn.init.xavier_uniform_(self.coeff_nn)
        with torch.no_grad():
            self.coeff_nn.copy_(F.normalize(self.coeff_nn, p=2, dim=-1))

        self.register_buffer(
            "indices", torch.randint(0, self.grid_size, (self.num_samples,))
        )

    def forward(self, x):
        batch_size = x.shape[0]
        adaptive_samples = self.indices

        adaptive_samples_expanded = (
            adaptive_samples.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, self.in_channels, self.num_samples)
        )
        x_at_samples = torch.gather(x, dim=-1, index=adaptive_samples_expanded)
        x_at_samples2 = x_at_samples.permute(0, 2, 1)

        kernel_values = torch.einsum("bni,ion->bno", x_at_samples2, self.coeff_nn)
        adaptive_contributions = torch.bmm(x_at_samples, kernel_values)

        adaptive_contributions_resampled = F.interpolate(
            adaptive_contributions,
            size=self.grid_size,
            mode="linear",
            align_corners=False,
        )

        return adaptive_contributions_resampled / self.num_samples


class SimpleBlock1d(nn.Module):
    """A single MCNO block with four MC convolution layers."""

    def __init__(self, width, grid_size, num_samples):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = AdaptiveMCConv1d(width, width, num_samples, grid_size)
        self.conv1 = AdaptiveMCConv1d(width, width, num_samples, grid_size)
        self.conv2 = AdaptiveMCConv1d(width, width, num_samples, grid_size)
        self.conv3 = AdaptiveMCConv1d(width, width, num_samples, grid_size)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.relu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.relu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.relu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.relu(x1 + x2)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net1d(nn.Module):
    """MCNO network for 1D operator learning."""

    def __init__(self, width, grid_size, num_samples):
        super().__init__()
        self.conv1 = SimpleBlock1d(width, grid_size, num_samples)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze(-1)

    def count_params(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())
