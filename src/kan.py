"""
src/kan.py
Custom PyTorch implementation of Kolmogorov-Arnold Network (KAN) Layer.
Based on the Kolmogorov-Arnold representation theorem (Liu et al., MIT 2024).
Replaces linear weights with learnable B-spline activation functions on edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 1. Grid of control points (pins) spanning from -1 to 1
        h = 2.0 / grid_size
        grid = torch.linspace(-1.0 - h * spline_order, 1.0 + h * spline_order, grid_size + 2 * spline_order + 1)
        # Shape: (in_features, num_grid_points)
        self.register_buffer("grid", grid.unsqueeze(0).expand(in_features, -1))

        # 2. Backbone weights (linear transformation with SiLU activation)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 3. Spline weights (the learnable heights of the spline pins)
        num_spline_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, num_spline_bases))
        
        # Initialize weights with small random numbers
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=1.0)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def compute_b_splines(self, x):
        """
        Computes the B-spline basis functions using the Cox-de Boor recurrence relation.
        x shape: (batch_size, in_features)
        Returns: (batch_size, in_features, num_spline_bases)
        """
        # Ensure x is in range [-1, 1] for stable spline evaluation
        x = torch.tanh(x).unsqueeze(-1)  # (batch, in_features, 1)
        grid = self.grid.unsqueeze(0)    # (1, in_features, num_grid)

        # Order 0 basis: 1 if inside the interval, 0 otherwise
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)

        # Cox-de Boor recurrence up to spline_order
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :, :-(k + 1)]
            left_den = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            left = (left_num / (left_den + 1e-8)) * bases[:, :, :-1]

            right_num = grid[:, :, k + 1:] - x
            right_den = grid[:, :, k + 1:] - grid[:, :, 1:-k]
            right = (right_num / (right_den + 1e-8)) * bases[:, :, 1:]

            bases = left + right

        return bases  # (batch_size, in_features, num_spline_bases)

    def forward(self, x):
        """
        Forward pass: y = Backbone(x) + Spline(x)
        """
        # Part 1: Backbone transformation: SiLU(x) @ base_weight.T
        base_output = F.linear(F.silu(x), self.base_weight)

        # Part 2: Spline transformation: bases * spline_weight
        # bases: (batch, in_features, num_spline_bases)
        bases = self.compute_b_splines(x)
        
        # Perform tensor contraction: sum over in_features and spline_bases
        # (batch, in_features, num_spline_bases) x (out_features, in_features, num_spline_bases)
        spline_output = torch.einsum("bik,oik->bo", bases, self.spline_weight)

        return base_output + spline_output

