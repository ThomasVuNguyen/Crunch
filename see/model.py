import torch
import torch.nn as nn

from config import HIDDEN_DIMS, IMG_SIZE


class SensorToImageMLP(nn.Module):
    """
    Maps 3 sensor inputs to 64x64x3 RGB image using MLP.

    Input: [batch, 3] - normalized sensor values (distance, pir, sound)
    Output: [batch, 3, 64, 64] - RGB image in range [0, 1]
    """

    def __init__(self, hidden_dims=None, img_size=IMG_SIZE):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        self.img_size = img_size
        output_size = img_size * img_size * 3

        layers = []
        in_dim = 3  # 3 sensor inputs

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = h_dim

        # Final layer to image size
        layers.append(nn.Linear(hidden_dims[-1], output_size))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, 3]
        out = self.net(x)  # [batch, img_size*img_size*3]
        out = out.view(-1, 3, self.img_size, self.img_size)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
