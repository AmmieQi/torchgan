import os
import sys
import unittest

import torch
from torchgan.layers import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


class TestLayers(unittest.TestCase):
    def match_layer_outputs(self, layer, input, output_shape):
        z = layer(input)

        self.assertEqual(z.shape, output_shape)

    def test_residual_block(self):
        input = torch.rand(16, 3, 10, 10)

        layer = ResidualBlock2d([3, 16, 32, 3], [3, 3, 1], paddings=[1, 1, 0])

        self.match_layer_outputs(layer, input, (16, 3, 10, 10))

        layer = ResidualBlock2d(
            [3, 16, 32, 1],
            [3, 3, 1],
            paddings=[1, 1, 0],
            shortcut=torch.nn.Conv2d(3, 1, 3, padding=1),
        )

        self.match_layer_outputs(layer, input, (16, 1, 10, 10))
