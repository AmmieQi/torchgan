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

    def test_residual_block2d(self):
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

        layer = ResidualBlock2d(
            [3, 16, 3],
            [1, 1],
        )

        self.match_layer_outputs(layer, input, (16, 3, 10, 10))

    def test_transposed_residula_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = ResidualBlockTranspose2d([3, 16, 32, 3], [3, 3, 1], paddings=[1, 1, 0])

        self.match_layer_outputs(layer, input, (16, 3, 10, 10))

        layer = ResidualBlockTranspose2d(
            [3, 16, 32, 1],
            [3, 3, 1],
            paddings=[1, 1, 0],
            shortcut=torch.nn.Conv2d(3, 1, 3, padding=1),
        )

        self.match_layer_outputs(layer, input, (16, 1, 10, 10))

        layer = ResidualBlockTranspose2d(
            [3, 16, 3],
            [1, 1],
        )

        self.match_layer_outputs(layer, input, (16, 3, 10, 10))

    def test_basic_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = BasicBlock2d(3, 13, 3, 1, 1)

        self.match_layer_outputs(layer, input, (16, 16, 10, 10))

    def test_bottleneck_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = BottleneckBlock2d(3, 13, 3, 1, 1)

        self.match_layer_outputs(layer, input, (16, 16, 10, 10))

    def test_transition_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = TransitionBlock2d(3, 16, 3, 1, 1)

        self.match_layer_outputs(layer, input, (16, 16, 10, 10))

    def test_transition_block_transpose2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = TransitionBlockTranspose2d(3, 16, 3, 1, 1)

        self.match_layer_outputs(layer, input, (16, 16, 10, 10))

    def test_dense_block2d(self):
        input = torch.rand(16, 3, 10, 10)

        layer = DenseBlock2d(5, 3, 16, BottleneckBlock2d, 3, padding=1)

        self.match_layer_outputs(layer, input, (16, 83, 10, 10))

