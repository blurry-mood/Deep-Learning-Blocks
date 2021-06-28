''' Test the initialization, forward and backward propagation of 
    both Hidden & InputAware.
'''
from deepblocks import InputAware
import torch
from torch.optim import Adam

import unittest

class TestConvBlock(unittest.TestCase):

    def test_forward(self):
        N = 2
        for in_channels, *_ in torch.randint(low=1, high=10, size=(N, 3)).tolist():
            for out_channels, mid_channels, kernel_size in torch.randint(low=1, high=10, size=(N, 3)).tolist():
                print(in_channels, out_channels, mid_channels, kernel_size,)
                # Predict the value of a tensor
                conv = InputAware(in_channels, out_channels,
                                   mid_channels, .1, kernel_size, padding=kernel_size//2)
                test = torch.rand(4, in_channels, 30, 30)
                y = conv(test)

                # Validate the output shape
                self.assertEqual(
                    tuple(y.shape), (4, out_channels, 30, 30))
                del test, y, conv

    def test_backward(self):
        N = 1
        for in_channels, width, height in torch.randint(low = 1, high=50, size=(N,3)).tolist():
            for out_channels, mid_channels, kernel_size in torch.randint(low = 1, high=50, size=(N,3)).tolist():

                # Init model & optimizer
                hidden = InputAware(in_channels, out_channels, mid_channels, .1, kernel_size)
                opt = Adam(hidden.parameters(), lr=1e-4)

                # Predict the value of one tensor
                test = torch.rand(4, in_channels, height, width)
                y = hidden(test)

                # Backpropagate one step
                cost = torch.sum(torch.abs(y))
                cost.backward()
                opt.step()

                del test, y, hidden, cost, opt


if __name__ == '__main__':
    unittest.main()
