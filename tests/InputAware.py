''' Test the initialization, forward and backward propagation of 
    both Hidden & ConvBlock.
'''
from deepblocks.CNN.InputAware import Hidden, ConvBlock
import torch
from torch.optim import Adam

import unittest

class TestHidden(unittest.TestCase):    

    def test_forward(self):
        N = 2
        for in_channels, width, height in torch.randint(low = 1, high=50, size=(N,3)).tolist():
            for out_channels, mid_channels, kernel_size in torch.randint(low = 1, high=50, size=(N,3)).tolist():
                
                # Predict the value of a tensor
                hidden = Hidden(in_channels, out_channels, mid_channels, .1, kernel_size)
                test = torch.rand(4, in_channels, height, width)
                y = hidden(test)
                
                # Validate the output shape
                self.assertEqual(tuple(y.shape), (4, out_channels,kernel_size,kernel_size))
                del test, y, hidden

    def test_backword(self):
        N = 2
        for in_channels, width, height in torch.randint(low = 1, high=50, size=(N,3)).tolist():
            for out_channels, mid_channels, kernel_size in torch.randint(low = 1, high=50, size=(N,3)).tolist():
                
                # Init model & optimizer
                hidden = Hidden(in_channels, out_channels, mid_channels, .1, kernel_size)
                opt = Adam(hidden.parameters(), lr=1e-4)
                
                # Predict the value of one tensor
                test = torch.rand(4, in_channels, height, width)
                y = hidden(test)

                # Backpropagate one step
                cost = torch.sum(torch.abs(y))
                cost.backward()
                opt.step()
                
                del test, y, hidden, cost, opt

class TestConvBlock(unittest.TestCase):    

    def test_forward(self):
        N = 2
        for in_channels, width, height in torch.randint(low = 1, high=50, size=(N,3)).tolist():
            for out_channels, mid_channels, kernel_size in torch.randint(low = 1, high=50, size=(N,3)).tolist():
                
                # Predict the value of a tensor
                hidden = ConvBlock(in_channels, out_channels, mid_channels, .1, kernel_size)
                test = torch.rand(4, in_channels, height, width)
                y = hidden(test)
                
                # Validate the output shape
                self.assertEqual(tuple(y.shape), (4, out_channels,kernel_size,kernel_size))
                del test, y, hidden

    def test_backword(self):
        N = 2
        for in_channels, width, height in torch.randint(low = 1, high=50, size=(N,3)).tolist():
            for out_channels, mid_channels, kernel_size in torch.randint(low = 1, high=50, size=(N,3)).tolist():
                
                # Init model & optimizer
                hidden = ConvBlock(in_channels, out_channels, mid_channels, .1, kernel_size)
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