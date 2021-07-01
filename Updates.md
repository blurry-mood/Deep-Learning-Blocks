## July 1, 2021:
* Implement & document **GA** Trainer.

## June 28, 2021:
*  Implement & document **InverseSigmoid** loss.
*  Implement & document **Cosine** loss.

## June 27, 2021:
* Tuple value for 'kernel_size' and bias term for the last conv2d in IA layer are now supported.

## June 26, 2021:
* Alter import statements in deepblocks/__init__.py
* Add **Squeeze-Excitation** block, with its tests and documentation.

## June 25, 2021:
* Add **FReLU** layer, with its tests and documentation.
* Add **FlipConv** layer, with its tests and documentation.
 
## June 21, 2021:
* Add unit test of the **InputAware** layer.
* Document this layer.

## June 20, 2021:
* Implement the **InputAware** layer, without the use of the bias in the last conv2d layer in ConvBlock.
* Add a notebook containing a performance test of the InputAware layer on MNIST against a vanilla CNN.