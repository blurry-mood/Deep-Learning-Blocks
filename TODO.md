
## What's next:
* InputAware Layer:
  - [ ] Add the bias to the customized conv2d
  - [ ] Support a tuple value for the kernel size

* Bank of kernels Layer:
  - [ ] Use neural turing machine to save kernels

* Loss funcs:
  - [ ] Inverse sigmoid
  - [ ] sinus function
  
* Object detector network:
  - [ ] Implement an object detector where the label is a mask with white object boundaries
  - [ ] Implement a method to separate the boxes from the mask
  - [ ] Train the model on multiple tasks: label containing the whole object, convolved object...
  
* Benchmark components:
  - [ ] Study how to benchmark model's (layer, block, network) performance: speed, accuracy, memory effeciency
  - [ ] Alter the notebooks folder with a benchmarking one

## Update package in Pypi:
> python3 setup.py sdist bdist_wheel
> twine upload  dist/*