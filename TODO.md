## What's next:

* Benchmarks:
  - [ ] Compare the GA trainer against the standard strategy on the same model and dataset: time, accuracy, memory.
  - [ ] Compare the CosineLoss & InverseSigmoidLoss against CrossEntropy on the same model & dataset: convergence rate, final training & test losses. 
  - [ ] Compare FlipConv2d layer against vanilla nn.Conv2d on a dataset of flip-invariant images: convergence rate, and performance.


* Bank of kernels Layer:
  - [ ] Use neural turing machine to save kernels
  
* Object detector network:
  - [ ] Implement an object detector where the label is a mask with white object boundaries
  - [ ] Implement a method to separate the boxes from the mask
  - [ ] Train the model on multiple tasks: label containing the whole object, convolved object...

## Update package in Pypi:
> python3 setup.py sdist bdist_wheel
> twine upload  dist/*