## What's next:
* Bank of kernels Layer:
  - [ ] Use neural turing machine to save kernels
  
* Object detector network:
  - [ ] Implement an object detector where the label is a mask with white object boundaries
  - [ ] Implement a method to separate the boxes from the mask
  - [ ] Train the model on multiple tasks: label containing the whole object, convolved object...

## Update package in Pypi:
> python3 setup.py sdist bdist_wheel
> twine upload  dist/*