## What's next:
* Bank of kernels Layer:
  - [ ] Use neural turing machine to save kernels

* Loss funcs:
  - [ ] Inverse sigmoid
  - [ ] sinus function
  
* Object detector network:
  - [ ] Implement an object detector where the label is a mask with white object boundaries
  - [ ] Implement a method to separate the boxes from the mask
  - [ ] Train the model on multiple tasks: label containing the whole object, convolved object...

* Training framework:
  - [ ] Implement a class using a GA algorithm that trains a network by training some of its sub-modules at each epoch:
      - [ ] An individual: sequence of booleans (whether or not to make the corresponding layer trainable)
      - [ ] Mating, crossover, mutation: Implement the standard GA operators.
      - [ ] Fitness: inverse (or minus) the *VALIDATION* loss.
      - [ ] 1 Generation == 1 Epoch.
      - [ ] Benchmark by comparing this training strategy with standard training by preserving the same model.

## Update package in Pypi:
> python3 setup.py sdist bdist_wheel
> twine upload  dist/*