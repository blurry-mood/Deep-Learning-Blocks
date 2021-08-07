## What's next:

* Adaptive Loss:
  - [ ] Use same principle of Focal loss to define a loss function that lowers the loss of easy examples by shifting their probability values to 1 & and even lower the proba of hard examples using weights summing up to 1.
  
* Object detector network:
  - [ ] Implement an object detector where the label is a mask with white object boundaries
  - [ ] Implement a method to separate the boxes from the mask
  - [ ] Train the model on multiple tasks: label containing the whole object, convolved object...