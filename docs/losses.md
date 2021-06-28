## Inverse Sigmoid loss
* This function is the inverse of the sigmoid function.

### **Formula**:
Given a class probability to be maximized p, the loss is calculated as follows:  

$loss = log(\frac{1}{\alpha p + \beta}-1) + \gamma$  
* log: is the natural logarithm of a number.
* $\alpha, \beta$: two positive floats that control the slope of the gradient around p=0 and p=1, such as:
  * $\alpha, \beta>0$
  * $\alpha + \beta<1$
* $\gamma$: a float that keeps the loss always positive, with $loss(p=1)=0$. It is calculated by the class, no need to specify it by the user.  

The following two figures show the loss function for different values of $alpha & beta$.
> * y-axis: the loss.
> * x-axis: the probability.

| | | |
------------------------- | :-----------: |-------------------------
![](/docs/imgs/InvSig_plot2.png) | ---------------- |  ![](/docs/imgs/InvSig_plot3.png)

### **How to use**:
1- Import Inv. Sig. Loss:
> from deepblocks.loss import InverseSigmoid

2- Create an instance:
>  is_loss = InverseSigmoid(alpha, beta, reduction)
* `alpha, beta`: Two floats satisfying the constraints described above.
* `reduction`: A string specifying the reduction to apply to the output: 'none' | 'mean' | 'sum'.

3- Compute the loss:
> loss = is_loss(y_hat, y)
* `y_hat`: A Float tensor with a shape (batch, C, d1, d2, ..., dK), C: number of classes. This tensor shouldn't be a Softmax's output.
* `y`: A Long tensor of indices, with a shape (batch, d1, d2, ..., dK), where each entry is non-negative and smaller than C.
* `loss`: A Float tensor with the same shape as `y` if `reduction='none'`, otherwise the output tensor is a scalar tensor.

