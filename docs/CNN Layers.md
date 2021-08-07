
# **Funnel ReLU (FReLU)**
This layer takes an input tensor I = (B, C, H, W), applies a spatial transformation O = (B, C, H, W), then returns the maximum value between I and O in an element-wise way.

### **Formula**:
$f(x_{c,i,j}) = max(x_{c,i,j}, T(x_{c,i,j}))$
* $T(x_{c,i,j})$ is the convolution operator with a kernel *p* applied to a window centered on $x_{c,i,j}$.   
Please note that *p* a window of *learnable* parameters, and it is shared in the same channel.

### **How to use**:
1- Import FReLU:
> from deepblocks import FReLU

2- Create an instance:
>  frelu = FReLU(in_channels, kernel_size=3)
* `in_channels`: The input channels.
* `kernel_size`: The dimension of the kernel *p*. It could be an integer or a tuple of two integers.   
  Note that the dimensions must be odd numbers.

### **Remarks**:
* For more insights, check the original paper: [Funnel Activation for Visual Recognition
](https://arxiv.org/abs/2007.11824)


# **Flip-Invariant Conv2d Layer**
This layer takes an input tensor (B, C, H, W) and applies a conv2d operator using special kernels.  
These kernels are horizontally and/or vertically symmetric.

### **Architecture**:
* Typically, a 2d convolution kernel is n by m matrix (in the image it's 3 by 7), where each entry is an independant variable, as depicted in the image below:  

![Flip-Invariant Conv2d Layer](/docs/imgs/FlipConv_std.png "Standard Conv2d kernel.")

* This new conv2d operator enables using kernels with entries repeated in the right and bottom parts of the kernel matrix (as defined by the user).  
To get a good gist of this kernel, let's ponder for a minute on the two images below.  
For the first one, the kernel matrix is symmetric with respect to the red column, it's horizontally symmetric.  
For the second image, the matrix is both horizontally and vertically symmetric.  

![Flip-Invariant Conv2d Layer](/docs/imgs/FlipConv_h.png "Horizontal flip-invariant Conv2d kernel.")  
![Flip-Invariant Conv2d Layer](/docs/imgs/FlipConv_vh.png "Horizontal and vertical flip-invariant Conv2d kernel.")
* This kernel structure enables making the layer *"almost"* invariant with respect to flipping. The resutling feature maps, after applying this type of convolution on different flipped tensors, are the same if each one is flipped in a certain way (for e.g by making the corner with lowest value in each channel go to the top left corner, all the feature maps are equal). In contrast, the standard convolution doesn't guarantee this property.


### **How to use**:
1- Import ConvBlock:
> from deepblocks import FlipConv2d

2- Create an instance:
>  flip_conv = FlipConv2d(h_invariant: bool = True, v_invariant: bool = True, **kwargs)
* `h_invariant`: If true, the kernels are horizonatally symmetrical.
* `v_invariant`: If true, the kernels are vertically symmetrical.
* `**kwargs`: Arguments used for the standard nn.Conv2d layer.
  
### **Remarks**:
* This layer is useful when the input image of a model is always is invariant to flipping.
* For `h_invariant == v_invariant == false`, this layer behaves as a standard conv2d layer.
* Depending on the values of `h_invariant & v_invariant`, this layer reduces the number of parameters by almost 4 times.


# **Squeeze-Excitation Block**
A block that independently scales channels of an input feature map.  
The library provides only the Inception-SE module. 
### **Architecture**:
The image below depicts the architecture of SE block as described in the orignal paper (link in *Remarks*).  

![Squeeze-Excitation Block](/docs/imgs/SE_block.png "Squeeze-Excitation Block")

### **How to use**:
1- Import ConvBlock:
> from deepblocks import SE

2- Create an instance:
>  se = SE(in_channels: int, ratio: int = 16)
* `in_channels`: Number of channels of the input tensor.
* `ratio`: an positive integer (must be lower than *in_channels*) that control the complexity of the SE block.
  
### **Remarks**:
* For more insights, check the original paper: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)