# Genetic Algorithm Trainer (GA)
Typically, deep learning practioners train all layers in the model at the same time, for a number of epochs.  
The GA training strategy differs in which layers should be trainable and which ones to freeze at every epoch (or a predefined number of epochs). A question arises: which layers to freeze at each epoch?   
This is where the GA comes into play.

### **Algorithm**:
In every epoch:  
* a certain number of distinct individuals (solutions)  are evaluated (by training each freezed model, and estimating its test loss) based on their fitnesses (inverse of test loss),
* then, the resulting model, which is yielded by the most fit individual, is stored for later use,
* ater that, the selection, crossover, and mutation operators are applied to the population,
* finally, the stored model is then passed to all individuals to train it in the next iteration.

> * Here, a **solution** could be seen as a sequence of boolean values: `True`, means that layer is trainable, otherwise, it's  `False`.
  


### **How to use**:
1- Import GA trainer:
> from deepblocks.trainer import GA

2- Create an instance:
>  ga_trainer = GA(Model, LitModel, model:nn.Module, modules:List[str], dm=None, n_individuals=10, loss:str='test_loss', **kwargs)
* `Model`: a Python class extending `nn.Module` which is used to create a PyTorch model. The constructor `__init__` must have the option of creating an object without passing any arguments.
* `LitModel`: a Python class extending `pl.LightningModule` which is used to create a litModel. The constructor should enable the possibility of creating a litModel by passing only an instance of the aforementioned class.
* `model`: an instance of `Model`.
* `modules`: A list of strings representing the sub-modules that can be freezed. Sub-sub-sub... modules could be accessed by join their names with a dot (.), for e.g `mod1.block2.layer5`.
* `dm`: a Lightning data module object, it should contain both `train_dataloader` and `test_dataloader`.
* `n_individuals`: the number of individuals in the population. Note that it shouldn't exceed the total number of possibilities `2**len(modules)`. Also, this number must be divisible by 4.
* `loss`: a string referring to the logged metric storing the test loss value.
* `kwargs`: a sequence of arguments that are passed to pl.Trainer class to instantiate the trainer. 
Preferably, it should include `max_epochs` which refers to the number epochs each model is trained before estimating its loss. Also, `weights_summary=None, progress_bar_refresh_rate=0` should be passed to avoid printing results of intermediate runs (things get messy).

3- Warmup: 
> ga.warmup(epochs)
* `epochs`: an integer referring for the number of epochs to train the whole layers.

4- Run training strategy:
> model = ga.run(n_generations)
* `n_generations`: an integer, the number of generations for the GA algorithm.
* `model`: the best resulting PyTorch model minimizing the test loss. Note that sub-modules, which are mentioned in `modules` argument, of the model are all trainable.


### **Remarks**:
* Currently, the GA trainer doesn't preserve the state of layers not mentioned in `modules`.
* 
