import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from tqdm.notebook import tqdm 

from typing import List, Optional, ClassVar
from functools import reduce


## Global variables 
_Model = None
_LitModel = None
_loss = ''
_modules = []

class Individual:

    def __init__(self, solution:Optional[torch.BoolTensor]=None):
        # Generate a random solution
        self.solution = torch.rand(len(_modules)) < .5 if solution is None else solution

    def mutate(self, proba=.1):
        # Switch the bit of certain modules
        mask = torch.rand(*self.solution.shape) < proba
        for i in range(mask.size(0)):
            if mask[i]:
                self.solution[i] = self.solution[i].logical_not()

    def crossover(self, other,):
        # 
        n = self.solution.size(0)
        pt = torch.randint(low=1, high=n, size=(1,))
        
        def offspring(obj1, obj2,):
            solution = torch.hstack([obj1.solution[:pt], obj2.solution[pt:]])
            return Individual(solution)

        return offspring(self, other,), offspring(other, self, )

    def _disable_layers(self, model):
        for sol, mod in zip(self.solution.tolist(), _modules):
            # Reach out that specific layer
            mod = [model] + mod.split('.')
            mod = reduce(lambda a, b : getattr(a, b), mod)

            # Disable/Enable the layer
            for param in mod.parameters():
                param.requires_grad = sol

    def save_model(self, model, id=None):
        self.path = f'individual {id}.pth'
        torch.save(model.state_dict(), self.path)
        del model
    
    def load_model(self):
        model = _Model()
        model.load_state_dict(torch.load(self.path))
        return model, _LitModel(model,)

    def fitness(self, trainer, dm):
        # Load model to RAM
        _, litmodel = self.load_model()

        # Set requires_grad=False wherever it needs to be
        self._disable_layers(litmodel.model)
        
        # Train & estimate performance
        trainer.fit(litmodel, dm)
        trainer.test(litmodel, verbose=False)

        # Save the model
        self.save_model(litmodel.model)

        # Return the fitness = 1 / (loss + eps)
        return 1/(trainer.logged_metrics[_loss] + 1e-6)

#####################################################################################################################################################

class GA:

    

    def __init__(self, Model, LitModel, model:nn.Module, modules:List[str], dm=None, n_individuals=10, loss:str='test_loss', **kwargs):
        
        assert n_individuals%4==0, 'The number of individuals must be divisible by 4.'
        assert n_individuals<=2**len(modules), f'The number of {n_individuals} can\'t exceed the number of all possibilities {2**len(modules)}'

        global _modules, _loss, _Model, _LitModel

        _modules = modules
        _loss = loss
        _Model = Model
        _LitModel = LitModel

        self.model = model
        self.trainer = Trainer(**kwargs)
        self.dm = dm

        self.individuals = []
        for i in range(n_individuals):
            self.individuals.append(Individual())
    
    def warmup(self, epochs):
        _copy = int(self.trainer.max_epochs)*1
        
        # Make all model layers trainable
        for param in self.model.parameters():
            param.requires_grad = True

        self.trainer.fit(_LitModel(self.model), self.dm)
        self.trainer.max_epochs = _copy

    def _mutate(self, pb=.1):
        for c in self.individuals:
            c.mutate(pb)

    def _save_models(self):
        for i, c in enumerate(self.individuals):
            c.save_model(self.model, i)

    def _evaluate(self):
        fitnesses = []
        with tqdm(total=len(self.individuals), leave=False) as pbar:
            for c in self.individuals:
                fitness = c.fitness(self.trainer, self.dm)
                fitnesses.append(fitness)
                pbar.set_postfix({'test loss': 1/fitness.item()-1e-6})
                pbar.update(1)

        return fitnesses

    def _roulette_wheel(self, fitnesses):
        fits = fitnesses/fitnesses.sum()
        n = fits.size(0)

        return np.random.choice(list(range(n)), size=n//2, replace=False, p=fits.numpy())
    
    def _selection_crossover(self, fitnesses):
        chosen = self._roulette_wheel(torch.tensor(fitnesses))
        population = []
        for i in range(0, len(chosen), 2):
            p1, p2 = chosen[i], chosen[i+1]
            off1, off2 = self.individuals[p1].crossover(self.individuals[p2])
            population.extend([off1, off2, self.individuals[p1], self.individuals[p2]])

        self.individuals = population

    def run(self, n_generations):
        with tqdm(total=n_generations) as pbar:
            for _ in range(n_generations):
                # Save model
                self._save_models()

                # Train/fitness
                fitnesses  = self._evaluate()

                # Retrieve and dispatch best model to others
                id = torch.tensor(fitnesses).argmax()
                best_fitness = fitnesses[id]
                self.model, _ = self.individuals[id].load_model()

                # Selection & Crossover
                self._selection_crossover(fitnesses)

                # Mutate
                self._mutate()

                # Replace duplicates
                n = 1
                while n > 0:
                    n = len(self.individuals)
                    set_ = set(map(lambda x: tuple(x.solution.tolist()), self.individuals))
                    n = n - len(set_)
                    self.individuals = [Individual(solution=torch.BoolTensor(x)) for x in set_]
                    for _ in range(n):
                        self.individuals.append(Individual())

                pbar.update(1)
                pbar.set_postfix({'Best loss': 1/best_fitness.item() - 1e-6})
