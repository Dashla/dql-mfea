import kerpy
from Environment import Env_class
import numpy as np
import pickle


def compute_indexes(dqltasks):
    dims = [i.layer_dims for i in dqltasks]
    dims = np.transpose(dims)
    indexes = [0]

    # Get the maximum sizes among dimensions
    for dim in dims:
        indexes.append(max(dim)+indexes[-1])

    # Assign layer start points
    for i, dql in enumerate(dqltasks):
        dql.layer_start_points = indexes

    # Compute last layers start and end indexes
    last = indexes[-2]
    for dql in dqltasks:
        next_last = last+dql.layer_dims[-1]
        dql.last_layer_indexes = (last, next_last)
        last = next_last


class DQLTask(Env_class):
    # ----------
    def __init__(self, environment_name, test_config=None):
        # Crear modelo basado en el environment
        Env_class.__init__(self, environment_name)

        # Crear modelo, shapes
        analyzed = kerpy.analyse_model(self.model)
        self.shapes = analyzed[0]
        self.mask = analyzed[1]
        self.weights = kerpy.flatten(analyzed[2])
        self.layer_dims = analyzed[3]

        # Configure
        self.layer_start_points = None
        self.last_layer_indexes = None
        self.best = None
        self.evhistory = []
        self.D = sum(self.layer_dims[:-1])
        self.episodes = 10
        self.max_steps = 300
        self.verbose = 1
        self.visualize = False

        if test_config is not None:
            self.episodes = test_config['episodes']
            self.max_steps = test_config['max_steps']
            self.verbose = test_config['verbose']
            self.visualize = test_config['visualize']

    def set_best(self, name, folder_name=None):
        name = name+'_weights.h5f'
        if folder_name:
            name = folder_name+'/'+name
        self.model.load_weights(name)

    # ----------
    def save_processed_candidate(self, candidate, name='', folder_name=None):
        c = np.split(candidate, self.layer_start_points[1:-1])
        p = [c[i][:x] for i, x in enumerate(self.layer_dims)]
        ncandidate = kerpy.flatten(p)
        if folder_name:
            name = folder_name+'/'+name
        pickle.dump(ncandidate, open(name+'_candidate.pickle', 'wb'))
        weights = kerpy.unravel(ncandidate, self.shapes)
        kerpy.set_weights(self.model, weights, self.mask)
        self.model.save_weights(name + '_weights.h5f')

    def save_processed_candidate_1(self, candidate, name='', folder_name=None):
        cortado = np.split(candidate, self.layer_start_points[1:-1])
        pesitos = [cortado[i][:x] for i, x in enumerate(self.layer_dims)][:-1]
        start, end = self.last_layer_indexes
        pesitos.append(candidate[start:end])
        ncandidate = kerpy.flatten(pesitos)
        if folder_name:
            name = folder_name+'/'+name
        pickle.dump(ncandidate, open(name+'_candidate.pickle', 'wb'))
        weights = kerpy.unravel(ncandidate, self.shapes)
        kerpy.set_weights(self.model, weights, self.mask)
        self.model.save_weights(name + '_weights.h5f')

    # ----------
    def fnc(self, candidate):
        # Usar kerpy para recomponer la red
        cortado = np.split(candidate, self.layer_start_points[1:-1])
        pesitos = [cortado[i][:x] for i, x in enumerate(self.layer_dims)][:-1]
        start, end = self.last_layer_indexes
        pesitos.append(candidate[start:end])
        ncandidate = kerpy.flatten(pesitos)
        weights = kerpy.unravel(ncandidate, self.shapes)
        kerpy.set_weights(self.model, weights, self.mask)

        # Evaluate model
        hist = self.test(self.episodes,
                         self.max_steps,
                         enable_visual=self.visualize,
                         verbose=self.verbose)

        # Update fitness
        fitness = -np.mean(hist['episode_reward'])
        if self.best is None or self.best[1] > fitness:
            self.best = (ncandidate, fitness)
        self.evhistory.append(-self.best[1])
        #print(fitness, self.best[1])

        return fitness
