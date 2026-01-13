from help_functions import *

class GHSOM:
    def __init__(self, input_dim, t1, t2, distance_k):
        self.t1 = t1
        self.t2 = t2
        self.input_dim = input_dim

        self.layer0_neuron = None
        self.qe0 = 0
        self.child_map = None

        if distance_k == np.inf:
            self.calculate_distance_func = chebyshev_distance
        elif distance_k == 1:
            self.calculate_distance_func = manhattan_distance
        elif distance_k == 2:
            self.calculate_distance_func = euclidean_distance
        elif distance_k < 1:
            raise ValueError('Distance must have positive non-zero k value')
        elif distance_k > 2:
            self.calculate_distance_func = lambda a, b, axis: generic_distance(a, b, axis, self.distance_k)
        else:
            raise ValueError('Distance k must be whole number between 1 and np.inf')

    def train(self, data):
        pass

