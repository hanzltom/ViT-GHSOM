import numpy as np
from help_functions import *

class GSOM:
    def __init__(self, input_dim, t1, iteration_num, parent_qe, stopping_condition, learning_rate, sigma,
                 calculate_distance_func, neighbourhood_func, calculate_decay, beta, initial_weights = None):
        self.input_dim = input_dim
        self.t1 = t1
        self.time = 1
        self.current_row_num = 2
        self.current_col_num = 2
        self.iteration_num = iteration_num
        self.horizontal_grow_condition = t1 * parent_qe
        self.vertical_grow_condition = stopping_condition
        self.learning_rate = learning_rate
        self.sigma = sigma

        if initial_weights is not None:
            self.weights = initial_weights
        else:
            self.weights = np.random.rand(self.current_row_num, self.current_col_num, self.input_dim)

        self.child_maps = {}
        self.mapped_data = {}

        # functions from GHSOM, not needed to check
        self.calculate_distance_func = calculate_distance_func
        self.neighbourhood_func = neighbourhood_func
        self.calculate_decay = calculate_decay
        self.beta = beta

