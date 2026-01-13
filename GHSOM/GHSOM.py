from help_functions import *
from GSOM import GSOM
from collections import deque

class GHSOM:
    def __init__(self, input_dim, t1, t2, training_epoch_num, learning_rate = 0.5,
                 distance_k = 2, neighbourhood_function = 'gaussian', decay_type='exponential',
                 beta=0.999, use_qe_for_vertical = True):

        self.input_dim = input_dim
        self.t1 = t1
        self.t2 = t2
        self.training_epoch_num = training_epoch_num
        self.learning_rate = learning_rate
        self.beta = beta

        self.layer0_weight = None
        self.global_stopping_criterion = 0 # vertical growth
        self.map_db = {}

        if distance_k == np.inf:
            self.calculate_distance_func = chebyshev_distance
        elif distance_k == 1:
            self.calculate_distance_func = manhattan_distance
        elif distance_k == 2:
            self.calculate_distance_func = euclidean_distance
        elif distance_k < 1:
            raise ValueError('Distance must have positive non-zero k value')
        elif distance_k > 2:
            self.calculate_distance_func = lambda a, b, axis: generic_distance(a, b, axis, distance_k)
        else:
            raise ValueError('Distance k must be whole number between 1 and np.inf')

        if neighbourhood_function == 'gaussian':
            self.neighbourhood_func = gaussian_neighbourhood
        elif neighbourhood_function == 'rectangular':
            self.neighbourhood_func = rectangular_neighbourhood
        elif neighbourhood_function == 'triangular':
            self.neighbourhood_func = triangular_neighbourhood
        elif neighbourhood_function == 'cosine':
            self.neighbourhood_func = cosine_down_to_zero_neighbourhood
        else:
            raise ValueError(f'Unknown neighbourhood function {neighbourhood_function}')

        if decay_type == 'exponential' and 0 < self.beta < 1:
            self.calculate_decay = decay_exponential
        elif decay_type == 'power' and self.beta < 0:
            self.calculate_decay = decay_power
        else:
            raise ValueError(f'Unknown decay type or invalid beta')

        self.use_qe_for_vertical = use_qe_for_vertical


    def initialize_layer0(self, data):
        self.layer0_weight = np.mean(data, axis=0)
        dists = self.calculate_distance_func(self.layer0_weight, data, 1)

        if self.use_qe_for_vertical:
            # Using qe as global stopping criterion
            reference_val = np.sum(dists)
        else:
            # Using mqe as global stopping criterion
            reference_val = np.mean(dists)

        self.global_stopping_criterion = self.t2 * reference_val
        print(f"Layer 0 Initialized, Global stopping criterion (tau2): {self.global_stopping_criterion:.4f}")

        return reference_val

    def train(self, data):
        layer0_val = self.initialize_layer0(data)

        root_map = GSOM(self.input_dim, self.t1, self.training_epoch_num, layer0_val,
                        self.calculate_distance_func, self.neighbourhood_func,
                        self.calculate_decay, self.learning_rate, self.beta)

        # Deque for maps, subdata and map_id
        queue = deque([root_map, data, "1"])

        while queue:
            current_map, current_data, map_id = queue.popleft()

            current_map.train_and_grow(current_data)
            self.map_db[map_id] = current_map

            self.check_and_expand(current_map, current_data, map_id, queue)

    def check_and_expand(self, parent_map, parent_data, parent_id, queue):
        pass