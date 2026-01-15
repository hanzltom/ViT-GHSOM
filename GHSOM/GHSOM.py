from help_functions import *
from GSOM import GSOM
from collections import deque

class GHSOM:
    def __init__(self, input_dim, t1, t2, training_epoch_num, learning_rate = 0.5,
                 distance_k = 2, neighbourhood_function = 'gaussian', decay_type='exponential',
                 beta=0.999, use_qe_for_vertical = True, min_samples_vertical_grow = 5):

        self.input_dim = input_dim
        self.t1 = t1
        self.t2 = t2
        self.training_epoch_num = training_epoch_num
        self.learning_rate = learning_rate
        self.beta = beta
        self.min_samples_vertical_grow = min_samples_vertical_grow

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

        root_gsom = GSOM(self.input_dim, self.t1, self.training_epoch_num, layer0_val,
                        self.calculate_distance_func, self.neighbourhood_func,
                        self.calculate_decay, self.learning_rate, self.beta)

        # Deque for gsom, subdata and map_id
        queue = deque()
        queue.append((root_gsom, data, "1"))

        # BFS for expanding maps
        while queue:
            current_gsom, current_data, map_id = queue.popleft()
            print(f"Training gsom: {map_id}")
            current_gsom.train_and_grow(current_data)
            self.map_db[map_id] = current_gsom

            self.check_and_expand(current_gsom, current_data, map_id, queue)

    def map_data_to_units(self, gsom_instance, data):
        mapping = {}

        for sample in data:
            bmu_idx = gsom_instance.find_BMU(sample)

            if bmu_idx not in mapping.keys():
                mapping[bmu_idx] = []
            mapping[bmu_idx].append(sample)

        return mapping

    def calculate_child_init_weights(self, parent_gsom, row, column):
        parent_weight = parent_gsom.get_weight_of_node((row, column))

        # get weights of the neighbours among the neuron which is being expanded
        n_up = parent_gsom.get_weight_of_node((row - 1, column)) if row > 0 else parent_weight
        n_down = parent_gsom.get_weight_of_node((row + 1, column)) if row < parent_gsom.current_row_num - 1 else parent_weight
        n_left = parent_gsom.get_weight_of_node((row, column - 1)) if column > 0 else parent_weight
        n_right = parent_gsom.get_weight_of_node((row, column + 1)) if column < parent_gsom.current_col_num - 1 else parent_weight

        # interpolation
        child_00 = (parent_weight + n_up + n_left) / 3
        child_01 = (parent_weight + n_up + n_right) / 3
        child_10 = (parent_weight + n_down + n_left) / 3
        child_11 = (parent_weight + n_down + n_right) / 3

        return np.array([[child_00, child_01], [child_10, child_11]])

    def check_and_expand(self, parent_gsom, parent_data, parent_id, queue):
        unit_errors, _ = parent_gsom.calculate_unit_errors(parent_data)

        data_mapping = self.map_data_to_units(parent_gsom, parent_data)
        data_num = 0

        for r in range(parent_gsom.current_row_num):
            for c in range(parent_gsom.current_col_num):
                print(f"Checking idx ({r}, {c}), num of samples: {len(data_mapping.get((r, c))) if data_mapping.get((r, c)) else 0}")
                unit_error_sum = unit_errors[r][c]

                if not self.use_qe_for_vertical:
                    samples_on_unit = len(data_mapping.get((r,c), []))
                    if samples_on_unit > 0:
                        unit_error_sum /= samples_on_unit
                print(f"Qe: {unit_error_sum}, > {self.global_stopping_criterion}")
                if unit_error_sum > self.global_stopping_criterion:
                    subset_data = data_mapping.get((r, c))
                    print(f"Condition to spawn child: subset len: {len(subset_data) if subset_data is not None else 0}")

                    if subset_data is not None and len(subset_data) > self.min_samples_vertical_grow:
                        child_id = f"{parent_id}_{r}-{c}"
                        print(
                            f"   -> Spawning child {child_id} (Error: {unit_error_sum:.2f} > {self.global_stopping_criterion:.2f})")

                        child_init_weights = self.calculate_child_init_weights(parent_gsom, r, c)

                        child_gsom = GSOM(self.input_dim, self.t1, self.training_epoch_num, unit_error_sum,
                                        self.calculate_distance_func, self.neighbourhood_func,
                                          self.calculate_decay, self.learning_rate, self.beta,
                                          child_init_weights)

                        queue.append((child_gsom, subset_data, child_id))