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

    def find_BMU(self, input_vector):
        dists = self.calculate_distance_func(self.weights, input_vector, 2)

        min_index = np.argmin(dists)
        bmu_idx = np.unravel_index(min_index, dists.shape)
        return bmu_idx

    def get_weight_of_node(self, node_idx):
        return self.weights[node_idx[0]][node_idx[1]]

    def update_time(self):
        self.time += 1

    def reset_time(self):
        self.time = 1

    def calculate_grid_distances(self, bmu_idx):
        row_coords, col_coords = np.meshgrid(np.arange(self.current_row_num),
                                         np.arange(self.current_col_num), indexing='ij')
        dist_sq = (row_coords - bmu_idx[0]) ** 2 + (col_coords - bmu_idx[1]) ** 2
        return np.sqrt(dist_sq)

    def calculate_neighbourhood_influence(self, bmu_idx, sigma_t):
        grid_dists = self.calculate_grid_distances(bmu_idx)
        return self.neighbourhood_func(grid_dists, sigma_t)

    def update_weights(self, input_vector, bmu_idx):
        eta_t = self.calculate_decay(self.learning_rate, self.beta, self.time)
        sigma_t = self.calculate_decay(self.sigma, self.beta, self.time)

        # shape (map_width, map_height)
        influence = self.calculate_neighbourhood_influence(bmu_idx, sigma_t)

        # shape (map_width, map_height, input_dim)
        diff = input_vector - self.weights

        # reshaping to (map_width, map_height, 1) to broadcast over diff
        influence_new = influence[:, :, np.newaxis]

        self.weights += eta_t * influence_new * diff

    def train(self, data):
        num_samples = data.shape[0]

        for epoch in range(self.iteration_num):
            indices = np.random.permutation(num_samples)
            for idx in indices:
                input_vector = data[idx]

                bmu_idx = self.find_BMU(input_vector)

                self.update_weights(input_vector, bmu_idx)

            self.update_time()


    def calculate_unit_errors(self, data):
        unit_errors = np.zeros((self.current_row_num, self.current_col_num))
        unit_hits = np.zeros((self.current_row_num, self.current_col_num))

        for sample in data:
            bmu_idx = self.find_BMU(sample)

            weight = self.get_weight_of_node(bmu_idx)
            dist = self.calculate_distance_func(weight, sample, 0)

            unit_errors[bmu_idx[0], bmu_idx[1]] += dist
            unit_hits[bmu_idx[0], bmu_idx[1]] += 1

        unit_hits_mask = unit_hits > 0
        active_units_count = np.sum(unit_hits_mask)

        # Global MQE = Total Error / number of active units in the GSOM
        total_error = np.sum(unit_errors)
        global_mqe = total_error / active_units_count if active_units_count > 0 else 0

        return unit_errors, global_mqe

    def find_dissimilar_neighbour(self, e_idx):
        e_weight = self.get_weight_of_node(e_idx)
        r, c = e_idx
        max_dist = 0
        d_idx = None

        coords_neighbours = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for rn, cn in coords_neighbours:
            if 0 <= cn < self.current_col_num and 0 <= rn < self.current_row_num:
                neighbor_neuron = self.weights[rn, cn]

                dist = self.calculate_distance_func(e_weight, neighbor_neuron, 0)
                if dist > max_dist:
                    max_dist = dist
                    d_idx = (rn, cn)

        return d_idx

    def add_col_between(self, col1, col2):
        # calculate weight of new col as mean of neighbours
        new_col_weights = (self.weights[:, col1] + self.weights[:, col2]) / 2.0
        self.weights = np.insert(self.weights, max(col1, col2), new_col_weights, axis=1)

        self.current_col_num += 1

    def add_row_between(self, row1, row2):
        # calculate weight of new row as mean of neighbours
        new_row_weights = (self.weights[row1] + self.weights[row2]) / 2.0
        self.weights = np.insert(self.weights, max(row1, row2), new_row_weights, axis=0)

        self.current_row_num += 1


    def grow(self, unit_error_matrix):
        max_index_flat = np.argmax(unit_error_matrix)
        e_index = np.unravel_index(max_index_flat, unit_error_matrix.shape)

        d_index = self.find_dissimilar_neighbour(e_index)

        if d_index is None:
            print("---------------------No neighbour found---------------")
            return

        er, ec = e_index
        dr, dc = d_index

        if er == dr: # same row
            self.add_col_between(ec, dc)
        elif ec == dc: # same col
            self.add_row_between(er,dr)
        else: raise ValueError("e_unit and d_unit not adjacent")

    def train_and_grow(self, data):
        while True:
            self.train(data)

            unit_error_matrix, mqe = self.calculate_unit_errors(data)

            if mqe < self.horizontal_grow_condition:
                break

            self.grow(unit_error_matrix)
            self.reset_time()

            if self.current_row_num > 50 or self.current_col_num > 50:
                print("Max GSOM size reached.")
                break