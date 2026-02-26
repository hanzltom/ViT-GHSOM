import numpy as np
from help_functions import *
from typing import Callable

class GSOM:
    """
    Class for Growing Self-Organizing Map
    """
    def __init__(self,
                 input_dim: int,
                 t1: float,
                 training_epoch_num: int,
                 parent_qe: float,
                 learning_rate: float,
                 beta: float,
                 max_gsom_size: int,
                 calculate_distance_func: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                 neighbourhood_func: Callable[[np.ndarray, float], np.ndarray],
                 calculate_decay: Callable[[float, float, int], float],
                 initial_weights: np.ndarray | None = None):
        """
        :param input_dim: Dimension of the SOM reference vectors (need to be same as input samples dimension).
        :param t1: Tau1 used to specify the factor for horizontal growth condition used in GSOM
        :param training_epoch_num: Number of epochs used to train each GSOM
        :param parent_qe: qe or mqe metric of the parent neuron unit located in the upper layer which this GSOM is extending as a child unit
        :param learning_rate: Learning rate used for learning.
        :param beta: Beta parameter for the decaying function. For exponential function, the beta must satisfy: 0 < beta < 1, for power function, the beta must satisfy: beta < 0.
        :param max_gsom_size: Maximum number of neuron units for GSOM, used to prevent infinite horizontal growth.
        :param calculate_distance_func: Distance function
        :param neighbourhood_func: Neighbourhood function
        :param calculate_decay: Decay function
        :param initial_weights: Initial weights, if None they are randomly initialized. Default is None.
        """
        self.input_dim = input_dim
        self.time = 1
        self.current_row_num = 2
        self.current_col_num = 2
        self.training_epoch_num = training_epoch_num
        self.horizontal_grow_condition = t1 * parent_qe
        self.learning_rate = learning_rate
        self.max_gsom_size = max_gsom_size

        self.sigma = max(self.current_row_num, self.current_col_num) / 2.0


        if initial_weights is not None:
            if initial_weights.ndim == 3 and initial_weights.shape[0] == 2 and initial_weights.shape[1] == 2 and initial_weights.shape[2] == self.input_dim:
                self.weights = initial_weights
            else:
                raise ValueError("Wrong shape of initial weights")
        else:
            self.weights = np.random.rand(self.current_row_num, self.current_col_num, self.input_dim)

        # functions from GHSOM, not needed to check
        self.calculate_distance_func = calculate_distance_func
        self.neighbourhood_func = neighbourhood_func
        self.calculate_decay = calculate_decay
        self.beta = beta

    def find_BMU(self, input_vector: np.ndarray) -> tuple[int,int]:
        """
        Method to find the BMU for given sample
        :param input_vector: Input sample
        :return: Index of the BMU
        """
        dists = self.calculate_distance_func(self.weights, input_vector, 2)

        min_index = np.argmin(dists)
        bmu_idx = np.unravel_index(min_index, dists.shape)
        return bmu_idx

    def get_weight_of_node(self, node_idx: tuple[int,int]):
        """
        Method returning the weight of node at given index
        :param node_idx: Index of the node
        :return: Weight of node
        """
        return self.weights[node_idx[0]][node_idx[1]]

    def update_time(self):
        """
        Method updating time for training
        """
        self.time += 1

    def reset_time(self):
        """
        Method resetting time when expanding the grid
        """
        self.time = 1

    def calculate_grid_distances(self, bmu_idx: tuple[int, int]) -> np.ndarray:
        """
        Method to calculate the distances between the BMU and other neurons on the grid using Euclidean distance
        :param bmu_idx: Index of the BMU
        :return: Distances to other neurons
        """
        # returns coordinates for each dimension
        x_coords, y_coords = np.meshgrid(np.arange(self.current_row_num),
                                         np.arange(self.current_col_num), indexing='ij')

        # calculating distances using Euclidean distance and broadcasting
        dist_sq = (x_coords - bmu_idx[0]) ** 2 + (y_coords - bmu_idx[1]) ** 2
        return np.sqrt(dist_sq)

    def calculate_neighbourhood_influence(self, bmu_idx: tuple[int, int], sigma_t: float) -> np.ndarray:
        """
        Method to calculate the neighbourhood influence from the BMU
        :param bmu_idx: Index of the BMU
        :param sigma_t: Sigma at given time
        :return: Updates for each neuron
        """
        grid_dists = self.calculate_grid_distances(bmu_idx)
        return self.neighbourhood_func(grid_dists, sigma_t)

    def update_weights(self, input_vector: np.ndarray[float], bmu_idx: tuple[int, int]):
        """
        Method to update the weights of the SOM
        :param input_vector: Input sample
        :param bmu_idx: Index of the BMU for given sample
        """
        eta_t = self.calculate_decay(self.learning_rate, self.beta, self.time)
        sigma_t = self.calculate_decay(self.sigma, self.beta, self.time)

        # shape (map_width, map_height)
        influence = self.calculate_neighbourhood_influence(bmu_idx, sigma_t)

        # shape (map_width, map_height, input_dim)
        diff = input_vector - self.weights

        # reshaping to (map_width, map_height, 1) to broadcast over diff
        influence_new = influence[:, :, np.newaxis]

        # updating weights
        self.weights += eta_t * influence_new * diff

    def train(self, data: np.ndarray):
        """
        Method to train current size of GSOM
        :param data: Mapped data
        """
        num_samples = len(data)

        for epoch in range(self.training_epoch_num):
            # permutate input samples
            indices = np.random.permutation(num_samples)
            for idx in indices:
                input_vector = data[idx]

                bmu_idx = self.find_BMU(input_vector)

                self.update_weights(input_vector, bmu_idx)

            self.update_time()


    def calculate_unit_errors(self, data) -> tuple[np.ndarray[np.ndarray[float]], float]:
        """
        Method calculating error for each neuron unit and MQE
        :param data: Mapped data
        :return: Error for each unit and MQE
        """
        # matrix to calculate error for each neuron unit
        unit_errors = np.zeros((self.current_row_num, self.current_col_num))
        # matrix to calculate number of input samples represented by each neuron, used to compute number of active neuron units
        unit_hits = np.zeros((self.current_row_num, self.current_col_num))

        for sample in data:
            bmu_idx = self.find_BMU(sample)

            weight = self.get_weight_of_node(bmu_idx)
            dist = self.calculate_distance_func(weight, sample, 0)

            # adding the error to the neuron unit
            unit_errors[bmu_idx[0], bmu_idx[1]] += dist
            # adding +1 sample represented by the unit
            unit_hits[bmu_idx[0], bmu_idx[1]] += 1

        # find number of active neuron units by filtering those that represent at least one sample and summing their number
        unit_hits_mask = unit_hits > 0
        active_units_count = np.sum(unit_hits_mask)

        # Global MQE = Total Error / number of active units in the GSOM
        # TODO add MQE computation also from mqe and not only qe, needs to be defined in the thesis
        total_error = np.sum(unit_errors)
        global_mqe = total_error / active_units_count if active_units_count > 0 else 0

        return unit_errors, global_mqe

    def find_dissimilar_neighbour(self, e_idx: tuple[int,int]) -> tuple[int,int]:
        """
        Method to find the most dissimilar neighbour in a rectangular grid to the neuron unit e at given index
        :param e_idx: Index in the grid of the neuron unit e
        :return: Index of the most dissimilar neighbour d to the unit e
        """
        e_weight = self.get_weight_of_node(e_idx)
        r, c = e_idx
        max_dist = 0
        d_idx = None

        # coordinates of possible neighbours
        coords_neighbours = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for rn, cn in coords_neighbours:
            # check valid neighbours for neurons located at the edge of grid
            if 0 <= cn < self.current_col_num and 0 <= rn < self.current_row_num:
                neighbor_neuron = self.weights[rn, cn]

                # compute distance between units
                dist = self.calculate_distance_func(e_weight, neighbor_neuron, 0)
                if dist > max_dist:
                    max_dist = dist
                    d_idx = (rn, cn)

        return d_idx

    def add_col_between(self, col1: int, col2: int):
        """
        Method which inserts a new column between given column indices
        :param col1: Index of the first column
        :param col2: Index of the second column
        """
        # calculate weight of new col as mean of neighbours
        new_col_weights = (self.weights[:, col1] + self.weights[:, col2]) / 2.0
        self.weights = np.insert(self.weights, max(col1, col2), new_col_weights, axis=1)

        self.current_col_num += 1

    def add_row_between(self, row1: int, row2: int):
        """
        Method which inserts a new row between given row indices
        :param row1: Index of the first row
        :param row2: Index of the second row
        """
        # calculate weight of new row as mean of neighbours
        new_row_weights = (self.weights[row1] + self.weights[row2]) / 2.0
        self.weights = np.insert(self.weights, max(row1, row2), new_row_weights, axis=0)

        self.current_row_num += 1


    def grow(self, unit_error_matrix: np.ndarray):
        """
        Method which grows the grid by one row or one column
        :param unit_error_matrix: Error matrix to find the neuron unit e with the highest error
        """
        max_index_flat = np.argmax(unit_error_matrix)
        e_index = np.unravel_index(max_index_flat, unit_error_matrix.shape)

        d_index = self.find_dissimilar_neighbour(e_index)

        if d_index is None:
            print(f"---------------------No neighbour found for {e_index}---------------")
            return

        er, ec = e_index
        dr, dc = d_index

        if er == dr: # same row, adding column between their cols
            self.add_col_between(ec, dc)
        elif ec == dc: # same col, adding row between their rows
            self.add_row_between(er,dr)
        else: raise ValueError("e_unit and d_unit not adjacent")

    def train_and_grow(self, data: np.ndarray):
        """
        Method to train GSOM and grow the gird if needed
        :param data: Mapped data
        """
        while True:
            self.train(data)

            unit_error_matrix, mqe = self.calculate_unit_errors(data)

            # Condition if the error is low enough to stop growing the grid
            if mqe < self.horizontal_grow_condition:
                break

            self.grow(unit_error_matrix)
            self.reset_time()
            #print(f"Novy shape {self.current_row_num}, {self.current_col_num}")

            # Condition to prevent infinite growth
            if self.max_gsom_size is not None:
                if self.current_row_num >= self.max_gsom_size or self.current_col_num >= self.max_gsom_size:
                    print("Max GSOM size reached.")
                    break

    def find_top2_BMUs(self, input_vector: np.ndarray[float]) -> tuple[np.ndarray, np.ndarray]:
        """
        Method to find the top 2 BMUs for given sample
        :param input_vector: Input sample
        :return: Two numpy arrays containing the (row, col) coordinates of the first and second BMU respectively.
        """
        # https://gist.github.com/EdisonLeeeee/df5a2427f902312bbd29151f79e728ab
        dists = self.calculate_distance_func(self.weights, input_vector, 2)
        dists_flat = dists.flatten()

        indices_flat = np.argpartition(dists_flat, 2)[:2]

        # Index of 1. BMU
        row1, col1 = np.unravel_index(indices_flat[0], dists.shape)
        # Index of 2. BMU
        row2, col2 = np.unravel_index(indices_flat[1], dists.shape)

        return np.array((row1, col1)), np.array((row2, col2))

    def calculate_TE(self, data: np.ndarray) -> float:
        """
        Method calculating TE error metric
        :param data: Mapped data
        :return: TE metric
        """
        error_count = 0
        num_samples = data.shape[0]

        for sample in data:
            # find top two closest neuron units to the sample
            idx1, idx2 = self.find_top2_BMUs(sample)
            # if the distance between indices of these sample is more than sqrt(2), they are not adjacent on the rectangular grid
            if euclidean_distance(idx1, idx2, None) > np.sqrt(2):
                error_count += 1

        return error_count / num_samples