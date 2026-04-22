# Filename: SOM.py
# Author: Tomas Hanzlik <hanzlto3@fit.cvut.cz>
# Created: 2026-04-22
# Description: Class for the standard SOM algorithm


from help_functions import *
from sklearn import metrics

class SOM:
    """
    Class for standard Self-Organizing Map
    """

    def __init__(self,
                 number_of_rows: int,
                 number_of_cols: int,
                 input_dim: int,
                 neighbourhood_function: str = 'gaussian',
                 distance_k: int = 2,
                 learning_rate: float = 0.5,
                 sigma_start: float | None = None,
                 sigma_end: float | None = 0.5,
                 decay_type: str = 'exponential',
                 beta: float = 0.999):
        """
        :param number_of_rows: Number of rows in the SOM output grid.
        :param number_of_cols: Number of columns in the SOM output grid.
        :param input_dim: Dimension of the SOM reference vectors (need to be same as input samples dimension).
        :param neighbourhood_function: Selection of neighbourhood function for influencing neighbouring neuron units. Must be o of ``gaussian``, ``rectangular``, ``triangular``, ``cosine``. Defaults to ``gaussian``.
        :param distance_k: k variable from the Minskowski family formula. Must be one of ``1`` for Manhattan distance, ``2`` for Euclidean distance, ``np.inf`` for Chebyshev distance, other ``k`` > 2 for generic distance with that k. Defaults to ``2``.
        :param learning_rate: Learning rate used for learning. Defaults to ``0.5``.
        :param sigma_start: Starting sigma used for influencing neighbouring neuron units. If ``None``, the sigma will be calculated as half of the bigger edge of the grid. Defaults to ``None``.
        :param sigma_end: Ending sigma, beta will be calculated to reach this value if ``exponential`` used as a decay function. Defaults to ``0.5``.
        :param decay_type: Selection of the decay function used for decaying sigma and learning rate. Must be one of ``exponential`` or ``power``. Defaults to ``exponential``.
        :param beta: Beta parameter for the decaying function. This value will not be used if ``exponential`` used as a decay function. For power function, the beta must satisfy: beta < 0. Defaults to ``0.999``.
        """
        self.map_rows = number_of_rows
        self.map_cols = number_of_cols
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.distance_k = distance_k
        self.beta = beta
        self.time = 1

        if sigma_start is None:
            self.sigma_start = max(self.map_rows, self.map_cols) / 2.0
        else:
            self.sigma_start = sigma_start
        self.sigma_end = sigma_end

        if self.distance_k == np.inf:
            self.calculate_distance_func = chebyshev_distance
        elif self.distance_k == 1:
            self.calculate_distance_func = manhattan_distance
        elif self.distance_k == 2:
            self.calculate_distance_func = euclidean_distance
        elif self.distance_k < 1:
            raise ValueError('Distance must have positive non-zero k value')
        elif self.distance_k > 2:
            self.calculate_distance_func = lambda a, b, axis: generic_distance(a, b, axis, self.distance_k)
        else:
            raise ValueError('Distance k must be whole number between 1 and np.inf')

        self.weights = np.random.rand(self.map_rows, self.map_cols, self.input_dim)

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

        self.decay_name = decay_type
        if decay_type == 'exponential' and 0 < self.beta < 1:
            self.calculate_decay = decay_exponential
        elif decay_type == 'power' and self.beta < 0:
            self.calculate_decay = decay_power
        else:
            raise ValueError(f'Unknown decay type or invalid beta')

        # matrice databases for visualizations
        self.weights_db = {}
        self.label_matrix_db = None
        self.u_matrix_db = None
        self.u_matrix_extended_db = None

    def get_weights(self) -> np.ndarray:
        """
        Method to get the weights of the SOM
        :return: Weights of all neurons
        """
        return self.weights

    def get_weight_of_node(self, node_idx: tuple[int, int]) -> np.ndarray:
        """
        Method to get the weight of the node at the given index
        :param node_idx: Index of the node
        :return: Weight of node at given index
        """
        return self.weights[node_idx[0]][node_idx[1]]

    def update_time(self):
        """
        Method to update the time for decay functions
        """
        self.time += 1

    def find_BMU(self, input_vector: np.ndarray[float], return_ravelled=False) -> tuple:
        """
        Method to find the BMU for given sample
        :param input_vector: Input sample
        :param return_ravelled: If True, return the flattened index. Defaults to ``False``.
        :return: Index of the BMU
        """
        dists = self.calculate_distance_func(self.weights, input_vector, 2)

        min_index = np.argmin(dists)
        if return_ravelled:
            return min_index

        bmu_idx = np.unravel_index(min_index, dists.shape)
        return bmu_idx

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

    def calculate_grid_distances(self, bmu_idx: tuple[int, int]) -> np.ndarray:
        """
        Method to calculate the distances between the BMU and other neurons on the grid using Euclidean distance
        :param bmu_idx: Index of the BMU
        :return: Distances to other neurons
        """
        # returns coordinates for each dimension
        x_coords, y_coords = np.meshgrid(np.arange(self.map_rows),
                                         np.arange(self.map_cols), indexing='ij')

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
        sigma_t = self.calculate_decay(self.sigma_start, self.beta, self.time)

        # shape (map_width, map_height)
        influence = self.calculate_neighbourhood_influence(bmu_idx, sigma_t)

        # shape (map_width, map_height, input_dim)
        diff = input_vector - self.weights

        # reshaping to (map_width, map_height, 1) to broadcast over diff
        influence_new = influence[:, :, np.newaxis]

        # updating weights
        self.weights += eta_t * influence_new * diff

    def calculate_QE(self, data: np.ndarray) -> float:
        """
        Method to calculate the QE metric
        :param data: Given dataset
        :return: Current QE
        """
        diff_total = 0
        for sample in data:
            bmu_idx = self.find_BMU(sample)
            weight = self.get_weight_of_node(bmu_idx)

            diff_total += euclidean_distance(weight, sample, 0)

        return diff_total / data.shape[0]

    def calculate_TE(self, data: np.ndarray) -> float:
        """
        Method to calculate the TE metric
        :param data: Given dataset
        :return: Current TE
        """
        error_count = 0
        num_samples = data.shape[0]

        for sample in data:
            idx1, idx2 = self.find_top2_BMUs(sample)

            # if the distance between top 2 BMU is bigger than sqrt(2), they cannot be neighbours in rectangular grid
            if euclidean_distance(idx1, idx2, None) > np.sqrt(2):
                error_count += 1

        return error_count / num_samples

    def calculate_purity(self, data: np.ndarray, label: np.ndarray) -> float:
        """
        Method to calculate the purity metric
        :param data: Given dataset
        :param label: Array of target labels for given data samples
        :return: Current purity
        """
        # https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
        true_labels = []
        cluster_labels = []

        for i, sample in enumerate(data):
            bmu_idx = self.find_BMU(sample, True)

            true_labels.append(label[i])
            cluster_labels.append(bmu_idx)

        contingency_matrix = metrics.cluster.contingency_matrix(true_labels, cluster_labels)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def describe_node(self, data: np.ndarray, label: np.ndarray, terms: np.ndarray, node_idx: tuple[int, int], num_words: int, print_samples: bool = False):
        """
        Method only for ``TEXT`` data.
        Method which describes the node at the given index with words with the highest TF-IDF weight. The words are calculated from the neuron's reference vector.
        :param data: Given dataset
        :param label: Array of target labels for given data samples
        :param terms: Words from the TfidfVectorizer. Obtainable by calling get_feature_names_out.
        :param node_idx: Index of the neuron to describe.
        :param num_words: Number of words
        :param print_samples: Bool if labels of the samples for given node are printed
        """
        weights = self.get_weight_of_node(node_idx)
        top_indices = weights.argsort()[-num_words:][::-1]
        top_words = [terms[ind] for ind in top_indices]
        print(f"Top {num_words} words: ", end="")
        for word in top_words:
            print(f"{word},", end=" ")
        print()

        if print_samples:
            correct_sample_idx = []
            for i, sample in enumerate(data):
                bmu_idx = self.find_BMU(sample, False)
    
                if bmu_idx[0] == node_idx[0] and bmu_idx[1] == node_idx[1]:
                    correct_sample_idx.append(i)
    
            print(f"Samples:", end=" ")
            for idx in correct_sample_idx:
                print(f"{label[idx]},", end=" ")
    
            print()


    def train_online(self, data: np.ndarray, y: np.ndarray, num_epochs: int):
        """
        Method to train the SOM
        :param data: Dataset
        :param y: Target labels
        :param num_epochs: Number of epoch iterations for training
        """
        # calculate beta if exponential decay function is used
        if self.decay_name == "exponential":
            self.beta = (self.sigma_end / self.sigma_start) ** (1 / num_epochs)

        # convert categorical data to numerical
        unique_labels, y_int = np.unique(y, return_inverse=True)
        num_samples = data.shape[0]

        if self.input_dim != data.shape[1]:
            raise ValueError("The dimension of input samples is not the same as dimension of SOM weights")

        for epoch in range(num_epochs):
            # permutate input samples
            indices = np.random.permutation(num_samples)
            for i in indices:
                input_vector = data[i]

                bmu_idx = self.find_BMU(input_vector)

                self.update_weights(input_vector, bmu_idx)

            self.update_time()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{num_epochs} complete. Sigma: {self.calculate_decay(self.sigma_start, self.beta, self.time):.4f}, Lr: {self.calculate_decay(self.learning_rate, self.beta, self.time):.4f}, QE: {self.calculate_QE(data):.4F}, TE: {self.calculate_TE(data):.4f}, Purity: {self.calculate_purity(data, y_int):.4f}")

            # save weights for visualizations
            if epoch == 0 or (epoch + 1) % 50 == 0:
                self.weights_db[epoch + 1] = self.weights.copy()

        # matrix databases for visualizations
        self.label_matrix_db = generate_label_matrix_db(self.weights_db, data, y_int, self.map_rows, self.map_cols,
                                                        self.calculate_distance_func)
        self.u_matrix_db = generate_u_matrix_db(self.weights_db)
        self.u_matrix_extended_db = generate_u_matrix_extended_db(self.weights_db)
