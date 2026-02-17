import numpy as np
from sklearn import metrics
from help_functions import *
from GSOM import GSOM
from collections import deque

class GHSOM:
    """
    Class for Growing Hierarchical Self-Organizing map
    """
    def __init__(self,
                 input_dim: int,
                 t1: float,
                 t2: float,
                 training_epoch_num: int,
                 learning_rate: float = 0.5,
                 beta: float = 0.999,
                 use_qe_for_vertical: bool = True,
                 min_samples_vertical_grow: int | None = 3,
                 max_gsom_size: int | None = 30,
                 distance_k: int = 2,
                 neighbourhood_function: str = 'gaussian',
                 decay_type: str = 'exponential'):
        """
        :param input_dim: Dimension of the SOM reference vectors (need to be same as input samples dimension).
        :param t1: Tau1 used to specify the factor for horizontal growth condition used in each GSOM
        :param t2: Tau2 used to specify the factor for vertical growth condition
        :param training_epoch_num: Number of epochs used to train each GSOM
        :param learning_rate: Learning rate used for learning. Defaults to ``0.5``.
        :param beta: Beta parameter for the decaying function. For exponential function, the beta must satisfy: 0 < beta < 1, for power function, the beta must satisfy: beta < 0. Defaults to ``0.999``.
        :param use_qe_for_vertical: If qe will be used for training, otherwise mqe, as specified in the research paper. Defaults to ``True``.
        :param min_samples_vertical_grow: Minimum number of samples required to generate a new GSOM child. Defaults to ``3``.
        :param max_gsom_size: Maximum number of neuron units for each GSOM, used to prevent infinite horizontal growth. Defaults to ``30``.
        :param distance_k: k variable from the Minkowski formula. Must be one of ``1`` for Manhattan distance, ``2`` for Euclidean distance, ``np.inf`` for Chebyshev distance, other ``k`` > 2 for generic distance with that k. Defaults to ``2``.
        :param neighbourhood_function: Selection of neighbourhood function for influencing neighbouring neuron units. Must be one of ``gaussian``, ``rectangular``, ``triangular``, ``cosine``. Defaults to ``gaussian``.
        :param decay_type: Selection of the decay function used for decaying sigma and learning rate. Must be one of ``exponential`` or ``power``. Defaults to ``exponential``.
        """

        self.input_dim = input_dim
        self.t1 = t1
        self.t2 = t2
        self.training_epoch_num = training_epoch_num
        self.learning_rate = learning_rate
        self.beta = beta
        self.use_qe_for_vertical = use_qe_for_vertical
        self.min_samples_vertical_grow = min_samples_vertical_grow
        self.max_gsom_size = max_gsom_size

        self.QE = 0.0
        self.TE = 0.0
        self.purity = 0.0

        self.layer0_weight = None # mean of data
        self.global_stopping_criterion = 0 # vertical growth
        self.gsom_db = {}

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



    def initialize_layer0(self, data: np.ndarray) -> float:
        """
        Method to initialize weight vector of Layer 0. The weight will be calculated as the mean (centroid) of input data. Global (vertical) stopping condition will be set as a mean/sum (depending on if qe or mqe is used for growth) multiplied by the Tau2 factor.
        :param data: Dataset
        :return: Error for the Layer 0
        """
        self.layer0_weight = np.mean(data, axis=0)
        dists = self.calculate_distance_func(self.layer0_weight, data, 1)

        if self.use_qe_for_vertical:
            # Using qe as global stopping criterion
            reference_val = np.sum(dists)
        else:
            # Using mqe as global stopping criterion
            reference_val = np.mean(dists)

        self.global_stopping_criterion = self.t2 * reference_val
        print(f"Layer 0 Initialized, Global stopping criterion (tau2 * (qe0 or mqe0)): {self.global_stopping_criterion:.4f}")

        return reference_val

    def train(self, data: np.ndarray, y: np.ndarray):
        """
        Method for training GHSOM
        :param data: Dataset
        :param y: Target labels
        """
        layer0_val = self.initialize_layer0(data)

        if self.input_dim != data.shape[1]:
            raise ValueError("The dimension of input samples is not the same as dimension of SOM weights")

        # create root GSOM - Layer 1
        root_gsom = GSOM(
            input_dim=self.input_dim,
            t1=self.t1,
            training_epoch_num=self.training_epoch_num,
            parent_qe=layer0_val,
            learning_rate=self.learning_rate,
            beta=self.beta,
            max_gsom_size=self.max_gsom_size,
            calculate_distance_func=self.calculate_distance_func,
            neighbourhood_func=self.neighbourhood_func,
            calculate_decay=self.calculate_decay,
            initial_weights=None
        )

        # Deque for gsom, subdata and map_id
        queue = deque()
        queue.append((root_gsom, data, "1"))

        # BFS for expanding child maps
        while queue:
            # get current gsom with its mapped data samples
            current_gsom, current_data, map_id = queue.popleft()
            # train current gsom
            current_gsom.train_and_grow(current_data)
            self.gsom_db[map_id] = current_gsom
            # check current gsom for vertical growth
            self.check_and_expand(current_gsom, current_data, map_id, queue)

        print("Training finished!")
        self.calculate_QE_TE_Purity(data, y)
        print(f"QE: {self.QE}, TE: {self.TE}, Purity: {self.purity}")

    def map_data_to_units(self,
                          gsom_instance: GSOM,
                          data: np.ndarray) -> dict[tuple[int, int]: list[np.ndarray]]:
        """
        Method to find all data samples represented by each neuron unit on the grid
        :param gsom_instance: Current GSOM instance
        :param data: Current data
        :return: Mapping
        """
        mapping = {}

        for sample in data:
            bmu_idx = gsom_instance.find_BMU(sample)

            if bmu_idx not in mapping.keys():
                mapping[bmu_idx] = []
            mapping[bmu_idx].append(sample)

        return mapping

    def map_data_with_labels_to_units(self, gsom_instance: GSOM,
                                      data: np.ndarray,
                                      y: np.ndarray
                                      ) -> dict[tuple[int, int]: list[list[np.ndarray], list[np.ndarray]]]:
        """
        Method to find all data samples with its target label represented by each neuron unit on the grid
        :param gsom_instance: Current GSOM instance
        :param data: Current data
        :param y: Target label of current data
        :return: Mapping
        """
        mapping = {}

        for i, sample in enumerate(data):
            bmu_idx = gsom_instance.find_BMU(sample)

            if bmu_idx not in mapping.keys():
                mapping[bmu_idx] = [[],[]]
            mapping[bmu_idx][0].append(sample)
            mapping[bmu_idx][1].append(y[i])

        return mapping

    def calculate_child_init_weights(self,
                                     parent_gsom: GSOM,
                                     row: int,
                                     column: int
                                     ) -> np.ndarray[[np.ndarray, np.ndarray], [np.ndarray, np.ndarray]]:
        """
        Method to calculate initial weights when adding a new child GSOM instance. Weights are interpolated as a mean of parent weight and parent's neighbours in corresponding direction (e.g. weight of top-left neuron unit will be calculated as a mean of parent, top neighbour and left neighbour).
        :param parent_gsom: Instance of parent GSOM
        :param row: Row index of neuron unit which this child is expanding
        :param column: Column index of neuron unit which this child is expanding
        :return: Array of weights
        """
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

    def check_and_expand(self,
                         parent_gsom: GSOM,
                         parent_data: np.ndarray,
                         parent_id: str,
                         queue: deque):
        """
        Method which checks if the parent GSOM is eligible for vertical expansion. It checks every neuron unit if it satisfies the global (vertical) stopping condition and if so, creates a new GSOM child.
        :param parent_gsom: GSOM instance being expanded
        :param parent_data: Current data mapped to parent GSOM
        :param parent_id: ID of the parent GSOM
        :param queue: Deque to add the new child GSOM to BFS deque for further growth
        """
        # calculate errors for each neuron unit
        unit_errors, _ = parent_gsom.calculate_unit_errors(parent_data)
        # map data to its representative neuron units
        data_mapping = self.map_data_to_units(parent_gsom, parent_data)

        for r in range(parent_gsom.current_row_num):
            for c in range(parent_gsom.current_col_num):

                unit_error_sum = unit_errors[r][c]

                # if qe not used for vertical, divide it by number of samples represented by the neuron unit
                if not self.use_qe_for_vertical:
                    samples_on_unit = len(data_mapping.get((r,c), []))
                    if samples_on_unit > 0:
                        unit_error_sum /= samples_on_unit

                # Vertical growth condition
                if unit_error_sum > self.global_stopping_criterion:
                    subset_data = data_mapping.get((r, c))

                    # Check if number of samples mapped to this neuron satisfies the minimum number of samples for vertical growth
                    if self.min_samples_vertical_grow is None or (subset_data is not None and len(subset_data) > self.min_samples_vertical_grow):

                        child_id = f"{parent_id}_{r}-{c}"
                        print(
                            f"   -> Spawning child {child_id} Num of samples: {len(subset_data)}, Error: {unit_error_sum:.2f} > {self.global_stopping_criterion:.2f})")

                        child_init_weights = self.calculate_child_init_weights(parent_gsom, r, c)

                        child_gsom = GSOM(
                            input_dim=self.input_dim,
                            t1=self.t1,
                            training_epoch_num=self.training_epoch_num,
                            parent_qe=unit_error_sum,
                            learning_rate=self.learning_rate,
                            beta=self.beta,
                            max_gsom_size=self.max_gsom_size,
                            calculate_distance_func=self.calculate_distance_func,
                            neighbourhood_func=self.neighbourhood_func,
                            calculate_decay=self.calculate_decay,
                            initial_weights=child_init_weights
                        )

                        queue.append((child_gsom, subset_data, child_id))

    def get_labels(self, X: np.ndarray, y: np.ndarray) -> dict[str: str]:
        """
        Methods which maps a label of the majority class for each neuron unit within the architecture
        :param X: Dataset
        :param y: Target labels
        :return: Mapping of label
        """
        label_names, y_int = np.unique(y, return_inverse=True)

        # BFS (current_gsom, current_data_index, map_id)
        queue = deque([(self.gsom_db["1"], X, y_int, "1")])

        hierarchy_labels = {}

        # BFS
        while queue:
            curr_gsom, curr_X, curr_y, curr_map_id = queue.popleft()

            if len(curr_X) == 0: continue # empty neuron

            # Map data to units in current map
            mapping = self.map_data_with_labels_to_units(curr_gsom, curr_X, curr_y)

            for r in range(curr_gsom.current_row_num):
                for c in range(curr_gsom.current_col_num):
                    unit_id = f"{curr_map_id}_{r}-{c}"

                    unit_data = mapping.get((r, c))

                    if unit_data is None:
                        hierarchy_labels[unit_id] = "Empty"
                        continue

                    subset_X, subset_y = unit_data
                    counts = np.bincount(subset_y) # get number of occurences for each class
                    majority_class_idx = np.argmax(counts)
                    hierarchy_labels[unit_id] = label_names[majority_class_idx]

                    # check for child gsom and if so add to bfs
                    if unit_id in self.gsom_db.keys():
                        queue.append((self.gsom_db[unit_id], subset_X, subset_y, unit_id))

        return hierarchy_labels

    def calculate_QE_TE_Purity(self, X: np.ndarray, y: np.ndarray):
        """
        Method to calculate QE, TE and Purity metrics
        :param X: Dataset
        :param y: Target labels
        """
        label_names, y_int = np.unique(y, return_inverse=True)

        queue = deque([(self.gsom_db["1"], X, y_int, "1")])

        total_global_qe = 0.0
        total_weighted_te = 0.0
        total_samples_processed = 0
        true_labels = []
        cluster_labels = []
        total_neuron_num = 0

        # BFS
        while queue:
            curr_gsom, curr_X, curr_y, curr_map_id = queue.popleft()

            if len(curr_X) == 0:
                continue
            elif len(curr_X) > 2:
                # TE is calculated as weighted mean through all GSOM instances, not just the leaf units as other metrics
                map_te = curr_gsom.calculate_TE(curr_X)
                num_samples = curr_X.shape[0]
                total_weighted_te += map_te * num_samples
                total_samples_processed += num_samples

            mapping = self.map_data_with_labels_to_units(curr_gsom, curr_X, curr_y)
            for r in range(curr_gsom.current_row_num):
                for c in range(curr_gsom.current_col_num):
                    unit_id = f"{curr_map_id}_{r}-{c}"

                    unit_data = mapping.get((r, c))

                    if unit_data is None or len(unit_data[0]) == 0:
                        continue

                    subset_X, subset_y = unit_data
                    subset_X = np.array(subset_X)

                    if unit_id in self.gsom_db.keys():
                        queue.append((self.gsom_db[unit_id], subset_X, subset_y, unit_id))
                    else:
                        weight_of_leaf = curr_gsom.get_weight_of_node((r,c))

                        # broadcasting weight x array of samples
                        dists = self.calculate_distance_func(weight_of_leaf, subset_X, 1)
                        total_global_qe += np.sum(dists)

                        true_labels.extend(subset_y)
                        # add cluter label for every sample in this leaf node
                        cluster_labels.extend([unit_id] * len(subset_y))

                        total_neuron_num += 1

        self.QE = total_global_qe / X.shape[0] if X.shape[0] > 0 else 0
        self.TE = total_weighted_te / total_samples_processed if total_samples_processed > 0 else 0
        contingency_matrix = metrics.cluster.contingency_matrix(true_labels, cluster_labels)
        self.purity =  np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

        print(f"GHSOM results: Number of neurons: {total_neuron_num}, QE: {self.QE}, TE: {self.TE}, Purity: {self.purity}")



