# TODO 1. read training and testing (and k-fold cross validation?) data from the datasets left
# TODO 2. rewrite label and timeline parsing to match sequence length
# TODO 3. deal with missing data
# TODO 4. optimize data loading (with multiprocessing, itertools, etc.)
"""
Proposed action data format: (skeleton_sequence, label_sequence, confidence_matrix)
"""
import math
import torch
import random
import numpy as np
from abc import abstractmethod
from typing import Tuple
from torch.autograd import Variable
from torch.utils.data import Dataset, Subset, random_split
from global_configs import DatasetProtocol
from utils.misc import ProgressBar
from utils.processing import (get_regression_matrix)


__all__ = ['SkeletonDataset', 'deserialize_dataset']


class SkeletonDataset(Dataset):
    """
    Parent class/interface of skeleton-based dataset classes which defines members required in common.
    Data to provide for each sample is in format (coordinate_sequence, label_sequence), where:
        - coordinate_sequence should be in shape (sequence_length, 3*joints_per_person); and
        - label_sequence should be in shape (sequence_length, 1) whose values are category indices.

    TODO: May need to add extra label types such as identity and age for each sample (make another abstract class)
    """
    def __init__(self, directory: str, sigma: float, train_portion: float,
                 is_translated: bool = True,
                 is_edged: bool = False,
                 is_rotated: bool = False,
                 is_filtered: bool = False,
                 forecast_t: int = 0,
                 downsample_factor: int = 1):
        print('Start initializing dataset.')
        directory.replace('\\', '/')
        if directory[-1] != '/':
            directory += '/'
        self.root_dir = directory
        self.regression_sigma = sigma
        self.has_interaction = False
        '''
        if is_edged:
            self.is_translated = False
        else:
            self.is_translated = is_translated
        '''
        self.is_translated = is_translated
        if is_rotated:
            self.is_filtered = False
        else:
            self.is_filtered = is_filtered
        self.is_edged = is_edged
        self.is_rotated = is_rotated
        self._labels = set()    # No duplicate item
        self._train_indices = []
        self._test_indices = []
        self._class_sample_counts = {}  # TODO, for weighted cross entropy or oversampling
        self._joints_per_person = 0
        self._min_seq_len = np.inf
        self._max_seq_len = 0
        self._min_x_coord = self._min_y_coord = self._min_z_coord = np.inf
        self._max_x_coord = self._max_y_coord = self._max_z_coord = -np.inf
        self._protocol = None
        self.forecast_t = forecast_t
        self.downsample_factor = downsample_factor
        self.preloaded = False

        # Initialize collections that map data to labels
        self._data_label_pairs = []
        self._action_start_end_frames = []

        # Read labels and corresponding data
        try:
            self.load_label_map()
        except KeyboardInterrupt:
            print('Data loading cancelled.')
        except Exception as e:
            raise IOError('Dataset not loaded correctly at the specified directory.') from e
        if len(self._data_label_pairs) == 0:
            raise IOError('No data was loaded from the specified directory.')
        if self.is_translated and self.is_edged:
            self._joints_per_person *= 2
        self._labels = ['~'] + list(self._labels)    # set label no. 0 as the unknown action
        self._labels = tuple(sorted(self._labels))   # Give ordering and prevent further manipulations
        # self.count_sequence_lengths()
        self._max_seq_len //= self.downsample_factor    # Not accurate
        self._min_seq_len //= self.downsample_factor    # Not accurate
        self.training_set, self.testing_set = None, None
        self.split_train_val_subsets(train_portion)
        print('Finished initializing', str(self) + '.')

    def split_train_val_subsets(self, train_portion: float = 0.5, train_indices: list = None,
                                test_indices: list = None):
        if train_indices and test_indices:
            del self.training_set, self.testing_set
            self._train_indices, self._test_indices = train_indices, test_indices
        if self._train_indices and self._test_indices:
            print('Split the dataset by indices (protocol) specified.')
            random.seed()
            random.shuffle(self._train_indices)
            self.training_set, self.testing_set = Subset(self, self._train_indices), Subset(self, self._test_indices)
        else:
            print('Split the dataset by portion specified.')
            training_size = int(train_portion * len(self))
            self.training_set, self.testing_set = random_split(self, [training_size, len(self) - training_size])
            self._train_indices, self._test_indices = self.training_set.indices, self.testing_set.indices

    def split_k_subset_pairs(self, k: int):
        """
        Split the dataset into k portions for cross validation. Each portion has total sample number divided by k
        amount for testing and the rest for training.
        :param k: Number of portions to split the dataset into. Positive integer expected.
        :return: List of portion pairs (kth_train_subset, kth_test_subset)
        """
        testing_size = int(len(self) / k)
        return [(Subset(self, list(range(i * testing_size)) + list(range((i + 1) * testing_size, len(self)))),   # train
                 Subset(self, list(range(i * testing_size, (i + 1) * testing_size))))     # test
                for i in range(k)]

    @abstractmethod
    def load_label_map(self):
        """
        Load the mapping of each data sample to its label. Sometimes it may load raw sequences as the data if reading
        from files is too slow and data size is not expensive, otherwise it loads the filenames for the labels instead.
        :return:
        """
        raise NotImplementedError('This method is only implemented by subclass')

    @abstractmethod
    def load_data_by_idx(self, data_idx: int):
        raise NotImplementedError('This method is only implemented by subclass')

    @property
    def label_size(self) -> int:
        return len(self._labels)

    def get_joint_number(self) -> int:
        return self._joints_per_person

    def get_labels(self):
        return self._labels

    def get_data_arrays(self, idx):
        """
        Default function which only works for trimmed dataset where any sequence is of only one class. For untrimmed
        dataset, the label tensor is not of the same number and thus this method needs to be reimplemented.
        :param idx:
        :return:
        """
        seq, label_name = self._data_label_pairs[idx]
        label_vector = np.repeat(self._labels.index(label_name), len(seq))
        return seq, label_vector

    def _update_extrema(self, seq: np.ndarray):
        """
        Call this method when loading a new sequence into the dataset.
        :param seq:
        :return:
        """
        length = seq.shape[0]
        if length > self._max_seq_len:
            self._max_seq_len = length
        if length < self._min_seq_len:
            self._min_seq_len = length
        temp = seq.reshape((seq.shape[0], seq.shape[1] // 3, 3)).transpose()
        max_x, min_x = temp[0].max(), temp[0].min()
        max_y, min_y = temp[1].max(), temp[1].min()
        max_z, min_z = temp[2].max(), temp[2].min()
        if max_x > self._max_x_coord:
            self._max_x_coord = max_x
        if min_x < self._min_x_coord:
            self._min_x_coord = min_x
        if max_y > self._max_y_coord:
            self._max_y_coord = max_y
        if min_y < self._min_y_coord:
            self._min_y_coord = min_y
        if max_z > self._max_z_coord:
            self._max_z_coord = max_z
        if min_z < self._min_z_coord:
            self._min_z_coord = min_z

    def get_max_seq_len(self):
        return self._max_seq_len

    def get_min_seq_len(self):
        return self._min_seq_len

    def get_max_coords(self):
        return self._max_x_coord, self._max_y_coord, self._max_z_coord

    def get_min_coords(self):
        return self._min_x_coord, self._min_y_coord, self._min_z_coord

    @property
    def indices_train(self):
        return self._train_indices

    @property
    def indices_test(self):
        return self._test_indices

    def serialize(self, directory: str):
        directory.replace('\\', '/')
        if directory[-1] != '/':
            directory += '/'
        filename = directory + str(self)
        if self.is_translated:
            filename += '_translated'
        if self.is_edged:
            filename += '_edged'
        if self.is_rotated:
            filename += '_rotated'
        if self.is_filtered:
            filename += '_filtered'
        if self._protocol is not None:
            protocol_name = str(self._protocol).replace('DatasetProtocol.', '')
            protocol_type = protocol_name.split('_')[-1].capitalize()
            filename += '_cross%s' % protocol_type
        filename += '.skeldat'
        torch.save(self, filename)
        print('Dataset saved as', filename + '.', 'Use deserialize_dataset() to load.')

    def save_as_database(self, out_dir: str):
        # TODO
        pass

    def load_from_database(self, filename: str):
        pass

    def preload_to_gpu(self, device: torch.device):
        print('Start loading the dataset to GPU RAM.')
        # pg = ProgressBar(80, len(self))
        for idx, (seq, target, confidence) in enumerate(self):
            self._data_label_pairs[idx] = (Variable(seq.to(device)),
                                           Variable(target.to(device)),
                                           Variable(confidence.to(device)))
            # pg.update(idx + 1)
        self.preloaded = True
        print('Finished loading the dataset to GPU RAM.')

    @property
    def train_test_protocol(self):
        return self._protocol

    # overrides
    def __len__(self):
        return len(self._data_label_pairs)

    # overrides
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.preloaded:  # Must call setattr() before the first access
            return self._data_label_pairs[idx]
        seq, label_vector = self.get_data_arrays(idx)
        step = self.downsample_factor
        length = len(seq)
        downsampled_idx = range(0, length, step)
        seq = torch.as_tensor(seq[downsampled_idx])
        label_vector = torch.as_tensor(label_vector[downsampled_idx], dtype=torch.int64)
        regression_matrix = get_regression_matrix(label_vector,
                                                  self.label_size,
                                                  self.regression_sigma,
                                                  math.ceil(self.forecast_t / self.downsample_factor))
        return seq, label_vector, regression_matrix

    def __next__(self):
        if self.iter_ptr < len(self):
            self.iter_ptr += 1
            return self[self.iter_ptr - 1]
        raise StopIteration

    def __iter__(self):
        self.iter_ptr = 0
        return self

    def __str__(self):
        raise NotImplementedError('This method is only implemented by subclass')


def deserialize_dataset(filename: str, pin_memory: bool,
                        device: torch.device = None,
                        new_downsample_factor: int = None,
                        new_regression_sigma: float = None) -> SkeletonDataset:
    dataset: SkeletonDataset = torch.load(filename)
    if new_downsample_factor is not None:
        assert new_downsample_factor >= 1
        dataset.downsample_factor = new_downsample_factor
    if new_regression_sigma is not None:
        assert new_regression_sigma > 0
        dataset.regression_sigma = new_regression_sigma
    print('Loaded', str(dataset), 'with translated %r, '
                                  'edged %r, '
                                  'rotated %r, '
                                  'and filtered %r!'
          % (dataset.is_translated, dataset.is_edged, dataset.is_rotated, dataset.is_filtered))
    # setattr(dataset, 'preloaded', False)
    if pin_memory and device:
        import gc
        dataset.preload_to_gpu(device)
        gc.collect()
        # dataset.preloaded = True
    return dataset
