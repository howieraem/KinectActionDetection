"""Defines CDT of the multi-task / multi-attribute datasets."""
import torch
import numpy as np
from typing import Tuple
from torch.autograd import Variable
from .skeleton_abstract import SkeletonDataset
from global_configs import SensorJointNumber, DatasetProtocol
from utils.misc import get_folders_and_files, ProgressBar
from utils.processing import (preprocess_skeleton_frame,
                              causal_savitzky_golay_filter,
                              standardize_coordinate_origin_sequence)


__all__ = ['SkeletonDatasetMultiTask', 'SkeletonDatasetK3Da', 'SkeletonDatasetJL', 'deserialize_dataset_multitask']


class SkeletonDatasetMultiTask(SkeletonDataset):
    def __init__(self, directory: str,
                 train_portion: float = 0.5,
                 is_translated: bool = True,
                 is_edged: bool = False,
                 is_rotated: bool = False,
                 is_filtered: bool = False,
                 *args, **kwargs):
        self._num_subjects = 0
        self._num_age_groups = 0
        super(SkeletonDatasetMultiTask, self).__init__(directory, 0, train_portion,
                                                       is_translated=is_translated,
                                                       is_edged=is_edged,
                                                       is_rotated=is_rotated,
                                                       is_filtered=is_filtered,
                                                       downsample_factor=2)

    def load_label_map(self):
        raise NotImplementedError

    @property
    def subject_label_size(self):
        return self._num_subjects

    @property
    def age_label_size(self):
        return self._num_age_groups

    def load_data_by_idx(self, data_idx: int):
        return self._data_label_pairs[data_idx]

    # overrides
    def get_data_arrays(self, idx):
        return self.load_data_by_idx(idx)

    # overrides
    def preload_to_gpu(self, device: torch.device):
        print('Start loading the dataset to GPU RAM.')
        pg = ProgressBar(80, len(self))
        for idx, data_label in enumerate(self):
            self._data_label_pairs[idx] = (Variable(data_label[0].to(device)),
                                           Variable(data_label[1].to(device)),
                                           Variable(data_label[2].to(device)),
                                           Variable(data_label[3].to(device)))
            pg.update(idx + 1)
        self.preloaded = True
        print('Finished loading the dataset to GPU RAM.')

    # overrides
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.preloaded:
            return self._data_label_pairs[idx]
        seq, action_label_vector, subject_label_vector, age_label_vector = \
            self._data_label_pairs[idx]
        step_size = self.downsample_factor
        length = len(seq)
        if step_size >= length:
            step_size = length - 1
        downsampled_idx = range(0, length, step_size)
        seq = torch.as_tensor(seq[downsampled_idx], dtype=torch.float32)
        action_label_vector = torch.as_tensor(action_label_vector[downsampled_idx], dtype=torch.int64)
        subject_label_vector = torch.as_tensor(subject_label_vector[downsampled_idx], dtype=torch.int64)
        age_label_vector = torch.as_tensor(age_label_vector[downsampled_idx], dtype=torch.int64)
        return seq, action_label_vector, subject_label_vector, age_label_vector

    def __str__(self):
        raise NotImplementedError


class SkeletonDatasetK3Da(SkeletonDatasetMultiTask):
    # TODO split train/test sets in a better manner
    def __init__(self, directory: str, protocol: DatasetProtocol,
                 is_translated: bool = True,
                 is_edged: bool = False,
                 is_rotated: bool = False,
                 is_filtered: bool = False,
                 *args, **kwargs):
        self._protocol_arg = protocol
        super(SkeletonDatasetK3Da, self).__init__(directory,
                                                  is_translated=is_translated,
                                                  is_edged=is_edged,
                                                  is_rotated=is_rotated,
                                                  is_filtered=is_filtered)
        self.downsample_factor = 2
        self._labels = ('Balance (2-leg Open Eyes)', 'Balance (2-leg Closed Eyes)', 'Chair Rise',
                        'Jump (Minimum)', 'Jump (Maximum)', 'One-Leg Balance (Closed Eyes)',
                        'One-Leg Balance (Open Eyes)', 'Semi-Tandem Balance', 'Tandem Balance',
                        'Walking (Towards the Kinect)', 'Walking (Away from the Kinect)',
                        'Timed-Up-and-Go', 'Hopping')
        self.preloaded = False
        self._num_subjects = 54
        self._num_age_groups = 2

    def load_label_map(self):
        import random
        subject_group1 = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 24, 25,
                          26, 27, 28, 29, 30, 35, 38, 40, 42, 43, 44, 47,
                          49, 50, 53]
        lack_of_data_subjects = range(15, 26)
        if self._protocol_arg is not None:
            self._protocol = self._protocol_arg
        del self._protocol_arg
        raw_subject_info = np.loadtxt(self.root_dir+'participant_details.csv', delimiter=',', dtype=object)[1:]
        subject_info = np.hstack((raw_subject_info[:, (1, 3, 4)].astype(np.int16),
                                  (raw_subject_info[:, 2] == 'm')[:, np.newaxis].astype(np.int16)))
        self._joints_per_person = SensorJointNumber.KINECT_V1   # SensorJointNumber.KINECT_V2
        part_folders, _ = get_folders_and_files(self.root_dir)
        data_directories = []
        for i in range(len(part_folders)):
            part_folders[i] = self.root_dir + part_folders[i] + '/'
            person_folders, _ = get_folders_and_files(part_folders[i])
            for j in range(len(person_folders)):
                person_folders[j] = part_folders[i] + person_folders[j] + '/'
                data_folders, _ = get_folders_and_files(person_folders[j])
                for k in range(len(data_folders)):
                    data_folders[k] = person_folders[j] + data_folders[k] + '/'
                data_directories += data_folders
        pb = ProgressBar(80, len(data_directories))
        subject_action_count = [0] * 54
        if self._protocol == DatasetProtocol.CROSS_SAMPLE:
            for _ in range(10000):
                random.shuffle(data_directories)
        offset = 0
        for idx, data_dir in enumerate(data_directories):
            label_info = data_dir.split('/')[-2].split('_')
            action_label = int(label_info[2]) - 1
            subject_label = self.correct_subject_label(int(label_info[0]))
            age_label = self.get_age_group_label(subject_info[subject_label, 0])
            gender_label = subject_info[subject_label, -1]
            subject_action_count[subject_label] += 1
            try:
                seq = np.loadtxt(data_dir+'skeletons_raw.csv',
                                 delimiter=',',
                                 dtype=np.float32)[:, :self._joints_per_person * 3] * 1000
                edged_seq = np.zeros(seq.shape)
                for fm_idx, frame in enumerate(seq):
                    seq[fm_idx] = preprocess_skeleton_frame(frame, is_15_joint=False,
                                                            to_rotate=self.is_rotated,
                                                            to_edge=False)
                    edged_seq[fm_idx] = preprocess_skeleton_frame(frame, is_15_joint=False,
                                                                  to_rotate=self.is_rotated,
                                                                  to_edge=True)
                if self.is_filtered:
                    seq = causal_savitzky_golay_filter(seq)
                    edged_seq = causal_savitzky_golay_filter(edged_seq)
                if self.is_edged and not self.is_translated:
                    seq = edged_seq
                elif self.is_translated:
                    seq = standardize_coordinate_origin_sequence(seq)
                    if self.is_edged:
                        seq = np.concatenate((seq, edged_seq), axis=-1)
                action_label_vector = np.repeat(action_label, len(seq))
                subject_label_vector = np.repeat(subject_label, len(seq))
                age_label_vector = np.repeat(age_label, len(seq))
                # gender_label_vector = np.repeat(gender_label, len(seq))
                self._update_extrema(seq)
                self._data_label_pairs.append((seq, action_label_vector, subject_label_vector,
                                               age_label_vector  # , gender_label_vector
                                               ))
            except OSError:
                offset += 1
                pass    # ignore accidental missing files
            idx -= offset
            if self._protocol is not None:
                if self._protocol == DatasetProtocol.CROSS_SAMPLE:
                    if subject_label not in lack_of_data_subjects:
                        if subject_action_count[subject_label] <= 6:
                            self._train_indices.append(idx)
                        else:
                            self._test_indices.append(idx)
                elif self._protocol == DatasetProtocol.CROSS_AGE:
                    if action_label not in (3, 11, 12):    # These actions not performed by both age groups
                        if age_label == 1:
                            self._train_indices.append(idx)
                        else:
                            self._test_indices.append(idx)
                elif self._protocol == DatasetProtocol.CROSS_GENDER:
                    if gender_label == 1:  # male
                        self._train_indices.append(idx)
                    else:
                        self._test_indices.append(idx)
                elif self._protocol == DatasetProtocol.CROSS_SUBJECT:
                    if subject_label in subject_group1:
                        self._train_indices.append(idx)
                    else:
                        self._test_indices.append(idx)
                else:
                    raise ValueError('Protocol specified is not available for this dataset.')
            pb.update(idx + 1 + offset)

    @staticmethod
    def correct_subject_label(old_label: int) -> int:
        # 27, 39, 40 and 41 are missing (starting from 1)
        if 26 < old_label < 39:
            return old_label - 1 - 1
        elif old_label >= 39:
            return old_label - 4 - 1
        return old_label - 1

    @staticmethod
    def get_age_group_label(age: int) -> int:
        if age <= 45:
            return 0
        else:
            return 1

    def __str__(self):
        return 'K3Da_Dataset'


class SkeletonDatasetJL(SkeletonDatasetMultiTask):
    def __init__(self, directory: str, protocol: DatasetProtocol,
                 is_translated: bool = True,
                 is_edged: bool = False,
                 is_rotated: bool = False,
                 is_filtered: bool = False, *args, **kwargs):
        from torch.utils.data import Subset
        self._2nd_train_indices = []
        self._2nd_test_indices = []
        self._3rd_train_indices = []
        self._3rd_test_indices = []
        self._protocol_arg = protocol
        super(SkeletonDatasetJL, self).__init__(directory,
                                                is_translated=is_translated,
                                                is_edged=is_edged,
                                                is_rotated=is_rotated,
                                                is_filtered=is_filtered)
        self._labels = ('Walk', 'Wave', 'Clap', 'Drink', 'Sit Down', 'Stand Up',
                        'Throw', 'Crouch', 'One-leg Balance', 'Jump', 'Unknown')
        self.preloaded = False
        self._num_subjects = 20
        self._num_age_groups = 2    # 3
        self.training_set2, self.testing_set2 = \
            Subset(self, self._2nd_train_indices), Subset(self, self._2nd_test_indices)
        self.training_set3, self.testing_set3 = \
            Subset(self, self._3rd_train_indices), Subset(self, self._3rd_test_indices)

    def load_label_map(self):
        age_group1_subjects = [0, 1, 2, 3, 4, 5, 6, 19, 13, 14, 19]
        subject_group1 = [0, 1, 2, 7, 8, 13, 14, 15, 16, 19]
        male_subjects = [2, 3, 4, 5, 8, 10, 11, 13, 14, 16, 18, 19]
        if self._protocol_arg is not None:
            self._protocol = self._protocol_arg
        del self._protocol_arg
        self._joints_per_person = SensorJointNumber.KINECT_V1
        import re
        subject_folders, _ = get_folders_and_files(self.root_dir)
        pg = ProgressBar(80, 664)
        count = 0
        for subject_folder in subject_folders:
            if not subject_folder.startswith('S'):
                continue
            try:
                subject_label = int(subject_folder[1:])
            except ValueError:
                continue
            _, data_filenames = get_folders_and_files(self.root_dir+subject_folder)
            for data_filename in data_filenames:
                try:
                    _, action_label, sample_id = re.findall(r'\d+', data_filename)
                except ValueError:
                    continue
                action_label, sample_id = int(action_label), int(sample_id)
                seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
                edged_seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
                with open(self.root_dir+subject_folder+'/'+data_filename, 'r') as data_file:
                    data_file.readline()    # action_label
                    data_file.readline()    # age_label = int(data_file.readline())
                    data_file.readline()    # subject_label
                    age_label = 0 if subject_label in age_group1_subjects else 1
                    while 1:
                        line = data_file.readline().rstrip()
                        if not line:
                            break
                        frame = np.array(line.split(), dtype=np.float32)
                        seq = np.vstack((seq, preprocess_skeleton_frame(frame,
                                                                        is_15_joint=False,
                                                                        to_edge=False,
                                                                        to_rotate=self.is_rotated)))
                        edged_seq = np.vstack((seq, preprocess_skeleton_frame(frame,
                                                                              is_15_joint=False,
                                                                              to_edge=True,
                                                                              to_rotate=self.is_rotated)))
                if self.is_filtered:
                    seq = causal_savitzky_golay_filter(seq)
                    edged_seq = causal_savitzky_golay_filter(edged_seq)
                if self.is_edged and not self.is_translated:
                    seq = edged_seq
                elif self.is_translated:
                    seq = standardize_coordinate_origin_sequence(seq)
                    if self.is_edged:
                        seq = np.concatenate((seq, edged_seq), axis=-1)
                self._update_extrema(seq)
                action_label_vector = np.repeat(action_label, len(seq))
                subject_label_vector = np.repeat(subject_label, len(seq))
                age_label_vector = np.repeat(age_label, len(seq))
                self._data_label_pairs.append((seq, action_label_vector, subject_label_vector,
                                               age_label_vector  # , gender_label_vector
                                               ))
                if self._protocol == DatasetProtocol.CROSS_SAMPLE:
                    if sample_id == 1:
                        self._train_indices.append(count)
                        self._2nd_train_indices.append(count)
                        self._3rd_test_indices.append(count)
                    elif sample_id == 3:
                        self._train_indices.append(count)
                        self._2nd_test_indices.append(count)
                        self._3rd_train_indices.append(count)
                    elif sample_id == 2:
                        self._test_indices.append(count)
                        self._2nd_train_indices.append(count)
                        self._3rd_train_indices.append(count)
                    else:
                        self._test_indices.append(count)
                        self._2nd_test_indices.append(count)
                        self._3rd_test_indices.append(count)
                elif self._protocol == DatasetProtocol.CROSS_SUBJECT:
                    if subject_label in subject_group1:
                        self._train_indices.append(count)
                    else:
                        self._test_indices.append(count)
                elif self._protocol == DatasetProtocol.CROSS_GENDER:
                    if subject_label in male_subjects:
                        self._train_indices.append(count)
                    else:
                        self._test_indices.append(count)
                elif self._protocol == DatasetProtocol.CROSS_AGE:
                    if age_label == 0:
                        self._train_indices.append(count)
                    else:
                        self._test_indices.append(count)
                else:
                    raise ValueError('Protocol specified is not available for this dataset.')
                count += 1
                pg.update(count)

    def __str__(self):
        return 'JL_Dataset'


def deserialize_dataset_multitask(filename: str,
                                  pin_memory: bool,
                                  device: torch.device = None,
                                  new_downsample_factor: int = None) -> SkeletonDatasetMultiTask:
    dataset: SkeletonDatasetMultiTask = torch.load(filename)
    if new_downsample_factor is not None:
        assert new_downsample_factor >= 1
        dataset.downsample_factor = new_downsample_factor
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
