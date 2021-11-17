# TODO 1. read training and testing (and k-fold cross validation?) data from the datasets left
# TODO 2. rewrite label and timeline parsing to match sequence length
# TODO 3. deal with missing data
# TODO 4. optimize data loading (with multiprocessing, itertools, etc.)
"""
Proposed action data format: (skeleton_sequence, start_frame_idx, end_frame_idx, label_idx)
"""
import enum
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import re
from lxml import etree
from overrides import overrides
from utils.misc import *


# Parameters for simple transformations that may improve learning
_COORDS_OFFSET = 0.
_SCALE_FACTOR = 2.0


class _SensorJointNumber(enum.IntEnum):
    openni = 15
    kinect_v1 = 20
    kinect_v2 = 25


class SkeletonDataset(Dataset):
    """
    Parent class of skeleton-based dataset classes which defines members required in common.
    Data to provide for each sample is in format (coordinate_sequence, timeline, label), where:
        - coordinate_sequence should be in shape (sequence_length, 3*joints_per_person); and
        - label should be in shape (sequence_length, 1) whose values are category indices.
    """
    def __init__(self, directory: str, train_portion: float=0.9):
        print('Start initializing dataset.')
        directory.replace('\\', '/')
        if directory[-1] != '/':
            directory += '/'
        self.root_dir = directory
        self._labels = set()    # No duplicate item
        self._joints_per_person = 0
        self._max_seq_len = 0
        self._label_data_lengths = np.zeros(self.label_size())

        # Initialize collections that map data to labels
        self._data_label_pairs = list()

        # Read labels and corresponding data
        try:
            self.load_label_map()
        except KeyboardInterrupt:
            print('Data loading cancelled.')
        except Exception as e:
            raise IOError('Dataset not located correctly at the specified directory.') from e
        if len(self._data_label_pairs) == 0:
            raise IOError('Dataset not located correctly at the specified directory.')
        self._labels = tuple(sorted(self._labels))   # Give ordering and prevent further manipulations
        # self.count_sequence_lengths()
        self.training_set, self.testing_set = None, None
        self.split_train_val_subsets(train_portion)
        print('Finished initializing', str(self) + '.')

    def split_train_val_subsets(self, train_portion: float):
        training_size = int(train_portion * len(self))
        self.training_set, self.testing_set = random_split(self, [training_size, len(self) - training_size])

    def load_label_map(self):
        """
        Load the mapping of each data sample to its label. Sometimes it may load raw sequences as the data if reading
        from files is too slow and data size is not expensive, otherwise it loads the filenames for the labels instead.
        :return:
        """
        raise NotImplementedError('This method is only implemented by subclass')

    def load_data_by_idx(self, data_idx: int):
        raise NotImplementedError('This method is only implemented by subclass')

    def label_size(self):
        return len(self._labels)

    def get_joint_number(self):
        return self._joints_per_person

    def get_labels(self):
        return self._labels

    def get_data_label_pair(self, idx):
        """
        Default function which only works for trimmed dataset where any sequence is of only one class. For trimmed
        dataset, the label tensor is not of the same number and thus this method should be reimplemented.
        :param idx:
        :return:
        """
        label_name = self._data_label_pairs[idx][1]
        seq = self.load_data_by_idx(idx)
        label_vector = np.repeat(self._labels.index(label_name), len(seq))
        return seq, label_vector

    @overrides
    def __len__(self):
        return len(self._data_label_pairs)

    @overrides
    def __getitem__(self, idx):
        seq, label_vector = self.get_data_label_pair(idx)
        assert len(seq) == len(label_vector), 'Label must be assigned to every sequence frame.'
        return torch.from_numpy(seq), torch.from_numpy(label_vector).long()

    def __next__(self):
        if self.iter_ptr < len(self):
            seq, label_vector = self.get_data_label_pair(self.iter_ptr)
            assert len(seq) == len(label_vector), 'Label must be assigned to every sequence frame.'
            self.iter_ptr += 1
            return torch.from_numpy(seq), torch.from_numpy(label_vector).long()
        raise StopIteration

    def __iter__(self):
        self.iter_ptr = 0
        return self

    def __str__(self):
        raise NotImplementedError('This method is only implemented by subclass')


class SkeletonDatasetG3D(SkeletonDataset):
    """
    Load G3D Dataset by:
        V. Bloom, D. Makris, and V. Argyriou, "G3d: A gaming action dataset and real time action recognition evaluation
        framework," in Computer Vision and Pattern Recognition Workshops (CVPRW), 2012 IEEE Computer Society Conference
        on, 2012, pp. 7-12: IEEE.

    Dataset description:
        G3D dataset contains a range of gaming actions captured with Microsoft Kinect. The Kinect enabled us to record
        synchronised video, depth and skeleton data. The dataset contains 10 subjects performing 20 gaming actions:
        punch right, punch left, kick right, kick left, defend, golf swing, tennis swing forehand, tennis swing
        backhand, tennis serve, throw bowling ball, aim and fire gun, walk, run, jump, climb, crouch, steer a car,
        wave, flap and clap. The 20 gaming actions are recorded in 7 action sequences. Most sequences contain
        multiple actions in a controlled indoor environment with a fixed camera, a typical setup for gesture based
        gaming. Each sequence is repeated three times by each subject as shown in Table below.


        Actor	Fighting	Golf	    Tennis	    Bowling	    FPS	        Driving	    Misc
        1	    22	23	24	25	26	27	28	29	30	31	32	33*	34	35	36	37	38	39	40	41	42

        2	    43	44	45	46	47	48	49	50	51	52	53	54	55	56	57	58	59	60	61	62	63

        3	    64	65	66	67	68	69	70	71	72	73	74	75	76	77	78	79	80	81	82	83	84

        4	    85	86	87	88	89	90	91	92	93	94	95	96	97	98	99	100	101	102	103	104	105

        5	    106	107	108	109	110	111	112	113	114	115	116	117	118	119	120	121	122	123	124	125	126

        6	    127	128	129	130	131	132	133	134	135	136	137	138	139	140	141	142	143	144	145	146	147

        7	    148	149	150	151	152	153	154	155	156	157	158	159	160	161	162	163	164	165	166	167	168

        8	    169	170	171	172	173	174	175	176	177	178	179	180	181	182	183	184	185	186	187	188	189

        9	    190	191	192	193	194	195	196	197	198	199	200	201	202	203	204	205	206	207	208	209	210

        10	    214	215	216	217	218	219	220	221	222	223	224	225	226	227	228	229	230	231	232	233	234

        *Please note sequence 33 was corrupted and is therefore not available.

    Formats:
        Due to the formats selected, it is possible to view all the recorded data and metadata without any special
        software tools. The three streams were recorded at 30fps in a mirrored view. The depth and colour images were
        stored as 640x480 PNG files and the skeleton data in XML files.
    """
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        self._joints_per_person = _SensorJointNumber.kinect_v1
        label_folders, _ = get_folders_and_files(self.root_dir)
        self._labels |= set(label_folders)  # assuming each folder's name is the corresponding label name
        for label in self._labels:
            label_dir = self.root_dir + '/' + label + '/'
            sample_folders, _ = get_folders_and_files(label_dir)
            for sample in sample_folders:
                data_dir = label_dir + sample + '/Skeleton/'
                _, data_filenames = get_folders_and_files(data_dir)
                data_file_ids = []
                for filename in data_filenames:
                    if filename.endswith('.xml'):
                        data_file_ids.append(int(re.findall(r'\d+', filename)[0]))
                    else:
                        continue
                data_file_ids = sorted(data_file_ids)
                seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person * 3)
                action_frame_indices = []
                for data_file_id in data_file_ids:
                    frame = []
                    tree = etree.parse(data_dir + 'Skeleton ' + str(data_file_id) + '.xml')
                    joints = tree.xpath('//Joint')
                    if len(joints) < 20:
                        continue
                    for joint in joints[:20]:
                        coords = joint[0]
                        frame.append(_SCALE_FACTOR * float(coords[0].text) * 1000. + _COORDS_OFFSET)  # x in mm
                        frame.append(_SCALE_FACTOR * float(coords[1].text) * 1000. + _COORDS_OFFSET)  # y in mm
                        frame.append(_SCALE_FACTOR * float(coords[2].text) * 1000. + _COORDS_OFFSET)  # z in mm
                    seq = np.vstack((seq, np.array(frame, dtype=np.float32)))
                    action_frame_indices.append(data_file_id)
                if seq.shape[0] != 0 and len(action_frame_indices) != 0:
                    self._data_label_pairs.append((seq, label))

    def load_data_by_idx(self, data_idx: int):
        return self._data_label_pairs[data_idx][0]

    def __str__(self):
        return 'G3D_Dataset'


class SkeletonDatasetMSRC(SkeletonDataset):
    # CSV-based skeleton data
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        raise NotImplementedError('This method is only implemented by subclass')

    def load_data_by_idx(self, data_idx: int):
        raise NotImplementedError('This method is only implemented by subclass')

    def __str__(self):
        return 'MSRC_Dataset'


class SkeletonDatasetMSRActionPair(SkeletonDataset):
    # MAT-based skeleton data
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        raise NotImplementedError('This method is only implemented by subclass')

    def load_data_by_idx(self, data_idx: int):
        raise NotImplementedError('This method is only implemented by subclass')

    def __str__(self):
        return 'MSR_ActionPair_Dataset'


class SkeletonDatasetMSRAction3D(SkeletonDataset):
    """
    Load MSR Action 3D Dataset (trimmed) by:
        W. Li, Z. Zhang, and Z. Liu, "Action recognition based on a bag of 3d points," in Computer Vision and Pattern
        Recognition Workshops (CVPRW), 2010 IEEE Computer Society Conference on, 2010, pp. 9-14: IEEE.

    Dataset description:
        The dataset contains 20 action types (by 10 subjects, each subject performs each action 2 or 3 times):
        high arm wave, horizontal arm wave, hammer, hand catch, forward punch, high throw, draw x, draw tick,
        draw circle, hand clap, two hand wave, side-boxing,bend, forward kick, side kick, jogging,
        tennis swing, tennis serve, golf swing, pickup & throw.

        There is a skeleton sequence file for each depth sequence in the Action3D dataset. A skeleton has 20 joint
        positions. Four real numbers are stored for each joint: x, y, z, c where (x, y) are screen coordinates,
        z is the depth value, and c is the confidence score. If a depth sequence has n frames, then the number of real
        numbers stored in the corresponding skeleton file is equal to: n*20*4.
    """
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        self._joints_per_person = _SensorJointNumber.kinect_v1

        # No text description in dataset files
        self._labels = ('high arm wave', 'horizontal arm wave', 'hammer', 'hand catch', 'forward punch', 'high throw',
                        'draw x', 'draw tick', 'draw circle', 'hand clap', 'two hand wave', 'side-boxing', 'bend',
                        'forward kick', 'side kick', 'jogging', 'tennis swing', 'tennis serve', 'golf swing', 'pickup',
                        'throw')
        _, data_filenames = get_folders_and_files(self.root_dir)
        for data_filename in data_filenames:
            if not data_filename.endswith('.txt'):
                continue
            action_id = int(data_filename.split('_')[0][1:])
            label = self._labels[action_id - 1]
            seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
            with open(self.root_dir + data_filename, 'r') as data_file:
                frame = []
                for line_idx, line in enumerate(data_file, 0):
                    if line.strip() == '':
                        break   # EOF
                    if line_idx != 0 and line_idx % self._joints_per_person == 0:
                        seq = np.vstack((seq, np.array(frame, dtype=np.float32)))
                        frame = []
                    temp = line.split()
                    frame.append(_SCALE_FACTOR * float(temp[0]) * 1000 + _COORDS_OFFSET)    # x in mm
                    frame.append(_SCALE_FACTOR * float(temp[1]) * 1000 + _COORDS_OFFSET)    # y in mm
                    frame.append(_SCALE_FACTOR * float(temp[2]) * 1000 + _COORDS_OFFSET)    # z in mm
                self._data_label_pairs.append((seq, label))

    def load_data_by_idx(self, data_idx: int):
        return self._data_label_pairs[data_idx][0]

    def __str__(self):
        return 'MSR_Action3D_Dataset'


class SkeletonDatasetOAD(SkeletonDataset):
    # TXT-based skeleton data
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        raise NotImplementedError('This method is only implemented by subclass')

    def load_data_by_idx(self, data_idx: int):
        raise NotImplementedError('This method is only implemented by subclass')

    def __str__(self):
        return 'UTD_Dataset'


class SkeletonDatasetUTD(SkeletonDataset):
    # TXT-based skeleton data
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        raise NotImplementedError('This method is only implemented by subclass')

    def load_data_by_idx(self, data_idx: int):
        raise NotImplementedError('This method is only implemented by subclass')

    def __str__(self):
        return 'UTD_Dataset'


class SkeletonDatasetUCF(SkeletonDataset):
    # Specialized ske-based skeleton data
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        raise NotImplementedError('This method is only implemented by subclass')

    def load_data_by_idx(self, data_idx: int):
        raise NotImplementedError('This method is only implemented by subclass')

    def __str__(self):
        return 'UCF_Dataset'


class SkeletonDatasetUTKinect(SkeletonDataset):
    """
    Load UTKinect-Action3D Dataset (untrimmed) by:
        L. Xia, C.-C. Chen, and J. K. Aggarwal, "View invariant human action recognition using histograms of 3d
        joints," in Computer vision and pattern recognition workshops (CVPRW), 2012 IEEE computer society conference
        on, 2012, pp. 20-27: IEEE.

    Dataset description:
        The videos was captured using a single stationary Kinect with Kinect for Windows SDK Beta Version. There are
        10 action types: walk, sit down, stand up, pick up, carry, throw, push, pull, wave hands, clap hands. There
        are 10 subjects, Each subject performs each actions twice. Three channels were recorded: RGB, depth and
        skeleton joint locations. The three channel are synchronized. The framerate is 30f/s. Note we only recorded
        the frames when the skeleton was tracked, so the frame number of the files has jumps. The final frame rate is
        about 15f/sec. (There is around 2% of the frames where there are multiple skeleton info recorded with
        slightly different joint locations. This is not caused by a second person. You can chooce either one.)

        In each video, the subject performs the 10 actions in a concatenate fation, the label of the each action
        segment is given in actionLabel.txt. The dataset contains 4 parts:

            (a) and (b) are RGB-D

            (c) Sketetal joint Locations (.txt):
                Each row contains the data of one frame, the first number is frame number, the following numbers
                are the (x,y,z) locations of joint 1-20. The x, y, and z are the coordinates relative to the sensor
                array, in meters.

            (d) Labels of action sequence (4KB)

    """
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        self._joints_per_person = _SensorJointNumber.kinect_v1

        # As the data loading also involves label tensor (where label are final indices) generations, pre-sorting is
        # required.
        self._labels = sorted(('walk', 'sitDown', 'standUp', 'pickUp', 'carry', 'throw', 'push', 'pull',
                               'waveHands', 'clapHands'))
        file_id = ''
        with open(self.root_dir + 'actionLabel.txt', 'r') as label_file:
            labels_and_time = []
            for line_idx, line in enumerate(label_file, 0):
                line = line.rstrip().replace(':', '')
                if line.strip() == '':
                    break   # EOF
                if line[0] == 's' and line[1].isdigit():
                    if line_idx != 0:
                        # store the last sequence
                        seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person * 3)
                        frame_label_indices = []
                        visited_frame_indices = []  # for discarding duplicate frames
                        with open(self.root_dir + "joints/" + "joints_" + file_id + ".txt") as data_file:
                            for data_line in data_file:
                                if data_line.rstrip() == '':
                                    break  # EOF
                                temp_list = data_line.split()
                                current_frame_idx = int(temp_list[0])
                                if current_frame_idx not in visited_frame_indices:
                                    frame = []
                                    for joint_idx in range(self._joints_per_person):
                                        frame.append(_SCALE_FACTOR * float(
                                            temp_list[joint_idx * 3 + 1]) * 1000. + _COORDS_OFFSET)  # x
                                        frame.append(_SCALE_FACTOR * float(
                                            temp_list[joint_idx * 3 + 2]) * 1000. + _COORDS_OFFSET)  # y
                                        frame.append(_SCALE_FACTOR * float(
                                            temp_list[joint_idx * 3 + 3]) * 1000. + _COORDS_OFFSET)  # z
                                    seq = np.vstack((seq, np.array(frame, dtype=np.float32)))
                                    for idx, time_label_pair in enumerate(labels_and_time, 0):
                                        if current_frame_idx >= time_label_pair[1]:
                                            if current_frame_idx <= time_label_pair[2]:
                                                # frame belongs to an action class
                                                frame_label_indices.append(self._labels.index(time_label_pair[0]))
                                                break
                                            else:
                                                if idx != len(labels_and_time) - 1:
                                                    next_time_label_pair = labels_and_time[idx + 1]
                                                    if current_frame_idx < next_time_label_pair[1]:
                                                        # unknown action
                                                        frame_label_indices.append(self.label_size())
                                                        break
                                                else:
                                                    # unknown action
                                                    frame_label_indices.append(self.label_size())
                                                    break
                                        else:
                                            # unknown action at the very beginning of the sequence
                                            frame_label_indices.append(self.label_size())
                                            break
                                    visited_frame_indices.append(current_frame_idx)
                        self._data_label_pairs.append((seq, np.array(frame_label_indices, dtype=np.int16)))

                    # prepare to record the next sequence
                    file_id = line
                    labels_and_time = []
                    continue
                temp_list = line.split()
                label_name = temp_list[0]
                if temp_list[1] == 'NaN' or temp_list[2] == 'NaN':  # Invalid frame numbers
                    continue
                labels_and_time.append((label_name, int(temp_list[1]), int(temp_list[2])))

    def load_data_by_idx(self, data_idx: int):
        return self._data_label_pairs[data_idx]

    @overrides
    def get_data_label_pair(self, idx):
        return self.load_data_by_idx(idx)

    def __str__(self):
        return 'UTKinect_Dataset'


class SkeletonDatasetFlorence(SkeletonDataset):
    """
    Load Florence 3D Actions Dataset (trimmed) by:
        L. Seidenari, V. Varano, S. Berretti, A. Bimbo, and P. Pala,
        "Recognizing actions from depth cameras as weakly aligned multi-part bag-of-poses," in Proceedings of
        the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2013, pp. 479-485.

    Dataset description:
        The dataset collected at the University of Florence during 2012, has been captured using a Kinect camera.
        It includes 9 activities: wave, drink from a bottle, answer phone, clap, tight lace, sit down, stand up,
        read watch, bow. During acquisition, 10 subjects were asked to perform the above actions for 2/3 times. This
        resulted in a total of 215 activity samples. We suggest a leave-one-actor-out protocol: train your
        classifier using all the sequences from 9 out of 10 actors and test on the remaining one. Repeat this
        procedure for all actors and average the 10 classification accuracy values.

        The file Florence_dataset_WorldCoordinates.txt contains the world coordinates for all the joints. Thanks to
        Maxime Devanne for parsing this data! Each line is formatted according to the following:

            %idvideo idactor idcategory  f1....fn
            where f1-f45 are world coordinates of all the 15 joints.

            Specifically:
                Head: f1-f3
                Neck: f4-f6
                Spine: f7-f9
                Left Shoulder: f10-f12
                Left Elbow: f13-f15
                Left Wrist: f16-f18
                Right Shoulder: f19-f21
                Right Elbow: f22-f24
                Right Wrist: f25-f27
                Left Hip: f28-f30
                Left Knee: f31-f33
                Left Ankle: f34-f36
                Right Hip: f37-f39
                Right Knee: f40-f42
                Right Ankle: f43-f45
    """
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        self._joints_per_person = _SensorJointNumber.openni

        # No text description in dataset files
        self._labels = ('wave', 'drink from a bottle', 'answer phone', 'clap', 'tight lace', 'sit down',
                        'stand up', 'read watch', 'bow')
        with open(self.root_dir + 'Florence_dataset_WorldCoordinates.txt', 'r') as label_file:
            video_idx = 0
            seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
            for line in label_file:
                if line.rstrip() == '':
                    break   # EOF
                temp_list = line.split()
                if video_idx != int(temp_list[0]) - 1:   # has reached next video
                    self._data_label_pairs.append((seq, self._labels[int(temp_list[2]) - 1]))
                    seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
                    video_idx = int(temp_list[0]) - 1
                frame = []
                for joint_idx in range(self._joints_per_person):
                    frame.append(_SCALE_FACTOR * float(temp_list[joint_idx*3 + 3]) + _COORDS_OFFSET)       # x
                    frame.append(_SCALE_FACTOR * float(temp_list[joint_idx*3 + 4]) + _COORDS_OFFSET)       # y
                    frame.append(_SCALE_FACTOR * float(temp_list[joint_idx*3 + 5]) + _COORDS_OFFSET)       # z
                seq = np.vstack((seq, np.array(frame, dtype=np.float32)))

    def load_data_by_idx(self, data_idx: int):
        return self._data_label_pairs[data_idx][0]

    def __str__(self):
        return 'Florence_Dataset'


class SkeletonDatasetCornell60(SkeletonDataset):
    """
    Load Cornell Activity Dataset 60 (CAD-60, trimmed) by:
        J. Sung, C. Ponce, B. Selman, and A. Saxena, "Human Activity Detection from RGBD Images,"
        plan, activity, and intent recognition, vol. 64, 2011.

    Dataset description:
        Each zipped directory contains following four types of files collected from one person.
        1) activityLabel.txt
            - A file that maps each # to corresponding activity. (# is a ten-digit integer)

        2) #.txt
            - Skeleton data files.
            - Skeleton data format details are in the next section of this README.

        3) and 4) are RGB-D

        Skeleton Data Format:
            Skeleton data consists of 15 joints. There are 11 joints that have both
            joint orientation and joint position. And, 4 joints that only have joint
            position. Each row follows the following format.

            Frame#,ORI(1),P(1),ORI(2),P(2),...,P(11),J(11),P(12),...,P(15)

                Frame# => integer starting from 1

                ORI(i) => orientation of ith joint
                            0 1 2
                            3 4 5
                            6 7 8
                          3x3 matrix is stored as followed by CONF
                            0,1,2,3,4,5,6,7,8,CONF

                P(i)   => position of ith joint followed by CONF
                            x,y,z,CONF
                          values are in millimeters

                CONF   => boolean confidence value (0 or 1)

                Joint number -> Joint name
                         1 -> HEAD
                         2 -> NECK
                         3 -> TORSO
                         4 -> LEFT_SHOULDER
                         5 -> LEFT_ELBOW
                         6 -> RIGHT_SHOULDER
                         7 -> RIGHT_ELBOW
                         8 -> LEFT_HIP
                         9 -> LEFT_KNEE
                        10 -> RIGHT_HIP
                        11 -> RIGHT_KNEE
                        12 -> LEFT_HAND
                        13 -> RIGHT_HAND
                        14 -> LEFT_FOOT
                        15 -> RIGHT_FOOT
    """
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        self._joints_per_person = _SensorJointNumber.openni
        for folder_idx in range(4):
            data_dir = self.root_dir + "data" + str(folder_idx + 1) + "/"
            with open(data_dir + 'activityLabel.txt', 'r') as label_file:
                for line in label_file:
                    if line.startswith('END'):
                        break   # EOF
                    segmented = line.rstrip().split(',')
                    label_name = segmented[1]
                    self._labels.add(label_name)
                    filename = data_dir + segmented[0] + '.txt'
                    seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
                    with open(filename) as data_file:
                        # TODO fix array length
                        for l in data_file:
                            if l.startswith('END'):
                                break   # EOF
                            frame = []
                            temp_list = l.split(',')
                            for full_info_joint_idx in range(11):
                                frame.append(_SCALE_FACTOR * float(
                                    temp_list[14 * full_info_joint_idx + 11]) + _COORDS_OFFSET)  # x
                                frame.append(_SCALE_FACTOR * float(
                                    temp_list[14 * full_info_joint_idx + 12]) + _COORDS_OFFSET)  # y
                                frame.append(_SCALE_FACTOR * float(
                                    temp_list[14 * full_info_joint_idx + 13]) + _COORDS_OFFSET)  # z
                            for part_info_joint_idx in range(4):
                                frame.append(_SCALE_FACTOR * float(
                                    temp_list[4 * part_info_joint_idx - 17]) + _COORDS_OFFSET)  # x
                                frame.append(_SCALE_FACTOR * float(
                                    temp_list[4 * part_info_joint_idx - 16]) + _COORDS_OFFSET)  # y
                                frame.append(_SCALE_FACTOR * float(
                                    temp_list[4 * part_info_joint_idx - 15]) + _COORDS_OFFSET)  # z
                            seq = np.vstack((seq, np.array(frame, dtype=np.float32)))
                    self._data_label_pairs.append((seq, label_name))

    def load_data_by_idx(self, data_idx: int):
        return self._data_label_pairs[data_idx][0]

    def __str__(self):
        return 'Cornell_60_Dataset'


class SkeletonDatasetUPCV(SkeletonDataset):
    """
    Load UPCV Action Dataset (trimmed) by:
        I. Theodorakopoulos, D. Kastaniotis, G. Economou, and S. Fotopoulos,
        "Pose-based human action recognition via sparse representation in dissimilarity space,"
        Journal of Visual Communication and Image Representation, vol. 25, no. 1, pp. 12-23, 2014.

    Dataset description:
        Actions:
            In this dataset 10 actions performed two times by 20 subjects (10 males and 10 females.) Totally 400
            action sequences where executed. Actions names are:
                [walk, seat, grab, phone, watch_clock, scratch_head, cross_arms, punch, kick, wave]

        Data:
            Skeletal data for every action are saved in separate .txt files with filenames action_{#}.txt in the
            data folder (where # denotes the action number). In every file, an action is represented with a
            sequence of frames in the form of a 20 joints skeletal model. Thus, every 20 lines a model of the human
            body consisted by 20 three dimensional joints (x,y,z) is provided. Joints are saved using the
            following sequence:
                Head
                ShoulderCenter
                ShoulderLeft
                ShoulderRight
                ElbowLeft
                ElbowRight
                WristLeft
                WristRight
                HandLeft
                HandRight
                Spine
                HipCenter
                HipLeft
                HipRight
                KneeLeft
                KneeRight
                AnkleLeft
                AnkleRight
                FootLeft
                FootRight
    """
    def __init__(self, directory: str, train_portion: float=0.9):
        super().__init__(directory, train_portion)

    def load_label_map(self):
        self._joints_per_person = _SensorJointNumber.kinect_v1
        data_dir = self.root_dir + 'dataset/'
        with open(self.root_dir + 'actions.txt', 'r') as label_file:
            for line_idx, line in enumerate(label_file):
                label_name = line[1:].rstrip()
                self._labels.add(label_name)
                filename = data_dir + 'action_' + str(line_idx + 1) + '.txt'
                self._data_label_pairs.append((filename, label_name))

    def load_data_by_idx(self, data_idx: int):
        seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
        with open(self._data_label_pairs[data_idx][0]) as data_file:
            frame = []
            for line_idx, line in enumerate(data_file):
                if line.strip() == '':
                    break   # EOF
                if line_idx != 0 and line_idx % self._joints_per_person == 0:
                    # iterator has reached the first joint in the next frame
                    seq = np.vstack((seq, np.array(frame, dtype=np.float32)))
                    frame = []
                temp_list = line.split(',')
                frame.append(_SCALE_FACTOR * float(temp_list[0])*1000. + _COORDS_OFFSET)     # x
                frame.append(_SCALE_FACTOR * float(temp_list[1])*1000. + _COORDS_OFFSET)     # y
                frame.append(_SCALE_FACTOR * float(temp_list[2])*1000. + _COORDS_OFFSET)     # z
        return seq

    def __str__(self):
        return 'UPCV_Dataset'
