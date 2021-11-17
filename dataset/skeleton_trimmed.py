import random
import numpy as np
from .skeleton_abstract import SkeletonDataset
from global_configs import DatasetProtocol, SensorJointNumber
from utils.misc import ProgressBar, get_folders_and_files
from utils.processing import (preprocess_skeleton_frame,
                              causal_savitzky_golay_filter,
                              standardize_coordinate_origin_sequence)


__all__ = ['SkeletonDatasetNTURGBD', 'SkeletonDatasetLSCD', 'SkeletonDatasetCornell60', 'SkeletonDatasetFlorence',
           'SkeletonDatasetMSRAction3D', 'SkeletonDatasetUPCV', 'SkeletonDatasetTST']


class SkeletonDatasetNTURGBD(SkeletonDataset):
    """
    Load the NTU-RGB+D Dataset by:
        A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang, "NTU RGB+ D: A large scale dataset for 3D human activity
        analysis," in Proceedings of the IEEE conference on computer vision and pattern recognition,
        2016, pp. 1010-1019.

    Dataset description:
        Each file/folder name in the dataset is in the format of SsssCcccPpppRrrrAaaa (e.g. S001C002P003R002A013),
        for which sss is the setup number, ccc is the camera ID, ppp is the performer ID, rrr is the replication
        number (1 or 2), and aaa is the action class label. For more details about setups, camera IDs, etc, please
        refer to the paper.

        302 of the captured samples have missing or incomplete skeleton data. Please refer to
        samples_with_missing_skeletons.txt.

    """
    def __init__(self, directory: str, protocol: DatasetProtocol,
                 regression_sigma: float = 5,
                 is_translated: bool = True,
                 is_edged: bool = False,
                 is_rotated: bool = False,
                 is_filtered: bool = False):
        super(SkeletonDatasetNTURGBD, self).__init__(directory, regression_sigma, 0.5,
                                                     is_translated=is_translated,
                                                     is_edged=is_edged,
                                                     is_rotated=is_rotated,
                                                     is_filtered=is_filtered)
        self._protocol = protocol

    def load_label_map(self):
        pass

    def load_data_by_idx(self, data_idx: int):
        pass

    def __str__(self):
        return 'NTU-RGB+D_Dataset'


class SkeletonDatasetLSCD(SkeletonDataset):
    """
    Load the Large Scale RGB-D Dataset for Action Recognition by:
        J. Zhang, W. Li, P. Wang, P. Ogunbona, S. Liu, and C. Tang, "A large scale rgb-d dataset for action
        recognition," in International Workshop on Understanding Human Activities through 3D Sensors,
        2016, pp. 101-114: Springer.

    Dataset description:
         It combines the following datasets:
            - MSRAction3DExt
            - UTKinect
            - MSRDailyActivity
            - MSR ActionPair
            - CAD120
            - CAD60
            - G3D
            - RGBD-HuDa (No skeleton provided)
            - UTD-MHAD
    """
    def __init__(self, directory: str,
                 regression_sigma: float = 5,
                 is_translated: bool = True,
                 is_edged: bool = False,
                 is_rotated: bool = False,
                 is_filtered: bool = False,
                 *args,
                 **kwargs):
        super(SkeletonDatasetLSCD, self).__init__(directory, regression_sigma, 0.5,
                                                  is_translated=is_translated,
                                                  is_edged=is_edged,
                                                  is_rotated=is_rotated,
                                                  is_filtered=is_filtered,
                                                  downsample_factor=1)

    def load_label_map(self):
        self._joints_per_person = SensorJointNumber.KINECT_V1
        self._labels = (
            'wave', 'horizontal arm wave', 'hammer',
            'hand catch', 'forward punch', 'throw',
            'draw x', 'draw tick', 'draw circle (clockwise)',
            'hand clap', 'two hand high wave', 'side-boxing',
            'bend', 'forward kick', 'side kick',
            'jogging', 'tennis swing forehand', 'tennis serve',
            'golf swing', 'pickup & throw', 'walk',
            'sit down', 'stand up', 'pick up from floor',
            'carry', 'push', 'pull',
            'drink', 'eat', 'read book',
            'talking on the phone', 'write on a paper', 'working on computer',
            'use vacuum cleaner', 'cheer up', 'play game',
            'lay down on sofa', 'play guitar', 'pick up from table',
            'put down to table', 'place a box', 'push a chair',
            'pull a chair', 'wear a hat', 'take off a hat',
            'put on a backpack', 'take off a backpack', 'stick a poster',
            'remove a poster', 'making cereal', 'taking medicine',
            'stacking objects', 'unstacking objects', 'microwaving food',
            'cleaning objects', 'taking food', 'arranging objects',
            'having a meal', 'writing on whiteboard', 'rinsing mouth',
            'brushing teeth', 'wearing contact lens', 'talking on couch',
            'cooking(chopping)', 'cooking(stirring)', 'opening pill container',
            'defend', 'tennis swing backhand', 'throw bowling ball',
            'aim and fire gun', 'jump', 'climb',
            'crouch', 'steer a car', 'flap',
            'mop the floor', 'enter the room', 'exit the room',
            'get up from bed', 'take off the jacket', 'put on the jacket',
            'swipe to the left', 'swipe to the right', 'cross arms in the chest',
            'basketball shoot', 'draw circle(counter clockwise)', 'draw triangle',
            'front boxing', 'baseball swing', 'arm curl',
            'two hand push', 'knock on door', 'forward lunge',
            'squat with two arms stretch out'
        )
        self._class_sample_counts = {
            'wave': 131, 'horizontal arm wave': 69, 'hammer': 69,
            'hand catch': 100, 'forward punch': 128, 'throw': 140,
            'draw x': 99, 'draw tick': 69, 'draw circle (clockwise)': 101,
            'hand clap': 151, 'two hand high wave': 89, 'side-boxing': 69,
            'bend': 70, 'forward kick': 129, 'side kick': 59,
            'jogging': 131, 'tennis swing forehand': 131, 'tennis serve': 131,
            'golf swing': 99, 'pickup & throw': 103, 'walk': 101,
            'sit down': 72, 'stand up': 72, 'pick up from floor': 62,
            'carry': 19, 'push': 20, 'pull': 20,
            'drink': 20, 'eat': 20, 'read book': 20,
            'talking on the phone': 24, 'write on a paper': 20, 'working on computer': 24,
            'use vacuum cleaner': 20, 'cheer up': 20, 'play game': 20,
            'lay down on sofa': 20, 'play guitar': 20, 'pick up from table': 30,
            'put down to table': 30, 'place a box': 30, 'push a chair': 30,
            'pull a chair': 30, 'wear a hat': 30, 'take off a hat': 30,
            'put on a backpack': 30, 'take off a backpack': 30, 'stick a poster': 30,
            'remove a poster': 30, 'making cereal': 16, 'taking medicine': 12,
            'stacking objects': 12, 'unstacking objects': 12, 'microwaving food': 12,
            'cleaning objects': 12, 'taking food': 12, 'arranging objects': 12,
            'having a meal': 12, 'writing on whiteboard': 4, 'rinsing mouth': 4,
            'brushing teeth': 4, 'wearing contact lens': 8, 'talking on couch': 4,
            'cooking(chopping)': 4, 'cooking(stirring)': 4, 'opening pill container': 12,
            'defend': 30, 'tennis swing backhand': 30, 'throw bowling ball': 61,
            'aim and fire gun': 90, 'jump': 30, 'climb': 30,
            'crouch': 30, 'steer a car': 30, 'flap': 30,
            'mop the floor': 0, 'enter the room': 0, 'exit the room': 0,
            'get up from bed': 0, 'take off the jacket': 0, 'put on the jacket': 0,
            'swipe to the left': 32, 'swipe to the right': 32, 'cross arms in the chest': 32,
            'basketball shoot': 32, 'draw circle(counter clockwise)': 32, 'draw triangle': 32,
            'front boxing': 32, 'baseball swing': 32, 'arm curl': 32,
            'two hand push': 32, 'knock on door': 32, 'forward lunge': 32,
            'squat with two arms stretch out': 31,
            # Might use this dict for cross entropy or data sampler weighting
        }
        pr = ProgressBar(80, len(self._labels))
        for action_idx, action_name in enumerate(self._labels):
            action_dir = self.root_dir + 'Skeleton/action%d/' % (action_idx + 1)
            _, filenames = get_folders_and_files(action_dir)
            random.shuffle(filenames)
            train_count = 0
            for filename in filenames:
                with open(action_dir + filename, 'r') as data_file:
                    need_conversion = False
                    seq_len = int(data_file.readline().rstrip())
                    num_joints = int(data_file.readline().rstrip())
                    if num_joints == 15:
                        need_conversion = True
                    _ = data_file.readline().rstrip()   # dimensions of each joint
                    seq = np.zeros((seq_len, self._joints_per_person * 3), dtype=np.float32)
                    frame = []
                    curr_frame_idx = 0
                    for line_idx, line in enumerate(data_file, 3):
                        x, y, z = line.split()[:3]
                        frame += [float(x), float(y), float(z)]
                        if len(frame) == num_joints * 3:
                            seq[curr_frame_idx] = preprocess_skeleton_frame(frame,
                                                                            is_15_joint=need_conversion,
                                                                            is_cad=True,
                                                                            to_edge=self.is_edged,
                                                                            to_rotate=self.is_rotated)
                            frame = []
                            curr_frame_idx += 1
                    if self.is_filtered:
                        seq = causal_savitzky_golay_filter(seq)
                    if self.is_translated and not self.is_edged:
                        seq = standardize_coordinate_origin_sequence(seq)
                    self._update_extrema(seq)
                    self._data_label_pairs.append((seq, action_name))
                    train_count += 1
                    curr_dataset_idx = len(self._data_label_pairs) - 1
                    if train_count <= 0.5 * self._class_sample_counts[action_name]:
                        self._train_indices.append(curr_dataset_idx)
                    else:
                        self._test_indices.append(curr_dataset_idx)
            pr.update(action_idx + 1)

    def load_data_by_idx(self, data_idx: int):
        return self._data_label_pairs[data_idx][0]

    def __str__(self):
        return 'LSCD_Dataset'


class SkeletonDatasetTST(SkeletonDataset):
    def __init__(self, directory: str, protocol: DatasetProtocol, regression_sigma: float = 5):
        super(SkeletonDatasetTST, self).__init__(directory, regression_sigma, 0.5)
        self._protocol = protocol

    def load_label_map(self):
        pass

    def load_data_by_idx(self, data_idx: int):
        pass

    def __str__(self):
        return 'TST_Fall_Detection_v2_Dataset'


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
    def __init__(self, directory: str, train_portion: float = 0.5, regression_sigma: float = 5):
        super(SkeletonDatasetMSRAction3D, self).__init__(directory, regression_sigma, train_portion)

    def load_label_map(self):
        self._joints_per_person = SensorJointNumber.KINECT_V1

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
                    frame.append(float(temp[0]) * 1000)    # x in mm
                    frame.append(float(temp[1]) * 1000)    # y in mm
                    frame.append(float(temp[2]) * 1000)    # z in mm
                self._data_label_pairs.append((seq, label))
                self._update_extrema(seq)

    def load_data_by_idx(self, data_idx: int):
        return self._data_label_pairs[data_idx][0]

    def __str__(self):
        return 'MSR_Action3D_Dataset'


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
    def __init__(self, directory: str,
                 train_portion: float = 0.5,
                 regression_sigma: float = 5,
                 is_translated: bool = True,
                 is_edged: bool = False):
        super(SkeletonDatasetFlorence, self).__init__(directory, regression_sigma, train_portion,
                                                      is_translated=is_translated,
                                                      is_edged=is_edged)

    def load_label_map(self):
        # self._joints_per_person = SensorJointNumber.OPENNI
        self._joints_per_person = SensorJointNumber.KINECT_V1

        # No text description in dataset files
        self._labels = ('wave', 'drink from a bottle', 'answer phone', 'clap', 'tight lace', 'sit down',
                        'stand up', 'read watch', 'bow')
        with open(self.root_dir + 'Florence_dataset_WorldCoordinates.txt', 'r') as label_file:
            video_idx = 0
            last_label_idx = 0
            seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
            for line in label_file:
                if line.rstrip() == '':
                    break   # EOF
                temp_list = line.split()
                if video_idx != int(temp_list[0]) - 1:   # has reached next video
                    if self.is_filtered:
                        seq = causal_savitzky_golay_filter(seq)
                    if self.is_translated and not self.is_edged:
                        seq = standardize_coordinate_origin_sequence(seq)
                    self._data_label_pairs.append((seq, self._labels[last_label_idx]))
                    self._update_extrema(seq)
                    seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
                    video_idx = int(temp_list[0]) - 1
                frame = []
                for joint_idx in range(SensorJointNumber.OPENNI):
                    frame.append(float(temp_list[joint_idx*3 + 3]))       # x
                    frame.append(float(temp_list[joint_idx*3 + 4]))       # y
                    frame.append(float(temp_list[joint_idx*3 + 5]))       # z
                # seq = np.vstack((seq, joints_to_edges_deprecated(frame, is_from_ni=True)))
                seq = np.vstack((seq, preprocess_skeleton_frame(frame,
                                                                is_15_joint=True,
                                                                to_rotate=self.is_rotated,
                                                                to_edge=self.is_edged)))
                last_label_idx = int(temp_list[2]) - 1

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
    def __init__(self, directory: str, train_portion: float = 0.5, regression_sigma: float = 5):
        super(SkeletonDatasetCornell60, self).__init__(directory, regression_sigma, train_portion)

    def load_label_map(self):
        self._joints_per_person = SensorJointNumber.OPENNI
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
                    with open(filename, 'r') as data_file:
                        # TODO fix array length
                        for l in data_file:
                            if l.startswith('END'):
                                break   # EOF
                            frame = []
                            temp_list = l.split(',')
                            for full_info_joint_idx in range(11):
                                frame.append(float(
                                    temp_list[14 * full_info_joint_idx + 11]))  # x
                                frame.append(float(
                                    temp_list[14 * full_info_joint_idx + 12]))  # y
                                frame.append(float(
                                    temp_list[14 * full_info_joint_idx + 13]))  # z
                            for part_info_joint_idx in range(4):
                                frame.append(float(
                                    temp_list[4 * part_info_joint_idx - 17]))  # x
                                frame.append(float(
                                    temp_list[4 * part_info_joint_idx - 16]))  # y
                                frame.append(float(
                                    temp_list[4 * part_info_joint_idx - 15]))  # z
                            seq = np.vstack((seq, np.array(frame, dtype=np.float32)))
                    self._data_label_pairs.append((seq, label_name))
                    self._update_extrema(seq)

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
    def __init__(self, directory: str, train_portion: float = 0.5, regression_sigma: float = 5):
        super(SkeletonDatasetUPCV, self).__init__(directory, regression_sigma, train_portion)

    def load_label_map(self):
        self._joints_per_person = SensorJointNumber.KINECT_V1
        data_dir = self.root_dir + 'dataset/'
        with open(self.root_dir + 'actions.txt', 'r') as label_file:
            for line_idx, line in enumerate(label_file):
                label_name = line[1:].rstrip()
                self._labels.add(label_name)
                filename = data_dir + 'action_' + str(line_idx + 1) + '.txt'
                seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person * 3)
                with open(filename, 'r') as data_file:
                    frame = []
                    for data_line_idx, data_line in enumerate(data_file):
                        if data_line.strip() == '':
                            break  # EOF
                        if data_line_idx != 0 and data_line_idx % self._joints_per_person == 0:
                            # iterator has reached the first joint in the next frame
                            seq = np.vstack((seq, np.array(frame, dtype=np.float32)))
                            frame = []
                        temp_list = data_line.split(',')
                        frame.append(float(temp_list[0]) * 1000.)  # x
                        frame.append(float(temp_list[1]) * 1000.)  # y
                        frame.append(float(temp_list[2]) * 1000.)  # z
                self._data_label_pairs.append((seq, label_name))
                self._update_extrema(seq)

    def load_data_by_idx(self, data_idx: int):
        return self._data_label_pairs[data_idx][0]

    def __str__(self):
        return 'UPCV_Dataset'
