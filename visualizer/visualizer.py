# TODO 1. Implement parent visualizer class
# TODO 2. Integrate to GUI


import cv2
import numpy as np
import re
import enum
from jcr.rnn import JcrGRU
from jcr.trainer import load_checkpoint
from jcr.evaluator import evaluate_frame
from utils.misc import get_folders_and_files
from utils.converter import *


class _SensorJointNumber(enum.IntEnum):
    openni = 15
    kinect_v1 = 20
    kinect_v2 = 25


class Visualizer(object):
    def __init__(self, directory: str, trained_model, labels):
        pass


class VisualizerUTKinect(Visualizer):
    def __init__(self, directory: str, trained_model, labels):
        super().__init__(directory, trained_model, labels)
        directory.replace('\\', '/')
        if directory[-1] != '/':
            directory += '/'
        self.root_dir = directory
        self.seq_ids = list()
        self._joints_per_person = _SensorJointNumber.kinect_v1
        self._val_model = trained_model
        self._labels = labels

        self.load_sequence_list()
        self.seq_ids = sorted(self.seq_ids)

    def load_sequence_list(self):
        rgb_dir = self.root_dir + 'RGB/'
        seq_dirs, _ = get_folders_and_files(rgb_dir)
        self.seq_ids.extend(seq_dirs)

    def visualize_sequence(self, idx):
        seq_dir = self.root_dir + 'RGB/' + self.seq_ids[idx] + '/'
        _, file_list = get_folders_and_files(seq_dir)
        img_ids = []
        for filename in file_list:
            if filename.endswith('.jpg'):
                img_ids.append(int(re.findall(r'\d+', filename)[0]))
            else:
                continue
        img_ids = sorted(img_ids)
        not_visited_joint_frames = []
        joint_seq = np.array([], dtype=np.float32).reshape(0, self._joints_per_person*3)
        with open(self.root_dir + "joints/" + "joints_" + self.seq_ids[idx] + ".txt") as joint_file:
            not_visited_joint_frames.extend(img_ids)
            for line in joint_file:
                line = line.rstrip()
                if line == '':  # end of file
                    break
                temp_list = line.split()
                current_frame_idx = int(temp_list[0])
                if current_frame_idx in not_visited_joint_frames:
                    not_visited_joint_frames.remove(current_frame_idx)  # There can be frame duplicates
                    frame = []
                    for joint_idx in range(self._joints_per_person):
                        frame.append(float(temp_list[joint_idx * 3 + 1]) * 1000.)     # x
                        frame.append(float(temp_list[joint_idx * 3 + 2]) * 1000.)     # y
                        frame.append(float(temp_list[joint_idx * 3 + 3]) * 1000.)     # z
                    joint_seq = np.vstack((joint_seq, np.array(frame, dtype=np.float32)))
        assert len(joint_seq) == len(img_ids), 'Image sequence length and joint sequence length do not match.'
        joint_seq_3d = reshape_2d_sequence_to_3d(joint_seq)
        for frame_idx, img_id in enumerate(img_ids, 0):
            joints = joint_seq_3d[frame_idx]
            pred_label_idx, pred_label_conf, _, _ = evaluate_frame(joint_seq[frame_idx], self._val_model)
            img = cv2.imread(seq_dir + 'colorImg' + str(img_id) + '.jpg')
            cv2.putText(img,
                        'Class prediction: ' + self._labels[pred_label_idx] +
                        ', Confidence: ' + str(round(pred_label_conf * 100., 3)) + '%',
                        (0, 20),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (0, 64, 224),
                        2)
            for joint_coords in joints:
                joint_img_coords = world_coords_to_image(joint_coords, 531.15, 640, 480)
                cv2.circle(img, joint_img_coords, 5, (255, 255, 0), thickness=2)
            cv2.namedWindow('UTKinect')
            cv2.imshow('UTKinect', img)
            if cv2.waitKey(34) == 27:
                break
