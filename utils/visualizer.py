import pickle
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
from .processing import (world_coords_array_to_image,
                         standardize_coordinate_origin_sequence)
from dataset.skeleton_abstract import SkeletonDataset
from global_configs import SensorJointNumber, SKELETON_EDGES, MIN_EVENT_PROBABILITY


__all__ = ['visualize_result']
np.set_printoptions(suppress=True, threshold=20000)


def visualize_result(original_dataset: SkeletonDataset,
                     trained_model: nn.Module,
                     device: torch.device,
                     fps: float = 30):
    labels = original_dataset.get_labels()
    null_class = original_dataset.label_size - 1
    random.seed()
    test_indices = original_dataset.indices_test
    random.shuffle(test_indices)
    delay = round((1 / fps) * 1000)
    trained_model.eval()
    joint_number = original_dataset.get_joint_number()
    joints_per_skeleton = joint_number
    if original_dataset.has_interaction:
        joint_number *= 2
    with torch.no_grad():
        for i in test_indices:
            data, label, confidence = original_dataset[i]
            seq_len = data.shape[0]
            data, label = data.numpy(), label.numpy()
            test_data = standardize_coordinate_origin_sequence(data.copy())
            test_data = torch.as_tensor(test_data, dtype=torch.float32).to(device)
            print('New sequence starts!')
            for frame_idx in range(seq_len):
                display_frame = data[frame_idx].reshape(joint_number, 3)
                test_frame = test_data[frame_idx]
                _, s_out, r_out = trained_model(test_frame[None, None, ...])
                ground_event, pred_event = '', ''

                class_probability, class_prediction = torch.max(s_out, -1)
                class_probability, class_prediction = class_probability.item(), class_prediction.item()
                start_probabilities, end_probabilities = torch.clamp(r_out[-1], min=0, max=1).transpose(0, 1)
                start_probability, start_class = torch.max(start_probabilities, 0)
                end_probability, end_class = torch.max(end_probabilities, 0)
                if start_probability >= MIN_EVENT_PROBABILITY and start_probability > end_probability:
                    pred_event = 'Action starting'
                    if class_prediction == null_class:
                        class_prediction = start_class.item()
                elif end_probability >= MIN_EVENT_PROBABILITY and end_probability > start_probability:
                    pred_event = 'Action ending'
                    if class_prediction == null_class:
                        class_prediction = end_class.item()
                elif class_prediction != null_class:
                    pred_event = 'Action ongoing'
                elif class_prediction == null_class:
                    pred_event = 'No action'
                predicted_label = labels[class_prediction]

                ground_truth_label_id = int(label[frame_idx])
                true_start_probs, true_end_probs = confidence[frame_idx].transpose(0, 1)
                true_start_prob, true_start_class = torch.max(true_start_probs, 0)
                true_end_prob, true_end_class = torch.max(true_end_probs, 0)
                if true_start_prob >= MIN_EVENT_PROBABILITY * 0.5 and true_start_prob > true_end_prob:
                    ground_event = 'Action starting'
                    if ground_truth_label_id == null_class:
                        ground_truth_label_id = true_start_class.item()
                elif true_end_prob >= MIN_EVENT_PROBABILITY * 0.5 and true_end_prob > true_start_prob:
                    ground_event = 'Action ending'
                    if ground_truth_label_id == null_class:
                        ground_truth_label_id = true_end_class.item()
                elif ground_truth_label_id != null_class:
                    ground_event = 'Action ongoing'
                elif ground_truth_label_id == null_class:
                    ground_event = 'No action'
                ground_truth_label = labels[ground_truth_label_id]
                img = np.zeros([480, 640, 3], dtype=np.uint8)
                img_coords = world_coords_array_to_image(display_frame, 531.15, 640, 480)
                for joint_pos in img_coords:
                    cv2.circle(img, tuple(joint_pos), 4, (255, 255, 0), thickness=2)
                for joint_pair in SKELETON_EDGES:
                    idx1, idx2 = joint_pair
                    cv2.line(img, tuple(img_coords[idx1]), tuple(img_coords[idx2]),
                             (255, 255, 0), thickness=1)
                    if original_dataset.has_interaction:
                        cv2.line(img,
                                 tuple(img_coords[idx1+joints_per_skeleton]),
                                 tuple(img_coords[idx2+joints_per_skeleton]),
                                 (255, 255, 0), thickness=1)
                img = cv2.resize(img, dsize=(1280, 960))
                info_string = 'Ground Truth Class: %s \n' \
                              'Predicted Class: %s \n' \
                              'Predicted Event: %s \n' \
                              'Ground Truth Event: %s' \
                              % (ground_truth_label,
                                 predicted_label,
                                 pred_event,
                                 ground_event)
                put_mutiple_lines_of_text(img, info_string, 13, 23)
                info_string2 = 'Predicted Class Probability: %.2f %% \n' \
                               'Predicted Start Probability: %.2f %% \n' \
                               'Predicted End Probability: %.2f %%' \
                               % (class_probability * 100,
                                  start_probability.item() * 100,
                                  end_probability.item() * 100)
                put_mutiple_lines_of_text(img, info_string2, 1280 - 510, 23)
                cv2.namedWindow('Skeleton Visualizer')
                cv2.imshow('Skeleton Visualizer', img)
                if cv2.waitKey(delay) == 27:
                    break
            print('Current sequence ends!')


def put_mutiple_lines_of_text(frame: np.ndarray, text: str, x: int, y: int, font_scale: float = 0.77):
    position = (x, y)
    color = (0, 255, 255)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    line_height = text_size[1] + 5
    x, y0 = position
    for i, line in enumerate(text.split("\n")):
        y = y0 + i * line_height
        cv2.putText(frame,
                    line,
                    (x, y),
                    font,
                    font_scale,
                    color,
                    thickness,
                    line_type)
