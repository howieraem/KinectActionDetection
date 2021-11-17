import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet import meter
from utils.pytorch import get_confidence_matrix, is_model_on_gpu
from dataset.skeleton import SkeletonDataset
import math
import warnings


def evaluate_frame(frame, model: nn.Module, regress_confidence_threshold=0.6, use_gpu: bool=True,
                   ground_truth: tuple=None):
    """

    :param frame:
    :param model:
    :param use_gpu:
    :param ground_truth: (start_conf, end_conf, label)
    :param regress_confidence_threshold:
    :return:
    """
    assert is_model_on_gpu(model) == use_gpu, 'Model not on the required device (CPU/GPU).'
    is_start = is_end = False
    frame = preprocess_array(frame, use_gpu)
    label_id, label_conf, start_conf, end_conf = recognize_frame(frame, model, regress_confidence_threshold, use_gpu)
    if start_conf >= regress_confidence_threshold:
        is_start = True
    if end_conf >= regress_confidence_threshold:
        is_end = True

    if ground_truth is not None:
        if ground_truth[0] and is_start:
            print('Prediction of start point correct.')
        if ground_truth[1] and is_end:
            print('Prediction of end point correct.')
        if label_id == ground_truth[2]:
            print('Prediction of class correct.')
        else:
            print('Prediction of class incorrect.')
    return label_id, label_conf, start_conf, end_conf


def preprocess_array(array, use_gpu):
    if type(array) == np.ndarray:
        array = torch.from_numpy(array)
    dim = len(array.shape)
    if dim < 3:
        for _ in range(3 - dim):
            array = array.unsqueeze(0)
    if use_gpu:
        array = array.cuda(0)
    return array


def postprocess_tensor(tensor: torch.tensor, use_gpu):
    if use_gpu:
        tensor = tensor.cpu()
    return tensor.numpy()


def recognize_frame(frame, model: nn.Module, regress_conf_threshold, use_gpu):
    model.eval()
    with torch.no_grad():
        c_out, s_out, r_out = model(Variable(frame))
    r_out = r_out.squeeze().permute(1, 0)   # results in dimensions (2, num_class + 1)
    pred_confs = postprocess_tensor(r_out.max(1)[0], use_gpu)
    pred_start_conf, pred_end_conf = float(pred_confs[0]), float(pred_confs[1])

    predicted_label = (postprocess_tensor(s_out.max(1)[0], use_gpu), postprocess_tensor(s_out.max(1)[1], use_gpu))
    label_idx, label_conf = int(predicted_label[1]), float(predicted_label[0])
    return label_idx, label_conf, pred_start_conf, pred_end_conf


def evaluate_sequence(sequence, model: nn.Module, use_gpu, ground_truth: tuple):
    assert is_model_on_gpu(model) == use_gpu, 'Model not on the required device (CPU/GPU).'
    sequence = preprocess_array(sequence, use_gpu)
    model.eval()
    with torch.no_grad():
        c_out, s_out, r_out = model(Variable(sequence))
    if use_gpu:
        r_out = r_out.cpu()
    r_out = r_out.detach().numpy()
    predicted_start, predicted_end = np.argmax(r_out[:, :, 0]), np.argmax(r_out[:, :, 1])
    timeline = ground_truth[0]
    start = int(timeline[0].long().numpy())
    end = int(timeline[-1].long().numpy())
    sl, el = get_localization_scores(predicted_start, predicted_end, start, end)
    label = ground_truth[1]
    action_len = len(timeline)
    predicted_classes = s_out.max(1)[1]
    accuracy = (predicted_classes == label).sum().item() / action_len
    return predicted_classes, accuracy, sl, el


def evaluate_dataset(model: nn.Module, dataset: SkeletonDataset, use_gpu, regression_sigma):
    assert is_model_on_gpu(model) == use_gpu, 'Model not on the required device (CPU/GPU).'
    tot_acc = tot_sl = tot_el = 0.
    num_classes = dataset.label_size()
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    confusion = meter.ConfusionMeter(num_classes + 1)   # Add one for no action or unknown actions
    for ii, (seq, timeline, label) in enumerate(val_loader, 0):
        seq_len = seq.shape[1]
        label_idx = label.long().numpy().data[0]
        confidence = get_confidence_matrix(seq_len, timeline, label_idx, num_classes,
                                           regression_sigma)
        timeline = timeline.squeeze()
        if use_gpu:
            label = label.cuda(0)
        class_predictions, acc, sl, el = evaluate_sequence(seq, model, use_gpu, (timeline, label))
        tot_acc += acc
        tot_sl += sl
        tot_el += el
        confusion.add(class_predictions, label)
    norm_confusion_mat, f1_score = get_normalized_confusion_matrix_and_f1_score(confusion.conf)
    num_samples = len(val_loader)
    avg_acc, avg_sl, avg_el = tot_acc / num_samples, tot_sl / num_samples, tot_el / num_samples
    return norm_confusion_mat, f1_score, avg_sl, avg_el, avg_acc


def get_normalized_confusion_matrix_and_f1_score(confusion_matrix: np.ndarray):
    num_classes = confusion_matrix.shape[0]
    f1 = np.zeros(num_classes)
    norm_confusion_mat = np.zeros([num_classes, num_classes])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for label_idx in range(num_classes):
            ground_truth_length = np.sum(confusion_matrix[label_idx, :])
            if ground_truth_length == 0:
                continue
            norm_confusion_mat[label_idx, :] = confusion_matrix[label_idx, :] / ground_truth_length
            f1[label_idx] = 2. * confusion_matrix[label_idx, label_idx] / \
                (ground_truth_length + np.sum(confusion_matrix[:, label_idx]))
    return norm_confusion_mat, f1


def get_localization_scores(predicted_start, predicted_end, true_start, true_end):
    """
    exp(-abs(t_pred_start-t_start)/(t_end-t_start))
    exp(-abs(t_pred_end-t_end)/(t_end-t_start))
    :param predicted_start:
    :param predicted_end:
    :param true_start:
    :param true_end:
    :return:
    """
    d_start = abs(predicted_start - true_start)
    d_end = abs(predicted_end - true_end)
    base = math.exp(true_start - true_end)
    return math.pow(base, d_start), math.pow(base, d_end)
