"""
The evaluation module for VA-JCR/VA-JCM models. Only works in Python >= 3.5.

Some code is forked from https://github.com/ECHO960/PKU-MMD/blob/master/evaluate.py related to the paper:
    C. Liu, Y. Hu, Y. Li, S. Song, and J. Liu, "PKU-MMD: A large scale benchmark for continuous multi-modal human
    action understanding," arXiv preprint arXiv:1703.07475, 2017.
"""
import math
import os
import logging
import pickle
import torch
import torch.nn as nn
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pylab as plt
from scipy.signal import find_peaks
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet import meter
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from .misc import write_sequence_labels_to_file, ProgressBar
from .processing import is_model_on_gpu, get_data_at_observation_levels
from global_configs import DatasetProtocol, RNN_NAME, ACTIVATION_NAME, HyperParamType
from utils.misc import load_checkpoint_jcm, load_checkpoint
from dataset.skeleton_abstract import *
from dataset.skeleton_multitask import *


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
np.warnings.filterwarnings('ignore')
plt.switch_backend('agg')


class UntrimmedDatasetEvaluator(object):
    __slots__ = ['dataset', 'dataset_name', 'num_classes', 'input_dim', 'device', 'model',
                 'epoch', 'output_path', 'logger', 'pred_folder', 'true_folder']

    """
    In prediction and true folders:
        - each file contains results for one video.
        - several lines in one file and each line contains: label, start_frame, end_frame, confidence
    """
    def __init__(self, dataset_path: str,
                 device: torch.device):
        self.dataset = deserialize_dataset(dataset_path, False)
        self.dataset_name = str(self.dataset)
        protocol = self.dataset.train_test_protocol
        if protocol is not None:
            if protocol == DatasetProtocol.CROSS_SUBJECT:
                protocol_name = 'crossSubject'
            elif protocol == DatasetProtocol.CROSS_VIEW:
                protocol_name = 'crossView'
            else:
                raise NotImplementedError
            self.dataset_name += '_' + protocol_name
        self.num_classes = self.dataset.label_size
        self.input_dim = self.dataset.get_joint_number() * 3
        if self.dataset.has_interaction:
            self.input_dim *= 2
        self.device = device
        self.model = None
        self.epoch = 0
        self.output_path = None
        self.true_folder = None
        self.pred_folder = None
        self.logger = None

    def set_model(self, model_path: str, output_path: str):
        import gc
        self.model, _, _, self.epoch, loss_plotter, hyperparams = load_checkpoint(model_path,
                                                                                  num_classes=self.num_classes,
                                                                                  input_dim=self.input_dim,
                                                                                  device=self.device,
                                                                                  deprecated=False)
        output_path.replace('\\', '/')
        model_path.replace('\\', '/')
        if not output_path.endswith('/'):
            output_path += '/'
        model_filename = model_path.split('/')[-1]
        try:
            loc_gamma = model_filename.index('gamma')
            gamma = float(model_filename[loc_gamma+5:loc_gamma+8])
        except ValueError:
            if 'FL' in model_filename:
                gamma = 1.0
            else:
                gamma = 0
        subdir_name = 'scores_%s_%s_%d_va%s_ln%s_fl%s_lbd%s_tl%d_gamma%.1f' % (
            RNN_NAME[hyperparams[HyperParamType.RNN_TYPE]],
            ACTIVATION_NAME[hyperparams[HyperParamType.ACTIVATION_TYPE]],
            self.epoch,
            hyperparams[HyperParamType.ENABLE_VA],
            hyperparams[HyperParamType.USE_LAYER_NORM],
            hyperparams[HyperParamType.USE_FOCAL_LOSS],
            str(hyperparams[HyperParamType.REGRESS_LAMBDA]),
            hyperparams[HyperParamType.TRUNCATED_LENGTH],
            gamma
        )
        dropouts = hyperparams[HyperParamType.DROPOUTS]
        subdir_name += '_dp%d%d%d' % (int(dropouts[0] * 10), int(dropouts[1] * 10), int(dropouts[2] * 10))
        protocol = self.dataset.train_test_protocol
        if protocol == DatasetProtocol.CROSS_SUBJECT:
            subdir_name += '_cs'
        elif protocol == DatasetProtocol.CROSS_VIEW:
            subdir_name += '_cv'
        self.output_path = output_path + subdir_name + '/'
        os.makedirs(self.output_path, exist_ok=True)
        loss_plotter.fig_path = self.output_path + '%s_%d_train_test_curves.png' % (self.dataset_name, self.epoch)
        loss_plotter.draw()
        self.logger = logging.getLogger(self.dataset_name + '_%d' % self.epoch)
        self.logger.setLevel('DEBUG')
        file_log_handler = logging.FileHandler(self.output_path + '%s_%d_eval_logfile'
                                                                  '.log' % (self.dataset_name, self.epoch),
                                               mode='w+')
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = logging.StreamHandler()
        self.logger.addHandler(stderr_log_handler)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)
        null_class = self.num_classes - 1
        self.pred_folder = self.output_path + 'pred_files/'
        self.true_folder = self.output_path + 'true_files/'
        os.makedirs(self.pred_folder, exist_ok=True)
        os.makedirs(self.true_folder, exist_ok=True)
        for idx, (_, label, _) in enumerate(self.dataset.testing_set):
            write_sequence_labels_to_file(label.numpy(),
                                          self.true_folder + '%s_%03d.txt' % (str(self.dataset), idx), null_class)
        self.logger.info('Training logs: %s' % loss_plotter.logs)
        gc.collect()

    def run_evaluation(self):
        assert self.model is not None, 'Model is not initialized before running the evaluation.'
        self._evaluate_basic()
        self._evaluate_advanced()

    def _evaluate_basic(self):
        labels = list(self.dataset.get_labels())
        confusion_mat, raw_f1, sl, el, acc, forecast_prs = \
            evaluate_untrimmed_dataset(self.model, self.dataset, self.device, self.pred_folder)
        filename_prefix = self.output_path + '%s_%d_' % (str(self.dataset), self.epoch)
        sl, el = sl[:-1], el[:-1]   # omit the blank class
        plot_localization_scores(np.mean(sl, axis=0), filename_prefix + 'sl.png')
        plot_localization_scores(np.mean(el, axis=0), filename_prefix + 'el.png')
        with open(filename_prefix + 'sl_el.bin', 'wb+') as f:
            pickle.dump((sl, el), f)
        raw_f1, sl, el = np.round(raw_f1, decimals=4), np.round(sl, decimals=4), np.round(el, decimals=4)
        self.logger.info('Raw F1 Scores: %s (avg: %.3f)' % (list(raw_f1), round(np.mean(raw_f1).item(), 3)))
        self.logger.info('SL Scores at 20%% Threshold: %s (avg: %.3f)' % (list(sl[:, 1]),
                                                                          round(np.mean(sl[:, 1]).item(), 3)))
        self.logger.info('EL Scores at 20%% Threshold: %s (avg: %.3f)' % (list(el[:, 1]),
                                                                          round(np.mean(el[:, 1]).item(), 3)))
        self.logger.info('Average Frame-level Accuracy: %.3f' % acc)
        confusion_mat = np.round(confusion_mat, decimals=2)
        df_cm = pd.DataFrame(confusion_mat, index=[label for label in labels],
                             columns=[label for label in labels])
        width = round(self.num_classes * 1.1)
        height = round(self.num_classes * 0.9)
        cm_filename_without_extension = filename_prefix + 'confusion_mat'
        plot_confusion_mat(df_cm, fig_width=width, fig_height=height, annot_font_size=13, label_font_size=12,
                           label_rotation=30, output_filename=cm_filename_without_extension+'.png')
        with open(cm_filename_without_extension + '.bin', 'wb+') as f:
            pickle.dump(df_cm, f)
        with open(filename_prefix + 'forecast_pr.bin', 'wb+') as f:
            pickle.dump(forecast_prs, f)
        s_pr, e_pr = forecast_prs
        plot_forecast_pr(s_pr[0], s_pr[1], filename_prefix + 'start_forecast_pr.png')
        plot_forecast_pr(e_pr[0], e_pr[1], filename_prefix + 'end_forecast_pr.png')

    def _evaluate_advanced(self):
        v_props = []  # proposal list separated by video
        v_grounds = []  # ground-truth list separated by video

        # ========== find all proposals separated by video========
        for video in os.listdir(self.pred_folder):
            prop = open(self.pred_folder + video, 'r').readlines()
            prop = [prop[x].replace(',', ' ') for x in range(len(prop))]
            prop = [[float(y) for y in prop[x].split()] for x in range(len(prop))]
            ground = open(self.true_folder + video, 'r').readlines()
            ground = [ground[x].replace(',', ' ') for x in range(len(ground))]
            ground = [[float(y) for y in ground[x].split()] for x in range(len(ground))]
            # append video name
            for x in prop:
                x.append(video)
            for x in ground:
                x.append(video)
            v_props.append(prop)
            v_grounds.append(ground)

        # ========== find all proposals separated by action categories ========
        # proposal list separated by class
        a_props = [[] for _ in range(self.num_classes)]
        # ground-truth list separated by class
        a_grounds = [[] for _ in range(self.num_classes)]

        for x in range(len(v_props)):
            for y in range(len(v_props[x])):
                a_props[int(v_props[x][y][0])].append(v_props[x][y])

        for x in range(len(v_grounds)):
            for y in range(len(v_grounds[x])):
                a_grounds[int(v_grounds[x][y][0])].append(v_grounds[x][y])

        # ========== find all proposals ========
        all_props = sum(a_props, [])
        all_grounds = sum(a_grounds, [])

        # ========== calculate protocols ========
        overlap_ratios = [0.1, 0.5]
        for overlap_ratio in overlap_ratios:
            self.logger.info('==============================================================\n'
                             'Advanced evaluations for theta = %.1f: ' % overlap_ratio)
            self.logger.info('F1 = %.3f' % get_f1(all_props, overlap_ratio, all_grounds, self.num_classes))
            self.logger.info('AP = %.3f' % get_ap(all_props, overlap_ratio, all_grounds, self.num_classes))
            self.logger.info('mAP_action = %.3f' % (sum([get_ap(a_props[x], overlap_ratio, a_grounds[x],
                                                                self.num_classes)
                                                        for x in range(self.num_classes - 1)])/(self.num_classes - 1)))
            self.logger.info('mAP_video = %.3f' % (sum([get_ap(v_props[x], overlap_ratio, v_grounds[x],
                                                               self.num_classes)
                                                       for x in range(len(v_props))])/len(v_props)))
            fig_path = self.output_path + '%s_%d_theta%.1f.png' % (str(self.dataset), self.epoch, overlap_ratio)
            plot_detect_pr(all_props, overlap_ratio, all_grounds, fig_path, self.num_classes)
        self.logger.info('2DAP = %.3f' % (sum([get_ap(all_props, (ratio + 1) * 0.05, all_grounds, self.num_classes)
                                              for ratio in range(20)]) / 20))


class MultiAttrDatasetEvaluator(object):
    def __init__(self, dataset_path: str, device: torch.device):
        self.device = device
        self.dataset = deserialize_dataset_multitask(dataset_path, False)
        self.dataset_name = str(self.dataset)
        self.input_dim = self.dataset.get_joint_number() * 3  # 3D coordinates
        if self.dataset.has_interaction:
            self.input_dim *= 2
        self.num_classes_tuple = (self.dataset.label_size, self.dataset.subject_label_size, self.dataset.age_label_size)
        self.model = None
        self.epoch = None
        self.output_path = None
        self.logger = None

    def set_model(self, model_path: str, output_path: str):
        self.model, _, _, self.epoch, loss_plotter, hyperparams, _, test_indices = \
            load_checkpoint_jcm(model_path, num_classes=self.num_classes_tuple, input_dim=self.input_dim,
                                device=self.device)
        assert test_indices == self.dataset.indices_test
        output_path.replace('\\', '/')
        model_path.replace('\\', '/')
        if not output_path.endswith('/'):
            output_path += '/'
        model_filename = model_path.split('/')[-1]
        try:
            loc_gamma = model_filename.index('gamma')
            gamma = float(model_filename[loc_gamma + 5:loc_gamma + 8])
        except ValueError:
            if 'FL' in model_filename:
                gamma = 1.0
            else:
                gamma = 0
        subdir_name = 'scores_%s_%s_%d_va%s_ln%s_fl%s_lbd%s_tl%d_gamma%.1f' % (
            RNN_NAME[hyperparams[HyperParamType.RNN_TYPE]],
            ACTIVATION_NAME[hyperparams[HyperParamType.ACTIVATION_TYPE]],
            self.epoch,
            hyperparams[HyperParamType.ENABLE_VA],
            hyperparams[HyperParamType.USE_LAYER_NORM],
            hyperparams[HyperParamType.USE_FOCAL_LOSS],
            str(hyperparams[HyperParamType.REGRESS_LAMBDA]),
            hyperparams[HyperParamType.TRUNCATED_LENGTH],
            gamma
        )
        dropouts = hyperparams[HyperParamType.DROPOUTS]
        subdir_name += '_dp%d%d%d' % (int(dropouts[0] * 10), int(dropouts[1] * 10), int(dropouts[2] * 10))
        protocol = self.dataset.train_test_protocol
        if protocol == DatasetProtocol.CROSS_SUBJECT:
            subdir_name += '_csb'
        elif protocol == DatasetProtocol.CROSS_SAMPLE:
            subdir_name += '_csa'
        elif protocol == DatasetProtocol.CROSS_AGE:
            subdir_name += '_cag'
        elif protocol == DatasetProtocol.CROSS_GENDER:
            subdir_name += '_cgd'
        self.output_path = output_path + subdir_name + '/'
        os.makedirs(self.output_path, exist_ok=True)
        loss_plotter.fig_path = self.output_path + '%s_%d_train_test_curves.png' % (self.dataset_name, self.epoch)
        loss_plotter.draw()
        self.logger = logging.getLogger(self.dataset_name + '_%d' % self.epoch)
        self.logger.setLevel('DEBUG')
        file_log_handler = logging.FileHandler(self.output_path + '%s_%d_eval_logfile'
                                                                  '.log' % (self.dataset_name, self.epoch),
                                               mode='w+')
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = logging.StreamHandler()
        self.logger.addHandler(stderr_log_handler)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

    def run_evaluation(self):
        assert self.model is not None, 'Model is not initialized before running the evaluation.'
        enable_multitask = (self.dataset.train_test_protocol == DatasetProtocol.CROSS_SAMPLE)
        action_labels = self.dataset.get_labels()
        subject_labels = range(self.dataset.subject_label_size)     # TODO
        age_labels = range(self.dataset.age_label_size)             # TODO
        confusion_mats, f1s, accs, acc_at_observ_lvls = \
            evaluate_multiattr_dataset(self.model, self.dataset, self.device)
        filename_prefix = self.output_path + '%s_%d_' % (str(self.dataset), self.epoch)
        with open(filename_prefix + 'acc_at_lvls.bin', 'wb+') as f:
            pickle.dump(acc_at_observ_lvls, f)

        # Ignore classes without any predictions (i.e. missing in test set)
        action_conf, subject_conf, age_conf = confusion_mats
        action_ignore_idx = np.where(~action_conf.any(axis=1))[0]
        subject_ignore_idx = np.where(~subject_conf.any(axis=1))[0]

        # Accuracies and F1s
        action_f1, subject_f1, age_f1 = f1s
        action_acc, subject_acc, age_acc = accs
        action_acc_lvls, subject_acc_lvls, age_acc_lvls = np.round(acc_at_observ_lvls, 4)
        action_f1, subject_f1, age_f1 = np.round(action_f1, 4), np.round(subject_f1, 4), np.round(age_f1, 4)
        action_f1 = np.delete(action_f1, action_ignore_idx, 0)
        self.logger.info('Action Accuracies: %s (avg: %.3f)' % (list(action_acc_lvls), action_acc))
        self.logger.info('Action F1 Scores: %s (avg: %.3f)' % (list(action_f1), round(np.mean(action_f1).item(), 3)))
        if enable_multitask:
            subject_f1 = np.delete(subject_f1, subject_ignore_idx, 0)
            self.logger.info('Subject Accuracies: %s (avg: %.3f)' % (list(subject_acc_lvls), subject_acc))
            self.logger.info('Subject F1 Scores: %s (avg: %.3f)' % (list(subject_f1),
                                                                    round(np.mean(subject_f1).item(), 3)))
            self.logger.info('Age Accuracies: %s (avg: %.3f)' % (list(age_acc_lvls), age_acc))
            self.logger.info('Age F1 Scores: %s (avg: %.3f)' % (list(age_f1),
                                                                round(np.mean(age_f1).item(), 3)))

        # Confusion matrices
        action_conf, subject_conf, age_conf = np.round(action_conf, 2), np.round(subject_conf, 2), np.round(age_conf, 2)
        cm_filename_without_extension = filename_prefix + 'confusion_mat'
        action_labels = np.delete(action_labels, action_ignore_idx, 0)
        action_conf = np.delete(action_conf, action_ignore_idx, 0)
        action_conf = np.delete(action_conf, action_ignore_idx, 1)
        act_df = pd.DataFrame(action_conf, index=[label for label in action_labels],
                              columns=[label for label in action_labels])
        plot_confusion_mat(act_df, fig_width=round(self.num_classes_tuple[0] * 1.1),
                           fig_height=round(self.num_classes_tuple[0] * 0.9),
                           annot_font_size=13, label_font_size=12, label_rotation=30,
                           output_filename=cm_filename_without_extension + '_action.png')
        with open(cm_filename_without_extension + '_action.bin', 'wb+') as f:
            pickle.dump(act_df, f)
        if enable_multitask:
            subject_labels = np.delete(subject_labels, subject_ignore_idx, 0)
            subject_conf = np.delete(subject_conf, subject_ignore_idx, 0)
            subject_conf = np.delete(subject_conf, subject_ignore_idx, 1)
            sub_df = pd.DataFrame(subject_conf, index=[label for label in subject_labels],
                                  columns=[label for label in subject_labels])
            plot_confusion_mat(sub_df, fig_width=round(self.num_classes_tuple[1] * 1.1),
                               fig_height=round(self.num_classes_tuple[1] * 0.9),
                               annot_font_size=13, label_font_size=12, label_rotation=0,
                               output_filename=cm_filename_without_extension + '_subject.png')
            with open(cm_filename_without_extension + '_subject.bin', 'wb+') as f:
                pickle.dump(sub_df, f)
            age_df = pd.DataFrame(age_conf, index=[label for label in age_labels],
                                  columns=[label for label in age_labels])
            plot_confusion_mat(age_df, fig_width=round(self.num_classes_tuple[2] * 1.1),
                               fig_height=round(self.num_classes_tuple[2] * 0.9),
                               annot_font_size=13, label_font_size=12, label_rotation=0,
                               output_filename=cm_filename_without_extension + '_age.png')
            with open(cm_filename_without_extension + '_age.bin', 'wb+') as f:
                pickle.dump(age_df, f)


def calc_pr(positive, proposal, ground):
    """
    Calculate precision and recall
    :param positive: number of positive proposals
    :param proposal: number of all proposals
    :param ground: number of ground truths
    :return:
    """
    if not proposal or not ground:
        return 0, 0
    return positive / proposal, positive / ground


def match(lst, ratio, ground, num_classes):
    """
    Match proposal and ground truth

    correspond_map: record matching ground truth for each proposal
    count_map: record how many proposals is each ground truth matched by
    index_map: index_list of each video for ground truth

    :param lst: list of proposals(label, start, end, confidence, video_name)
    :param ratio: overlap ratio
    :param ground: list of ground truth(label, start, end, confidence, video_name)
    :param num_classes:
    """

    def overlap(prop, gt):
        l_p, s_p, e_p, c_p, v_p = prop
        l_g, s_g, e_g, c_g, v_g = gt
        if v_p != v_g or int(l_p) != int(l_g):
            return 0
        denominator = max(e_p, e_g) - min(s_p, s_g)
        if not denominator:     # avoid division by zero, i.e. one-frame prediction
            return 0
        return (min(e_p, e_g) - max(s_p, s_g)) / denominator

    corres_map = [-1] * len(lst)
    count_map = [0] * len(ground)
    # generate index_map to speed up
    index_map = [[] for _ in range(num_classes)]
    for x in range(len(ground)):
        index_map[int(ground[x][0])].append(x)

    for x in range(len(lst)):
        for y in index_map[int(lst[x][0])]:
            if overlap(lst[x], ground[y]) < ratio:
                continue
            if overlap(lst[x], ground[y]) < overlap(lst[x], ground[corres_map[x]]):
                continue
            corres_map[x] = y
        if corres_map[x] != -1:
            count_map[corres_map[x]] += 1
    positive = sum([(x > 0) for x in count_map])
    return corres_map, count_map, positive


def plot_detect_pr(lst, ratio, ground, output_filename, num_classes):
    """
    plot precision-recall figure of given proposal
    :param lst: list of proposals(label, start, end, confidence, video_name)
    :param ratio: overlap ratio
    :param ground: list of ground truth(label, start, end, confidence, video_name)
    :param output_filename:
    :param num_classes
    """
    lst.sort(key=lambda y: y[3])  # sorted by confidence
    correspond_map, count_map, positive = match(lst, ratio, ground, num_classes)
    number_proposal = len(lst)
    number_ground = len(ground)
    old_precision, old_recall = calc_pr(positive, number_proposal, number_ground)

    recalls = [old_recall]
    precisions = [old_precision]
    for x in range(len(lst)):
        number_proposal -= 1
        if correspond_map[x] == -1:
            continue
        count_map[correspond_map[x]] -= 1
        if count_map[correspond_map[x]] == 0:
            positive -= 1

        precision, recall = calc_pr(positive, number_proposal, number_ground)
        if precision > old_precision:
            old_precision = precision
        recalls.append(recall)
        precisions.append(old_precision)
        # old_recall = recall
    bin_filename_prefix = output_filename.replace('.png', '')
    with open(bin_filename_prefix + '_recall_precision.bin', 'wb+') as f:
        pickle.dump((recalls, precisions), f)
    fig = plt.figure()
    plt.axis([0, 1, 0, 1])
    plt.plot(recalls, precisions, 'r')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.title('Action Detection Precision-Recall Curve for theta = %.1f' % ratio)
    # plt.show()
    if not output_filename.endswith('.png'):
        output_filename += '.png'
    fig.savefig(output_filename)


def get_f1(lst, ratio, ground, num_classes):
    """
    f1-score
    :param lst: list of proposals(label, start, end, confidence, video_name)
    :param ratio: overlap ratio
    :param ground: list of ground truth(label, start, end, confidence, video_name)
    :param num_classes:
    """
    correspond_map, count_map, positive = match(lst, ratio, ground, num_classes)
    precision, recall = calc_pr(positive, len(lst), len(ground))
    score = 2 * precision * recall / (precision + recall)
    return score


def get_ap(lst, ratio, ground, num_classes):
    """
    Interpolated Average Precision

    score = sigma(precision(recall) * delta(recall))
    Note that when overlap ratio < 0.5,
    one ground truth will correspond to many proposals
    In that case, only one positive proposal is counted

    :param lst: list of proposals(label, start, end, confidence, video_name)
    :param ratio: overlap ratio
    :param ground: list of ground truth(label, start, end, confidence, video_name)
    :param num_classes:
    """
    lst.sort(key=lambda x: x[3])  # sorted by confidence
    correspond_map, count_map, positive = match(lst, ratio, ground, num_classes)
    score = 0
    number_proposal = len(lst)
    number_ground = len(ground)
    old_precision, old_recall = calc_pr(positive, number_proposal, number_ground)

    for x in range(len(lst)):
        number_proposal -= 1
        if correspond_map[x] == -1:
            continue
        count_map[correspond_map[x]] -= 1
        if count_map[correspond_map[x]] == 0:
            positive -= 1

        precision, recall = calc_pr(positive, number_proposal, number_ground)
        if precision > old_precision:
            old_precision = precision
        score += old_precision * (old_recall - recall)
        old_recall = recall
    return score


def plot_confusion_mat(dataframe: pd.DataFrame, fig_width: int, fig_height: int, annot_font_size: int,
                       label_font_size: int, label_rotation: int, output_filename: str):
    plt.figure(figsize=(fig_width, fig_height))
    heatmap = sn.heatmap(dataframe, annot=True, vmin=0, vmax=1, square=True, annot_kws={'size': annot_font_size},
                         xticklabels=True, yticklabels=True)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=label_rotation, ha='right', fontsize=label_font_size)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=label_rotation, ha='right', fontsize=label_font_size)
    plt.xlabel('Predicted Label', fontsize=label_font_size+1)
    plt.ylabel('True Label', fontsize=label_font_size+1)
    plt.title('Confusion Matrix')
    plt.gcf().subplots_adjust(left=0.3, bottom=0.15)
    cmf = heatmap.get_figure()
    # cmf.show()
    cmf.savefig(output_filename)


def evaluate_multiattr_sequence(sequence: torch.Tensor, model: nn.Module):
    assert sequence.dim() == 2
    seq_len = len(sequence)
    predicted_action_classes = torch.zeros(seq_len, dtype=torch.int64)
    predicted_subject_classes = torch.zeros(seq_len, dtype=torch.int64)
    predicted_age_classes = torch.zeros(seq_len, dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for idx, frame in enumerate(Variable(sequence)):
            _, (action_out, subject_out, age_out) = model(frame.unsqueeze(0).unsqueeze(0))
            predicted_action_classes[idx] = torch.max(action_out, -1)[1].item()
            predicted_subject_classes[idx] = torch.max(subject_out, -1)[1].item()
            predicted_age_classes[idx] = torch.max(age_out, -1)[1].item()
    return predicted_action_classes, predicted_subject_classes, predicted_age_classes


def evaluate_multiattr_dataset(model: nn.Module, dataset: SkeletonDatasetMultiTask, device):
    model.to(device)
    num_action_classes = dataset.label_size
    num_subject_classes = dataset.subject_label_size
    num_age_classes = dataset.age_label_size
    acc_at_observ_lvls = np.zeros((3, 11))
    all_pred_action = torch.zeros(0, dtype=torch.int64)
    all_true_action = torch.zeros(0, dtype=torch.int64)
    all_pred_subject = torch.zeros(0, dtype=torch.int64)
    all_true_subject = torch.zeros(0, dtype=torch.int64)
    all_pred_age = torch.zeros(0, dtype=torch.int64)
    all_true_age = torch.zeros(0, dtype=torch.int64)
    action_confusion = meter.ConfusionMeter(num_action_classes)
    subject_confusion = meter.ConfusionMeter(num_subject_classes)
    age_confusion = meter.ConfusionMeter(num_age_classes)
    num_test_samples = len(dataset.testing_set)
    pg = ProgressBar(80, num_test_samples)
    for idx, (seq, action_label, subject_label, age_label) in enumerate(dataset.testing_set, 0):
        pred_act, pred_sub, pred_age = evaluate_multiattr_sequence(seq.to(device), model)
        sampled_pred = get_data_at_observation_levels(np.vstack((pred_act, pred_sub, pred_age)).transpose())
        sampled_true = get_data_at_observation_levels(np.vstack((action_label, subject_label, age_label)).transpose())
        sampled_pred, sampled_true = sampled_pred.transpose(), sampled_true.transpose()
        for category_idx, category_true in enumerate(sampled_true):
            acc_at_observ_lvls[category_idx] += (category_true == sampled_pred[category_idx])
        all_true_action = torch.cat((all_true_action, action_label))
        all_pred_action = torch.cat((all_pred_action, pred_act))
        all_true_subject = torch.cat((all_true_subject, subject_label))
        all_pred_subject = torch.cat((all_pred_subject, pred_sub))
        all_true_age = torch.cat((all_true_age, age_label))
        all_pred_age = torch.cat((all_pred_age, pred_age))
        pg.update(idx + 1)
    action_confusion.add(all_pred_action, all_true_action)
    subject_confusion.add(all_pred_subject, all_true_subject)
    age_confusion.add(all_pred_age, all_true_age)
    action_confusion, _ = get_normalized_confusion_matrix_and_f1_score(action_confusion.conf)
    subject_confusion, _ = get_normalized_confusion_matrix_and_f1_score(subject_confusion.conf)
    age_confusion, _ = get_normalized_confusion_matrix_and_f1_score(age_confusion.conf)
    avg_action_f1 = f1_score(all_true_action.numpy(), all_pred_action.numpy(), average=None)
    avg_subject_f1 = f1_score(all_true_subject.numpy(), all_pred_subject.numpy(), average=None)
    avg_age_f1 = f1_score(all_true_age.numpy(), all_pred_age.numpy(), average=None)
    acc_at_observ_lvls /= num_test_samples
    avg_action_acc = accuracy_score(all_true_action.numpy(), all_pred_action.numpy())
    avg_subject_acc = accuracy_score(all_true_subject.numpy(), all_pred_subject.numpy())
    avg_age_acc = accuracy_score(all_true_age.numpy(), all_pred_age.numpy())
    return ((action_confusion, subject_confusion, age_confusion),
            (avg_action_f1, avg_subject_f1, avg_age_f1),
            (avg_action_acc, avg_subject_acc, avg_age_acc),
            acc_at_observ_lvls)


def plot_localization_scores(l_scores: np.ndarray, out_filename: str):
    """Plots and saves the figure of R-based localization scores against thresholds."""
    assert len(l_scores) == 10
    fig = plt.figure()
    x = np.arange(0, 1.0, 0.1)
    plt.plot(x, l_scores, 'g', linewidth=2, markersize=12)
    plt.grid(True)
    axes = plt.gca()
    axes.set_xlim([0, 0.9])
    axes.set_ylim([0, 1])
    plt.xlabel('Threshold')
    plt.ylabel('Localization Score')
    plt.title('Localization Scores at Different Thresholds')
    if not out_filename.endswith('.png'):
        out_filename += '.png'
    fig.savefig(out_filename)


def evaluate_untrimmed_sequence(sequence: torch.Tensor, model: nn.Module, true_confidence: np.ndarray):
    """Evaluates a untrimmed sequence, i.e. a sequence that contains multiple classes."""
    assert sequence.dim() == 2
    seq_len = len(sequence)
    predicted_classes = []
    class_probs = []
    tot_r_out = torch.Tensor()
    model.eval()
    # Aggregates result at each time step
    with torch.no_grad():
        for idx, frame in enumerate(Variable(sequence)):
            _, s_out, r_out = model(frame.unsqueeze(0).unsqueeze(0))
            class_prob, pred_idx = torch.max(s_out, -1)
            predicted_classes += [pred_idx.item()]
            class_probs += [class_prob.item()]
            tot_r_out = torch.cat((tot_r_out, r_out.cpu()))
    tot_r_out = torch.clamp(tot_r_out, min=0, max=1).detach().numpy()
    num_targets = true_confidence.shape[1]
    all_sl = np.zeros((num_targets, 10))
    all_el = np.zeros((num_targets, 10))
    is_action_in_seq = np.zeros(num_targets)
    # Evaluates R-based SL/EL scores for each class except the background class ~
    for i in range(num_targets):
        if np.max(true_confidence[:, i]) > 0:
            true_starts = find_peaks(np.pad(true_confidence[:, i, 0], (1, 1), 'constant'),
                                     height=1, distance=8)[0] - 1
            predicted_starts = -np.ones((10, len(true_starts)), dtype=np.int)
            for idx, true_start in enumerate(true_starts):
                begin_scan_idx = true_start - 30
                if begin_scan_idx < 0:
                    begin_scan_idx = 0
                end_scan_idx = true_start + 30
                if end_scan_idx >= seq_len:
                    end_scan_idx = seq_len - 1
                predicted_start_in_window = np.argmax(tot_r_out[begin_scan_idx:end_scan_idx + 1, i, 0])
                actual_start = predicted_start_in_window + begin_scan_idx
                for th in range(10):
                    confidence_thres = th / 10
                    if tot_r_out[actual_start, i, 0] > confidence_thres:
                        predicted_starts[th, idx] = actual_start
            true_ends = find_peaks(np.pad(true_confidence[:, i, 1], (1, 1), 'constant'),
                                   height=1, distance=8)[0] - 1
            predicted_ends = -np.ones((10, len(true_starts)), dtype=np.int)
            for idx, true_end in enumerate(true_ends):
                begin_scan_idx = true_end - 30
                if begin_scan_idx < 0:
                    begin_scan_idx = 0
                end_scan_idx = true_end + 30
                if end_scan_idx >= seq_len:
                    end_scan_idx = seq_len - 1
                predicted_end_in_window = np.argmax(tot_r_out[begin_scan_idx:end_scan_idx + 1, i, 1])
                actual_end = predicted_end_in_window + begin_scan_idx
                for th in range(10):
                    confidence_thres = th / 10
                    if tot_r_out[actual_end, i, 1] > confidence_thres:
                        predicted_ends[th, idx] = actual_end
            sl, el = get_localization_score_arrays(predicted_starts, predicted_ends, true_starts, true_ends)
            all_sl[i] = sl
            all_el[i] = el
            is_action_in_seq[i] += 1   # increment if the sequence contains such action
    return predicted_classes, all_sl, all_el, class_probs, is_action_in_seq, tot_r_out


def evaluate_untrimmed_dataset(model: nn.Module, dataset: SkeletonDataset, device, output_path):
    """Evaluates untrimmed datasets by averaging sequence-wise results."""
    assert dataset.preloaded is False
    model.to(device)
    num_classes = dataset.label_size
    null_class = num_classes - 1
    confusion = meter.ConfusionMeter(num_classes)
    predictions = labels = np.array([])
    tot_sl = np.zeros((num_classes, 10))
    tot_el = np.zeros((num_classes, 10))
    tot_train_regress_confidence = np.array([]).reshape((0, num_classes, 2))
    tot_test_regress_confidence = np.array([]).reshape((0, num_classes, 2))
    action_in_seq_counts = np.zeros(num_classes)
    pg = ProgressBar(80, len(dataset.testing_set))
    for idx, (seq, label, confidence) in enumerate(dataset.testing_set, 0):
        confidence = confidence.numpy()
        y_pred, sl, el, class_probs, action_in_seq, r_out = \
            evaluate_untrimmed_sequence(seq.to(device), model, confidence)
        tot_train_regress_confidence = np.concatenate((tot_train_regress_confidence, confidence))
        tot_test_regress_confidence = np.concatenate((tot_test_regress_confidence, r_out))
        y_true = label.cpu().numpy().flatten()
        write_sequence_labels_to_file(y_pred, output_path + '%s_%03d.txt' % (str(dataset), idx),
                                      null_class, class_probs)
        predictions = np.append(predictions, y_pred)
        labels = np.append(labels, y_true)
        tot_sl += sl
        tot_el += el
        action_in_seq_counts += action_in_seq
        confusion.add(torch.LongTensor(y_pred), label)
        pg.update(idx + 1)
    norm_confusion_mat, _ = get_normalized_confusion_matrix_and_f1_score(confusion.conf)
    action_in_seq_counts = np.expand_dims(action_in_seq_counts, -1)
    avg_sl, avg_el = tot_sl / action_in_seq_counts, tot_el / action_in_seq_counts
    avg_f1 = f1_score(labels, predictions, average=None)
    avg_acc = accuracy_score(labels, predictions)
    avg_forecast_prs = calc_forecast_prs(tot_train_regress_confidence, tot_test_regress_confidence)
    return norm_confusion_mat, avg_f1, avg_sl, avg_el, avg_acc, avg_forecast_prs


def get_normalized_confusion_matrix_and_f1_score(confusion_matrix: np.ndarray):
    """Returns the normalized confusion matrix and F1 from a raw confusion matrix."""
    num_classes = confusion_matrix.shape[0]
    f1 = np.zeros(num_classes)
    norm_conf_mat = np.zeros([num_classes, num_classes])
    for label_idx in range(num_classes):
        ground_truth_length = np.sum(confusion_matrix[label_idx, :])
        if ground_truth_length == 0:
            continue
        norm_conf_mat[label_idx, :] = confusion_matrix[label_idx, :] / ground_truth_length
        f1[label_idx] = 2. * confusion_matrix[label_idx, label_idx] / \
            (ground_truth_length + np.sum(confusion_matrix[:, label_idx]))
    return norm_conf_mat, f1


def get_localization_score_arrays(predicted_starts: np.ndarray, predicted_ends: np.ndarray,
                                  true_starts: np.ndarray, true_ends: np.ndarray):
    """Returns R-based localization scores at different confidence thresholds."""
    assert predicted_starts.shape[-1] == predicted_ends.shape[-1] == true_starts.shape[-1] == true_ends.shape[-1]
    intervals = true_starts - true_ends
    intervals[intervals >= 0] = -1  # invalid ground truths produced by downsampling
    base = np.exp(1 / intervals)
    predicted_starts[predicted_starts < 0] = 1e6
    predicted_ends[predicted_ends < 0] = -1e6
    avg_sl = np.mean(base ** np.abs(predicted_starts - true_starts), axis=1)
    avg_el = np.mean(base ** np.abs(predicted_ends - true_ends), axis=1)
    return avg_sl, avg_el   # both of shape (10, ) for thresholds 0-0.9


def calc_forecast_prs(true_confidence: np.ndarray, pred_confidence: np.ndarray):
    """Evaluates precision-recall for start/end forecasts."""
    assert true_confidence.shape == pred_confidence.shape
    true_start_confidence, true_end_confidence = true_confidence[:, :-1].transpose((2, 0, 1))
    pred_start_confidence, pred_end_confidence = pred_confidence[:, :-1].transpose((2, 0, 1))
    seq_len, num_actions = true_start_confidence.shape
    true_start_mat = np.zeros((num_actions, seq_len), dtype=np.int)     # one hot
    pred_start_mat = pred_start_confidence.transpose()
    true_end_mat = np.zeros((num_actions, seq_len), dtype=np.int)       # one hot
    pred_end_mat = pred_end_confidence.transpose()
    for i in range(num_actions):
        true_starts = find_peaks(np.pad(true_start_confidence[:, i], (1, 1), 'constant'),
                                 height=1, distance=8)[0] - 1
        for true_start in true_starts:
            begin_scan_idx = true_start - 15
            if begin_scan_idx < 0:
                begin_scan_idx = 0
            end_scan_idx = true_start   # strictly forecast, not causal / lagging
            if end_scan_idx >= seq_len:
                end_scan_idx = seq_len - 1
            true_start_mat[i, begin_scan_idx:end_scan_idx + 1] = 1

        true_ends = find_peaks(np.pad(true_end_confidence[:, i], (1, 1), 'constant'),
                               height=1, distance=8)[0] - 1
        for true_end in true_ends:
            begin_scan_idx = true_end - 15
            if begin_scan_idx < 0:
                begin_scan_idx = 0
            end_scan_idx = true_end    # strictly forecast, not causal / lagging
            if end_scan_idx >= seq_len:
                end_scan_idx = seq_len - 1
            true_end_mat[i, begin_scan_idx:end_scan_idx + 1] = 1
    start_precisions, start_recalls, _ = precision_recall_curve(true_start_mat.ravel(),
                                                                pred_start_mat.ravel())
    end_precisions, end_recalls, _ = precision_recall_curve(true_end_mat.ravel(),
                                                            pred_end_mat.ravel())
    return (start_recalls[1:], start_precisions[1:]), (end_recalls[1:], end_precisions[1:])


def plot_forecast_pr(recalls: np.ndarray, precisions: np.ndarray, output_filename: str):
    """Plots precision-recall curve for start/end forecasts."""
    plt.figure()
    plt.plot(recalls, precisions, 'b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    plt.title('Action Forecast Precision-Recall Curve')
    if not output_filename.endswith('.png'):
        output_filename += '.png'
    plt.savefig(output_filename)
