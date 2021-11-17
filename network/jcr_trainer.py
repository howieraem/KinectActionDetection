"""Training loop for VA-JCR."""
import random
import logging
import torch
import torch.nn as nn
import numpy as np
from torch import nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from livelossplot import PlotLosses
from sklearn.metrics import accuracy_score
from dataset.skeleton_abstract import SkeletonDataset
from utils.processing import *
from global_configs import RNN_NAME, ACTIVATION_NAME


# plt.switch_backend('agg')     # Uncomment this if training headlessly


__all__ = ['train_jcr', 'FocalLossDebug']


def train_jcr(jcr_model: nn.Module, dataset: SkeletonDataset, optimizer: Optimizer, scheduler: lr_scheduler,
              hyperparameters: list, total_epoches, log_interval, display_interval, save_interval,
              loss_plotter: PlotLosses, start_epoch: int, device: torch.device, gpu_count: int,
              use_multiprocess: bool, preloaded: bool, gamma: float, accuracy_threshold: float,
              ignore_background_class: bool, *args, **kwargs):
    # Loads hyperparameters
    regression_lambda = hyperparameters[HyperParamType.REGRESS_LAMBDA]
    opt_name = 'SGD' if hyperparameters[HyperParamType.USE_SGD] else 'Adam'
    dropouts = hyperparameters[HyperParamType.DROPOUTS]
    use_view_adaptive = hyperparameters[HyperParamType.ENABLE_VA]
    enable_augment = hyperparameters[HyperParamType.ENABLE_AUGMENT]
    try:
        rnn_type = hyperparameters[HyperParamType.RNN_TYPE]
        activation_type = hyperparameters[HyperParamType.ACTIVATION_TYPE]
        use_focal_loss = hyperparameters[HyperParamType.USE_FOCAL_LOSS]
        truncated_length = hyperparameters[HyperParamType.TRUNCATED_LENGTH]
    except IndexError:  # previously trained model
        rnn_type = RNNType.LSTM if hyperparameters[HyperParamType.USE_LSTM] else RNNType.GRU
        activation_type = ActivationType.SELU
        use_focal_loss = True
        truncated_length = 5000     # don't do truncated BPTT
    try:
        use_layer_norm = hyperparameters[HyperParamType.USE_LAYER_NORM]
    except IndexError:
        use_layer_norm = True if rnn_type == RNNType.SRU else False

    # Defines the filename (prefix) for logging and model saving
    filename = ('./trained/'
                + str(dataset)
                + '_lambda' + str(regression_lambda)
                + '_' + RNN_NAME[rnn_type]
                + '_' + ACTIVATION_NAME[activation_type]
                + '_' + opt_name + str(hyperparameters[HyperParamType.INIT_LR])
                + '_dp' + str(int(dropouts[0] * 10)) + str(int(dropouts[1] * 10)) + str(int(dropouts[2] * 10))
                + '_decayFact' + str(hyperparameters[HyperParamType.LR_DECAY])
                + '_decayPati' + str(hyperparameters[HyperParamType.LR_DECAY_PATIENCE])
                + '_tl' + str(truncated_length)
                )
    if use_view_adaptive:
        filename += '_VA'
    if use_layer_norm:
        filename += '_LN'
    if use_focal_loss:
        filename += '_FL'
    if enable_augment:
        filename += '_aug'
    for arg in args:
        filename += '_' + str(arg)
    if kwargs is not None:
        for key, value in kwargs.items():
            filename += '_' + str(key) + '.' + str(value)
    class_weights = torch.ones(dataset.label_size).to(device)
    ignore_idx = (dataset.label_size - 1) if ignore_background_class else -100
    if use_focal_loss and gamma > 0:
        classification_criterion = FocalLossDebug(weight=class_weights, gamma=gamma, ignore_index=ignore_idx)
    else:
        classification_criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_idx)
        gamma = 0
    filename += '_gamma%.1f' % gamma

    # Initializes logger
    logger = logging.getLogger(filename)
    logger.setLevel('DEBUG')
    file_log_handler = logging.FileHandler(filename + '.log', mode='w+')
    logger.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)

    # Moves model to training device and initializes dataset loader
    jcr_model.to(torch.device('cpu'))
    if gpu_count > 1:
        jcr_model = nn.DataParallel(jcr_model)
    jcr_model.to(device)
    train_data_loader = DataLoader(dataset.training_set,
                                   batch_size=1,    # hyperparameters[HyperParamType.BATCH_SIZE],
                                   shuffle=True,
                                   # collate_fn=unprocessed_collate,
                                   num_workers=2 if use_multiprocess else 0,)
    test_data_loader = DataLoader(dataset.testing_set,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=2 if use_multiprocess else 0,)
    avg_loss = avg_accuracy = avg_loss_val = avg_accuracy_val = np.nan
    regression_criterion = torch.nn.MSELoss(reduction='none')
    classification_criterion.to(device)
    regression_criterion.to(device)

    logger.info('Model fitting starts!')
    try:
        for epoch in range(total_epoches):
            epoch_loss = epoch_correct = epoch_loss_val = epoch_correct_val = 0.
            # Training stage
            if len(train_data_loader) != 0.:
                jcr_model.train()
                for batch_idx, (data, target, confidence) in enumerate(train_data_loader, 0):
                    if enable_augment:
                        random.seed()
                        '''
                        step = random.randint(1, 4)
                        length = data.shape[1]
                        if step >= length:
                            step = length - 1
                        downsampled_idx = range(0, length, step)
                        data = data[:, downsampled_idx]
                        target = target[:, downsampled_idx]
                        confidence = confidence[:, downsampled_idx]
                        '''
                        data = affine_transform_sequence(data,
                                                         alpha=random.gauss(0, 0.0001),
                                                         beta=random.gauss(0, 0.001))
                    if not preloaded:
                        data, target, confidence = Variable(data.to(device)), \
                            Variable(target.to(device)), Variable(confidence.to(device))
                    if enable_augment:
                        data *= Variable(torch.randn_like(data) * 0.0005 + 1)
                    batch_correct = 0.
                    # Slices original sequence into chunks of the truncated length k
                    data_parts = torch.split(data, truncated_length, dim=1)
                    target_parts = torch.split(target, truncated_length, dim=1)
                    confidence_parts = torch.split(confidence, truncated_length, dim=1)
                    if target_parts[-1].shape[1] < 16:   # merge the last subsequence that is too short
                        data_parts, target_parts, confidence_parts = \
                            list(data_parts), list(target_parts), list(confidence_parts)
                        data_parts[-2] = torch.cat(data_parts[-2:], dim=1)
                        target_parts[-2] = torch.cat(target_parts[-2:], dim=1)
                        confidence_parts[-2] = torch.cat(confidence_parts[-2:], dim=1)
                        del data_parts[-1], target_parts[-1], confidence_parts[-1]
                    # Trains on each chunk, i.e. gradient not involving previous chunks
                    for (data_part, target_part, confidence_part) in \
                            zip(data_parts, target_parts, confidence_parts):
                        optimizer.zero_grad()
                        classification_out, softmax_out, regression_out = jcr_model(data_part)
                        pred_idx = torch.max(softmax_out, -1)[1]
                        y_true = target_part.cpu().numpy().flatten()
                        y_pred = pred_idx.cpu().numpy().flatten()
                        current_correct = accuracy_score(y_true, y_pred)
                        batch_correct += current_correct / len(data_parts)
                        classification_loss = classification_criterion(classification_out, target_part.squeeze(0))
                        raw_regression_loss = regression_criterion(regression_out, confidence_part.squeeze(0))
                        regression_loss = (raw_regression_loss[..., 0] + raw_regression_loss[..., 1]).mean()
                        loss = (classification_loss + regression_lambda * regression_loss)
                        loss.backward()     # TBPTT with k1 = k2 = truncated_length
                        # nn.utils.clip_grad_norm_(jcr_model.parameters(), 1)   # Uncomment to clip gradient
                        optimizer.step()
                        current_loss = loss.item()
                        epoch_loss += current_loss / len(data_parts)
                    epoch_correct += batch_correct
                    if batch_idx % log_interval == 0:
                        logger.info('Training Epoch {} Batch {}\t\t\tLoss: {:.3f}\t\t\tClassification Accuracy: {:.3f}%'
                                    ''.format(epoch + 1 + start_epoch,
                                              batch_idx + 1,
                                              epoch_loss / (batch_idx + 1),
                                              100. * epoch_correct / (batch_idx + 1)))
                avg_loss = epoch_loss / len(train_data_loader)      # divided by number of batches
                avg_accuracy = epoch_correct / len(train_data_loader.dataset)   # divided by total number of data

                # Adjust learning rate
                scheduler.step(avg_loss)

            # Testing stage
            if len(test_data_loader) != 0.:
                jcr_model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target, confidence) in enumerate(test_data_loader, 0):
                        if enable_augment:
                            random.seed()
                            '''
                            step = random.randint(1, 4)
                            length = data.shape[1]
                            if step >= length:
                                step = length - 1
                            downsampled_idx = range(0, length, step)
                            data = data[:, downsampled_idx]
                            target = target[:, downsampled_idx]
                            confidence = confidence[:, downsampled_idx]
                            '''
                            data = affine_transform_sequence(data,
                                                             alpha=random.gauss(0, 0.0001),
                                                             beta=random.gauss(0, 0.001))
                        if not preloaded:
                            data, target, confidence = Variable(data.to(device)), \
                                Variable(target.to(device)), Variable(confidence.to(device))
                        if enable_augment:
                            data *= Variable(torch.randn_like(data) * 0.0005 + 1)
                        classification_out, softmax_out, regression_out = jcr_model(data)
                        classification_loss = classification_criterion(classification_out, target.squeeze(0))
                        raw_regression_loss = regression_criterion(regression_out, confidence.squeeze(0))
                        regression_loss = (raw_regression_loss[..., 0] + raw_regression_loss[..., 1]).mean()
                        loss = classification_loss + regression_lambda * regression_loss
                        current_loss = loss.item()
                        pred_idx = torch.max(softmax_out, -1)[1]    # top-1
                        y_true = target.cpu().numpy().flatten()
                        y_pred = pred_idx.cpu().numpy().flatten()
                        current_correct = accuracy_score(y_true, y_pred)
                        epoch_loss_val += current_loss
                        epoch_correct_val += current_correct
                avg_loss_val = epoch_loss_val / len(test_data_loader)   # divided by number of batches
                avg_accuracy_val = epoch_correct_val / len(test_data_loader.dataset)
                logger.info('Testing Epoch {}\t\t\tLoss: {:.3f}\t\t\tClassification Accuracy: {:.3f}%'
                            ''.format(epoch + 1 + start_epoch,
                                      avg_loss_val,
                                      100. * avg_accuracy_val))

            # Records intermediate training results
            loss_plotter.update({
                'log loss': avg_loss,
                'val_log loss': avg_loss_val,
                'accuracy': avg_accuracy,
                'val_accuracy': avg_accuracy_val
            })
            if (epoch + 1 + start_epoch) % display_interval == 0.:
                loss_plotter.draw()     # Plots the loss and accuracy so far
            ok_to_save = (avg_accuracy_val > accuracy_threshold)
            if (epoch + 1 + start_epoch) % save_interval == 0. or ok_to_save:
                model_filename = filename + '_Epoch%d.tar' % (epoch + 1 + start_epoch)
                save_checkpoint(epoch + start_epoch, jcr_model, optimizer, scheduler, loss_plotter,
                                hyperparameters, model_filename)
                print('Intermediate step states saved.')
    except KeyboardInterrupt:
        # Warning: trying to save states here does NOT always work
        print('Training interrupted.')


def save_checkpoint(epoch: int,
                    model: nn.Module,
                    optimizer: Optimizer,
                    scheduler: lr_scheduler,
                    loss_plotter: PlotLosses,
                    hyperparameters: list,
                    filename: str):
    """Saves model parameters, optimizer parameters, training history, and hyperparameters."""
    state = {'epoch': epoch, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'loss_plotter': loss_plotter, 'hyperparameters': hyperparameters}
    torch.save(state, filename)


class FocalLossDebug(nn.Module):
    """
    Implements focal loss proposed in the paper:
        T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in
        Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980-2988.
    """
    def __init__(self, alpha=None, gamma: float = 2, ignore_index: int = -100, reduction='mean', weight=None):
        super(FocalLossDebug, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self._cee = nn.CrossEntropyLoss(weight=weight,
                                        ignore_index=ignore_index,
                                        reduction=reduction)

    def forward(self, logit, target):
        logpt = -1 * self._cee(logit, target)
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss
