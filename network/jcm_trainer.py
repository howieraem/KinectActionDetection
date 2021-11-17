"""Training loop for VA-JCM."""
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from livelossplot import PlotLosses
from sklearn.metrics import accuracy_score
from .jcr import JCM
from network.jcr_trainer import FocalLossDebug
from dataset.skeleton_multitask import *
from utils.processing import *
from global_configs import RNN_NAME, ACTIVATION_NAME


__all__ = ['train_jcm']
np.set_printoptions(suppress=True, precision=3, threshold=20)


def l1_term(layer: nn.Linear):
    return torch.norm(layer.weight, p=1)


def train_jcm(jcm_model: JCM, dataset: SkeletonDatasetMultiTask,
              optimizer: Optimizer, scheduler: lr_scheduler, hyperparameters: list,
              total_epoches, log_interval, display_interval, save_interval,
              loss_plotter: PlotLosses, start_epoch: int, device: torch.device, gpu_count: int,
              use_multiprocess: bool, preloaded: bool, attempt_id: int, *args, **kwargs):
    # Initializes logger
    logger = logging.getLogger(str(dataset))
    logger.setLevel('DEBUG')
    os.makedirs('./trained/%d' % attempt_id, exist_ok=True)
    file_log_handler = logging.FileHandler('./trained/%d/train_%s_logfile.log' % (attempt_id, str(dataset)),
                                           mode='w+')
    logger.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)

    # Loads hyperparameters
    enable_augment = hyperparameters[HyperParamType.ENABLE_AUGMENT]
    opt_name = 'SGD' if hyperparameters[HyperParamType.USE_SGD] else 'Adam'
    dropouts = hyperparameters[HyperParamType.DROPOUTS]
    use_view_adaptive = hyperparameters[HyperParamType.ENABLE_VA]
    lambda_weight = hyperparameters[HyperParamType.REGRESS_LAMBDA]
    enable_multitask = int(dataset.train_test_protocol == DatasetProtocol.CROSS_SAMPLE)
    rescale_loss_factor = 1 if enable_multitask else 1 / (0.9*lambda_weight)
    assert 0 <= lambda_weight <= 1
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

    # Loads model to training device and initializes dataset loader
    jcm_model.to(torch.device('cpu'))
    if gpu_count > 1:
        jcm_model = nn.DataParallel(jcm_model)
    jcm_model.to(device)
    train_data_loader = DataLoader(dataset.training_set,
                                   batch_size=1,    # hyperparameters[HyperParamType.BATCH_SIZE],
                                   shuffle=True,
                                   # collate_fn=unprocessed_collate,
                                   num_workers=2 if use_multiprocess else 0,)
    test_data_loader = DataLoader(dataset.testing_set,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=2 if use_multiprocess else 0,)
    avg_loss = avg_loss_val = np.nan
    avg_accuracy1 = avg_accuracy1_val = avg_accuracy2 = avg_accuracy2_val = avg_accuracy3 = avg_accuracy3_val = np.nan
    action_class_weights = torch.ones(dataset.label_size).to(device)
    subject_class_weights = torch.ones(dataset.subject_label_size).to(device)
    age_class_weights = torch.ones(dataset.age_label_size).to(device)
    if use_focal_loss:
        criterion1 = FocalLossDebug(weight=action_class_weights, gamma=1)
        criterion2 = FocalLossDebug(weight=subject_class_weights, gamma=2)
        criterion3 = FocalLossDebug(weight=age_class_weights, gamma=2)
    else:
        criterion1 = nn.CrossEntropyLoss(weight=action_class_weights)
        criterion2 = nn.CrossEntropyLoss(weight=subject_class_weights)
        criterion3 = nn.CrossEntropyLoss(weight=age_class_weights)
    criterion1.to(device)
    criterion2.to(device)
    criterion3.to(device)
    mu = 5e-6
    num_train_batches = len(train_data_loader)
    num_test_samples = len(test_data_loader.dataset)

    try:
        for epoch in range(total_epoches):
            epoch_loss = epoch_loss_val = 0.
            epoch_correct1 = epoch_correct2 = epoch_correct3 = 0.
            # epoch_correct1_val = epoch_correct2_val = epoch_correct3_val = 0.
            # Training stage
            if len(train_data_loader) != 0.:
                jcm_model.train()
                for batch_idx, (data, target1, target2, target3) in enumerate(train_data_loader, 0):
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
                        data, target1, target2, target3 = Variable(data.to(device)), Variable(target1.to(device)), \
                            Variable(target2.to(device)), Variable(target3.to(device))
                    if enable_augment:
                        data *= Variable(torch.randn_like(data) * 0.0005 + 1)
                    data_parts = torch.split(data, truncated_length, dim=1)
                    target1_parts = torch.split(target1, truncated_length, dim=1)
                    target2_parts = torch.split(target2, truncated_length, dim=1)
                    target3_parts = torch.split(target3, truncated_length, dim=1)
                    if target1_parts[-1].shape[1] < 4:   # merge the last subsequence that is too short
                        data_parts, target1_parts, target2_parts, target3_parts = \
                            list(data_parts), list(target1_parts), list(target2_parts), list(target3_parts)
                        data_parts[-2] = torch.cat(data_parts[-2:], dim=1)
                        target1_parts[-2] = torch.cat(target1_parts[-2:], dim=1)
                        target2_parts[-2] = torch.cat(target2_parts[-2:], dim=1)
                        target3_parts[-2] = torch.cat(target3_parts[-2:], dim=1)
                        del data_parts[-1], target1_parts[-1], target2_parts[-1], target3_parts[-1]
                    num_chunks = len(data_parts)
                    for (data_part, target1_part, target2_part, target3_part) in \
                            zip(data_parts, target1_parts, target2_parts, target3_parts):
                        target1_part = target1_part.squeeze(0)
                        target2_part = target2_part.squeeze(0)
                        target3_part = target3_part.squeeze(0)
                        optimizer.zero_grad()
                        logit_outs, softmax_outs = jcm_model(data_part)
                        pred1_idx = torch.max(softmax_outs[0], -1)[1]
                        pred2_idx = torch.max(softmax_outs[1], -1)[1]
                        pred3_idx = torch.max(softmax_outs[2], -1)[1]
                        all_y1_true = target1_part.cpu().numpy().flatten()
                        all_y2_true = target2_part.cpu().numpy().flatten()
                        all_y3_true = target3_part.cpu().numpy().flatten()
                        all_y1_pred = pred1_idx.cpu().numpy().flatten()
                        all_y2_pred = pred2_idx.cpu().numpy().flatten()
                        all_y3_pred = pred3_idx.cpu().numpy().flatten()
                        epoch_correct1 += accuracy_score(all_y1_true, all_y1_pred) / num_chunks
                        epoch_correct2 += accuracy_score(all_y2_true, all_y2_pred) / num_chunks
                        epoch_correct3 += accuracy_score(all_y3_true, all_y3_pred) / num_chunks
                        action_class_loss = (criterion1(logit_outs[0], target1_part) + mu * l1_term(jcm_model.fc1))
                        subject_class_loss = (criterion2(logit_outs[1], target2_part) + mu * l1_term(jcm_model.fc2))
                        age_class_loss = (criterion3(logit_outs[2], target3_part) + mu * l1_term(jcm_model.fc3))
                        loss = (0.9 * lambda_weight * action_class_loss * rescale_loss_factor +
                                0.9 * (1 - lambda_weight) * enable_multitask * subject_class_loss +
                                0.1 * enable_multitask * age_class_loss)
                        loss.backward()     # TBPTT with k1 = k2 = truncated_length
                        # nn.utils.clip_grad_norm_(jcr_model.parameters(), 1)
                        optimizer.step()
                        epoch_loss += loss.item() / num_chunks
                    if batch_idx % log_interval == 0:
                        logger.info('Training Epoch {} Batch {}\t\t\t'
                                    'Loss: {:.3f}\t\t\t'
                                    'Action Accuracy: {:.3f}%\t\t\t'
                                    'Subject Accuracy: {:.3f}%\t\t\t'
                                    'Age Accuracy: {:.3f}%'
                                    ''.format(epoch + 1 + start_epoch,
                                              batch_idx + 1,
                                              epoch_loss / (batch_idx + 1),
                                              100. * epoch_correct1 / (batch_idx + 1),
                                              100. * epoch_correct2 / (batch_idx + 1),
                                              100. * epoch_correct3 / (batch_idx + 1)
                                              )
                                    )
                avg_loss = epoch_loss / num_train_batches
                avg_accuracy1 = epoch_correct1 / num_train_batches
                avg_accuracy2 = epoch_correct2 / num_train_batches
                avg_accuracy3 = epoch_correct3 / num_train_batches

                # Adjust learning rate
                scheduler.step(avg_loss)  # Using training loss

            # Testing stage
            if len(test_data_loader) != 0.:
                epoch_correct1_val = epoch_correct2_val = epoch_correct3_val = 0.
                val_accuracy_at_observation_levels = np.zeros((11, 3))
                jcm_model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target1, target2, target3) in enumerate(test_data_loader, 0):
                        target1 = target1.squeeze(0)
                        target2 = target2.squeeze(0)
                        target3 = target3.squeeze(0)
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
                            data, target1, target2, target3 = Variable(data.to(device)), Variable(target1.to(device)), \
                                                              Variable(target2.to(device)), Variable(target3.to(device))
                        if enable_augment:
                            data *= Variable(torch.randn_like(data) * 0.0005 + 1)
                        logit_outs, softmax_outs = jcm_model(data)
                        action_class_loss = (criterion1(logit_outs[0], target1) + mu * l1_term(jcm_model.fc1))
                        subject_class_loss = (criterion2(logit_outs[1], target2) + mu * l1_term(jcm_model.fc2))
                        age_class_loss = (criterion3(logit_outs[2], target3) + mu * l1_term(jcm_model.fc3))
                        loss = (0.9 * lambda_weight * action_class_loss * rescale_loss_factor +
                                0.9 * (1 - lambda_weight) * enable_multitask * subject_class_loss +
                                0.1 * enable_multitask * age_class_loss)
                        pred1_idx = torch.max(softmax_outs[0], -1)[1]
                        pred2_idx = torch.max(softmax_outs[1], -1)[1]
                        pred3_idx = torch.max(softmax_outs[2], -1)[1]
                        all_y1_true = target1.cpu().numpy().flatten()
                        all_y2_true = target2.cpu().numpy().flatten()
                        all_y3_true = target3.cpu().numpy().flatten()
                        all_y1_pred = pred1_idx.cpu().numpy().flatten()
                        all_y2_pred = pred2_idx.cpu().numpy().flatten()
                        all_y3_pred = pred3_idx.cpu().numpy().flatten()
                        y1_true = get_data_at_observation_levels(all_y1_true)
                        y2_true = get_data_at_observation_levels(all_y2_true)
                        y3_true = get_data_at_observation_levels(all_y3_true)
                        y1_pred = get_data_at_observation_levels(all_y1_pred)
                        y2_pred = get_data_at_observation_levels(all_y2_pred)
                        y3_pred = get_data_at_observation_levels(all_y3_pred)
                        epoch_correct1_val += accuracy_score(all_y1_true, all_y1_pred)
                        epoch_correct2_val += accuracy_score(all_y2_true, all_y2_pred)
                        epoch_correct3_val += accuracy_score(all_y3_true, all_y3_pred)
                        for i in range(11):
                            val_accuracy_at_observation_levels[i, 0] += (y1_true[i] == y1_pred[i])
                            val_accuracy_at_observation_levels[i, 1] += (y2_true[i] == y2_pred[i])
                            val_accuracy_at_observation_levels[i, 2] += (y3_true[i] == y3_pred[i])
                        epoch_loss_val += loss.item()
                avg_loss_val = epoch_loss_val / len(test_data_loader)   # divided by number of batches
                val_accuracy_at_observation_levels /= num_test_samples
                avg_accuracy1_val = epoch_correct1_val / num_test_samples
                avg_accuracy2_val = epoch_correct2_val / num_test_samples
                avg_accuracy3_val = epoch_correct3_val / num_test_samples
                logger.info('\n=================================================================================='
                            '===========================================\n'
                            'Testing Epoch {}\t\t\t'
                            'Loss: {:.3f}\n'
                            'Action Accuracies: {} (avg: {:.3f})\n'
                            'Subject Accuracies: {} (avg: {:.3f})\n'
                            'Age Accuracies: {} (avg: {:.3f})\n'
                            '==========================================='
                            '==================================================================================\n'
                            ''.format(epoch + 1 + start_epoch,
                                      avg_loss_val,
                                      list(np.round(val_accuracy_at_observation_levels[:, 0], decimals=4)),
                                      avg_accuracy1_val,
                                      list(np.round(val_accuracy_at_observation_levels[:, 1], decimals=4)),
                                      avg_accuracy2_val,
                                      list(np.round(val_accuracy_at_observation_levels[:, 2], decimals=4)),
                                      avg_accuracy3_val
                                      )
                            )

            # Record intermediate training results. Only action classification is recorded at the moment.
            loss_plotter.update({
                'log loss': avg_loss,
                'val_log loss': avg_loss_val,
                'action accuracy': avg_accuracy1,
                'val_action accuracy': avg_accuracy1_val,
                'subject accuracy': avg_accuracy2,
                'val_subject accuracy': avg_accuracy2_val,
                'age accuracy': avg_accuracy3,
                'val_age accuracy': avg_accuracy3_val,
            })
            if (epoch + 1 + start_epoch) % display_interval == 0.:
                loss_plotter.draw()
            ok_to_save = (avg_accuracy1_val >= 0.85 and
                          avg_accuracy2_val >= 0.8 and
                          avg_accuracy3_val >= 0.85)
            if (epoch + 1 + start_epoch) % save_interval == 0. or ok_to_save:
                filename = ('./trained/'
                            + '%d/' % attempt_id
                            + str(dataset)
                            + '_Epoch' + str(epoch + 1 + start_epoch)
                            + '_Lambda' + str(lambda_weight)
                            + '_' + RNN_NAME[rnn_type]
                            + '_' + ACTIVATION_NAME[activation_type]
                            + '_' + opt_name + str(hyperparameters[HyperParamType.INIT_LR])
                            + '_dp' + str(int(dropouts[0]*10)) + str(int(dropouts[1]*10)) + str(int(dropouts[2]*10))
                            + '_decayFact' + str(hyperparameters[HyperParamType.LR_DECAY])
                            + '_decayPati' + str(hyperparameters[HyperParamType.LR_DECAY_PATIENCE])
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
                filename += '.tar'
                save_checkpoint_jcm(epoch + start_epoch, jcm_model, optimizer, scheduler, loss_plotter,
                                    hyperparameters, dataset.indices_train, dataset.indices_test, filename)
                print('Intermediate step states saved.')
    except KeyboardInterrupt:
        # Warning: trying to save states here does NOT always work
        print('Training interrupted.')


def save_checkpoint_jcm(epoch: int,
                        model: nn.Module,
                        optimizer: Optimizer,
                        scheduler: lr_scheduler,
                        loss_plotter: PlotLosses,
                        hyperparameters: list,
                        train_indices,
                        test_indices,
                        filename: str):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'loss_plotter': loss_plotter,
             'hyperparameters': hyperparameters, 'train_indices': train_indices,
             'test_indices': test_indices}
    torch.save(state, filename)
