# TODO 1. solve batch variable sequence length problem
# TODO 2. might reimplement as a class instead
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from livelossplot import PlotLosses
from jcr.rnn import JcrGRU
from dataset.skeleton import SkeletonDataset
from utils.pytorch import *
from utils.misc import deprecated


import signal
signal.signal(signal.SIGINT, signal.default_int_handler)    # Ensuring all keyboard interrupts are captured


def train_jcr(jcr_model: nn.Module, dataset: SkeletonDataset, batch_size: int,
              optimizer: Optimizer, scheduler: lr_scheduler,
              regression_sigma: float, regression_lambda: float,
              epoches, log_interval, display_interval, save_interval,
              loss_plotter: PlotLosses, start_epoch: int=0, use_gpu: bool=True,
              use_multiprocess: bool=False, dropouts: tuple=(),
              *args, **kwargs):
    assert is_model_on_gpu(jcr_model) == use_gpu, 'Model not on the required device (CPU/GPU).'
    train_data_loader = DataLoader(dataset.training_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=unprocessed_collate,
                                   num_workers=4 if use_multiprocess else 0)
    test_data_loader = DataLoader(dataset.testing_set,
                                  batch_size=1,
                                  shuffle=False)
    num_targets = dataset.label_size()
    avg_loss = avg_accuracy = avg_loss_val = avg_accuracy_val = np.nan
    try:
        classification_criterion = torch.nn.CrossEntropyLoss()
        regression_criterion = torch.nn.MSELoss()
        if use_gpu:
            classification_criterion.cuda(0)
            regression_criterion.cuda(0)

        for epoch in range(epoches):
            epoch_loss = epoch_correct = epoch_loss_val = epoch_correct_val = 0.
            # Training stage
            if len(train_data_loader) != 0.:
                jcr_model.train()
                for batch_idx, samples in enumerate(train_data_loader, 0):
                    classification_loss = Variable(torch.zeros(1))
                    regression_loss = Variable(torch.zeros(1))
                    if use_gpu:
                        classification_loss, regression_loss = classification_loss.cuda(0), regression_loss.cuda(0)
                    batch_correct = 0.
                    optimizer.zero_grad()
                    for (data, target) in samples:
                        confidence = get_confidence_matrix(target, num_targets, regression_sigma)
                        # target = torch.LongTensor([label]).expand(seq_len)  # Label vector to match sequence length
                        if use_gpu:
                            data, confidence, target = \
                                Variable(data.cuda(0)), Variable(confidence.cuda(0)), Variable(target.cuda(0))
                        else:
                            data, confidence, target = Variable(data), Variable(confidence), Variable(target)
                        classification_out, softmax_out, regression_out = jcr_model(data.unsqueeze(0))
                        current_correct = (softmax_out.max(1)[1] == target).sum().item() / len(data)
                        batch_correct += current_correct
                        classification_loss += classification_criterion(classification_out, target)
                        regression_loss += regression_criterion(regression_out, confidence)
                    loss = (classification_loss + regression_lambda * regression_loss) / len(samples)
                    loss.backward()
                    optimizer.step()
                    current_loss = loss.item()
                    epoch_loss += current_loss
                    epoch_correct += batch_correct
                    if batch_idx % log_interval == 0:
                        print('Training Epoch {} Batch {} \t\t\tLoss: {:.3f}\t\t\tClassification Accuracy: '
                              '{:.3f}%'.format(epoch + 1 + start_epoch,
                                               batch_idx + 1,
                                               epoch_loss / (batch_idx + 1),
                                               100. * epoch_correct / ((batch_idx + 1) * len(samples))))
                avg_loss = epoch_loss / len(train_data_loader)      # divided by number of batches
                avg_accuracy = epoch_correct / len(train_data_loader.dataset)   # divided by total number of data

                # Adjust learning rate
                scheduler.step(avg_loss)  # Using training loss

            # Testing stage
            if len(test_data_loader) != 0.:
                jcr_model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(test_data_loader, 0):
                        confidence = get_confidence_matrix(target.squeeze(), num_targets, regression_sigma)
                        # target = torch.LongTensor([label]).expand(seq_len)  # Label vector to match sequence length
                        if use_gpu:
                            data, confidence, target = \
                                Variable(data.cuda(0)), Variable(confidence.cuda(0)), Variable(target.cuda(0))
                        else:
                            data, confidence, target = Variable(data), Variable(confidence), Variable(target)
                        classification_out, softmax_out, regression_out = jcr_model(data)
                        classification_loss = classification_criterion(classification_out, target.squeeze())
                        regression_loss = regression_criterion(regression_out, confidence)
                        loss = classification_loss + regression_lambda * regression_loss
                        current_loss = loss.item()
                        current_correct = (softmax_out.max(1)[1] == target).sum().item() / data.shape[1]
                        epoch_loss_val += current_loss
                        epoch_correct_val += current_correct
                avg_loss_val = epoch_loss_val / len(test_data_loader)   # divided by number of batches
                avg_accuracy_val = epoch_correct_val / len(test_data_loader.dataset)
                print('Testing Epoch {}\t\t\tLoss: {:.3f}\t\t\tClassification Accuracy: {:.3f}%'.format(
                    epoch + 1 + start_epoch,
                    avg_loss_val,
                    100. * avg_accuracy_val))

            # Record intermediate training results
            loss_plotter.update({
                'log loss': avg_loss,
                'val_log loss': avg_loss_val,
                'accuracy': avg_accuracy,
                'val_accuracy': avg_accuracy_val
            })
            if (epoch + 1 + start_epoch) % display_interval == 0.:
                loss_plotter.draw()
            if (epoch + 1 + start_epoch) % save_interval == 0.:
                filename = str(dataset) + '_Epoch_' + str(epoch + 1 + start_epoch) + \
                           '_Lambda_' + str(regression_lambda)
                for arg in args:
                    filename += '_' + str(arg)
                filename += '.tar'
                save_checkpoint(epoch + start_epoch, jcr_model, optimizer, scheduler, loss_plotter,
                                dataset, dropouts, filename)
                print('Intermediate step states saved.')
    except KeyboardInterrupt:
        # Warning: trying to save states here does NOT always work
        print('Training interrupted.')


def resume_training(filename: str, init_learning_rate: float, lr_decay: float, lr_decay_step: int,
                    batch_size: int, regression_sigma: float, regression_lambda: float,
                    epoches, log_interval, display_interval, save_interval,
                    use_gpu: bool=True, use_multiprocess: bool=False, dropouts: tuple=(),
                    *args, **kwargs):
    model, optimizer, scheduler, last_epoch, loss_plotter, skeleton_dataset = \
        load_checkpoint(filename,
                        dropouts=dropouts,
                        initial_lr=init_learning_rate,
                        lr_decay=lr_decay,
                        lr_decay_step=lr_decay_step)
    train_jcr(model, skeleton_dataset, batch_size,
              optimizer, scheduler,
              regression_sigma, regression_lambda,
              epoches,
              log_interval,
              display_interval,
              save_interval,
              loss_plotter,
              last_epoch,
              use_gpu,
              use_multiprocess,
              dropouts,
              args  # extra info
              )


@deprecated
def train_jcr_old(jcr_model: nn.Module, dataset: SkeletonDataset, batch_size: int,
                  optimizer: Optimizer, scheduler: lr_scheduler,
                  regression_sigma: float, regression_lambda: float,
                  epoches, log_interval, display_interval, save_interval,
                  loss_plotter: PlotLosses, start_epoch: int=0, use_gpu: bool=True,
                  use_multiprocess: bool=False, dropouts: tuple=(),
                  *args, **kwargs):
    assert is_model_on_gpu(jcr_model) == use_gpu, 'Model not on the required device (CPU/GPU).'
    train_data_loader = DataLoader(dataset.training_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=unprocessed_collate,
                                   num_workers=4 if use_multiprocess else 0)
    test_data_loader = DataLoader(dataset.testing_set,
                                  batch_size=1,
                                  shuffle=False)
    avg_loss = avg_accuracy = avg_loss_val = avg_accuracy_val = np.nan
    num_targets = dataset.label_size()
    try:
        classification_criterion = torch.nn.CrossEntropyLoss()
        regression_criterion = torch.nn.MSELoss()
        if use_gpu:
            classification_criterion.cuda(0)
            regression_criterion.cuda(0)

        for epoch in range(epoches):
            epoch_loss = 0.
            epoch_correct = 0
            epoch_loss_val = 0.
            epoch_correct_val = 0

            # Training stage
            if len(train_data_loader) != 0.:
                jcr_model.train()
                for batch_idx, (data, timeline, label) in enumerate(train_data_loader, 0):
                    seq_len = data.shape[1]
                    label_idx = label.long().numpy().data[0]
                    confidence = get_confidence_matrix_old(seq_len, timeline, label_idx, num_targets,
                                                           regression_sigma)
                    label = label.expand(seq_len)   # Label vector to match sequence length
                    if use_gpu:
                        data, confidence, label = Variable(data.cuda(0)), Variable(confidence.cuda(0)), \
                                                   Variable(label.cuda(0))
                    else:
                        data, confidence, label = Variable(data), Variable(confidence), Variable(label)
                    optimizer.zero_grad()
                    classification_out, softmax_out, regression_out = jcr_model(data)
                    classification_loss = classification_criterion(classification_out, label)
                    regression_loss = regression_criterion(regression_out, confidence)
                    loss = classification_loss + regression_lambda * regression_loss
                    loss.backward()
                    optimizer.step()
                    current_loss = loss.item()
                    current_correct = (softmax_out.max(1)[1] == label).sum().item() / seq_len
                    epoch_loss += current_loss
                    epoch_correct += current_correct
                    if batch_idx % log_interval == 0:
                        print('Training Epoch {} Batch {} \t\t\tLoss: {:.3f}\t\t\tClassification Accuracy: '
                              '{:.3f}%'.format(epoch + 1 + start_epoch,
                                               batch_idx + 1,
                                               epoch_loss / (batch_idx + 1),
                                               100. * epoch_correct / ((batch_idx + 1) * len(data))))
                avg_loss = epoch_loss / len(train_data_loader)  # divided by number of batches
                avg_accuracy = epoch_correct / len(train_data_loader.dataset)

                # Adjust learning rate
                scheduler.step(avg_loss)  # Using training loss

            # Testing stage
            if len(test_data_loader) != 0.:
                jcr_model.eval()
                with torch.no_grad():
                    for batch_idx, (data, timeline, label) in enumerate(test_data_loader, 0):
                        seq_len = data.shape[1]
                        label_idx = label.long().numpy().data[0]
                        confidence = get_confidence_matrix_old(seq_len, timeline, label_idx, num_targets,
                                                               regression_sigma)
                        label = label.expand(seq_len)   # Label vector to match sequence length
                        if use_gpu:
                            data, confidence, label = Variable(data.cuda(0)), \
                                                      Variable(confidence.cuda(0)), \
                                                      Variable(label.cuda(0))
                        else:
                            data, confidence, label = Variable(data), \
                                                      Variable(confidence), \
                                                      Variable(label)
                        classification_out, softmax_out, regression_out = jcr_model(data)
                        classification_loss = classification_criterion(classification_out, label)
                        regression_loss = regression_criterion(regression_out, confidence)
                        loss = classification_loss + regression_lambda * regression_loss
                        current_loss = loss.item()
                        current_correct = (softmax_out.max(1)[1] == label).sum().item() / seq_len
                        epoch_loss_val += current_loss
                        epoch_correct_val += current_correct
                avg_loss_val = epoch_loss_val / len(test_data_loader)   # divided by number of batches
                avg_accuracy_val = epoch_correct_val / len(test_data_loader.dataset)
                print('Testing Epoch {}\t\t\tLoss: {:.3f}\t\t\tClassification Accuracy: {:.3f}%'.format(
                    epoch + 1 + start_epoch,
                    avg_loss_val,
                    100. * avg_accuracy_val))

            # Record intermediate training results
            loss_plotter.update({
                'log loss': avg_loss,
                'val_log loss': avg_loss_val,
                'accuracy': avg_accuracy,
                'val_accuracy': avg_accuracy_val
            })
            if (epoch + 1 + start_epoch) % display_interval == 0.:
                loss_plotter.draw()
            if (epoch + 1 + start_epoch) % save_interval == 0.:
                filename = str(dataset) + '_Epoch_' + str(epoch + 1 + start_epoch) + \
                           '_Lambda_' + str(regression_lambda)
                for arg in args:
                    filename += '_' + str(arg)
                filename += '.tar'
                save_checkpoint(epoch + start_epoch, jcr_model, optimizer, scheduler, loss_plotter,
                                dataset, dropouts, filename)
                print('Intermediate step states saved.')
    except KeyboardInterrupt:
        # Warning: trying to save states here does NOT always work
        print('Training interrupted.')


def save_checkpoint(epoch: int,
                    model: nn.Module,
                    optimizer: Optimizer,
                    scheduler: lr_scheduler,
                    loss_plotter: PlotLosses,
                    dataset: SkeletonDataset,
                    dropouts: tuple,
                    filename: str):
    state = {'epoch': epoch, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'loss_plotter': loss_plotter, 'dropout': dropouts, 'dataset': dataset}
    torch.save(state, filename)


def load_checkpoint(filename: str,
                    dropouts: tuple=(),
                    initial_lr: float=1e-5,
                    lr_decay: float=0.1,
                    lr_decay_step: int=50,
                    hidden_dims: tuple=(100, 110, 100),
                    use_sgd_optimizer=False,
                    use_gpu=True):
    """

    :param filename:
    :param dropouts:
    :param initial_lr:
    :param lr_decay:
    :param lr_decay_step:
    :param hidden_dims:
    :param use_sgd_optimizer:
    :param use_gpu:
    :return:
    """
    if use_gpu:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch'] + 1
    dataset: SkeletonDataset = checkpoint['dataset']
    num_classes = dataset.label_size()
    input_dim = dataset.get_joint_number() * 3  # 3D coordinates

    if len(dropouts) != 0:
        assert len(dropouts) == 3, 'The JCR-RNN model needs exactly three dropouts.'
    else:
        try:
            dropouts: tuple = checkpoint['dropouts']
        except KeyError:
            dropouts = (0.3, 0.5, 0.5)

    if use_gpu:
        model = JcrGRU(input_dim, num_classes, 1,
                       subnet1_dropout=dropouts[0],
                       subnet2_dropout=dropouts[1],
                       subnet3_dropout=dropouts[2],
                       subnet1_hidden_dim=hidden_dims[0],
                       subnet2_hidden_dim=hidden_dims[1],
                       subnet3_hidden_dim=hidden_dims[2]
                       ).cuda(0)
    else:
        model = JcrGRU(input_dim, num_classes, 1,
                       subnet1_dropout=dropouts[0],
                       subnet2_dropout=dropouts[1],
                       subnet3_dropout=dropouts[2],
                       subnet1_hidden_dim=hidden_dims[0],
                       subnet2_hidden_dim=hidden_dims[1],
                       subnet3_hidden_dim=hidden_dims[2],
                       use_gpu=False)
    model.load_state_dict(checkpoint['state_dict'])
    if use_sgd_optimizer:
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay)
    scheduler.load_state_dict(checkpoint['scheduler'])
    loss_plotter = checkpoint['loss_plotter']
    print("Loaded checkpoint at epoch {}".format(start_epoch))
    return model, optimizer, scheduler, start_epoch, loss_plotter, dataset
