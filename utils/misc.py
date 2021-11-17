import datetime
import sys
import time
import os
import torch
from numpy import mean
from global_configs import HyperParamType, RNNType, ActivationType
from livelossplot import PlotLosses
from network.jcr import *


def get_folders_and_files(directory: str):
    """Returns sorted lists of files and folders at a directory."""
    filenames = os.listdir(directory)
    folders = []
    files = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(directory), filename)):  # check if it is a folder
            folders.append(filename)
        elif os.path.isfile(os.path.join(os.path.abspath(directory), filename)):  # check if it is a file
            files.append(filename)
    return sorted(folders), sorted(files)


def is_file_empty(path):
    return os.stat(path).st_size == 0


class ProgressBar:
    """Progress bar to display in console."""
    def __init__(self, length, max_value):
        assert length > 0 and max_value > 0
        self.length, self.max_value, self.start = length, max_value, time.time()

    def update(self, value):
        assert 0 < value <= self.max_value
        delta = (time.time() - self.start) * (self.max_value - value) / value
        format_spec = [value / self.max_value,
                       value,
                       len(str(self.max_value)),
                       self.max_value,
                       len(str(self.max_value)),
                       '#' * int((self.length * value) / self.max_value),
                       self.length,
                       datetime.timedelta(seconds=int(delta))
                       if delta < 60 * 60 * 10 else '-:--:-']

        sys.stdout.write('\r{:=5.0%} ({:={}}/{:={}}) [{:{}}] ETA: {}'.format(*format_spec))
        if value == self.max_value:
            print()     # Add a new line char


def write_sequence_labels_to_file(y_seq, filename: str, null_class_index: int, confidence_seq=None):
    """Converts framewise predictions to intervals and writes to files for further evaluations."""
    prev_class_index = null_class_index
    start_frame = 0
    sequence_end_index = len(y_seq) - 1
    if confidence_seq is None:
        confidence_seq = [1.0] * len(y_seq)
    with open(filename, 'w+') as f:
        for idx, class_index in enumerate(y_seq):
            if class_index != prev_class_index or idx == sequence_end_index:
                end_frame = idx - 1
                avg_confidence = mean(confidence_seq[start_frame:end_frame+1]).item()
                f.write('%d, %d, %d, %.4f\n' % (prev_class_index, start_frame, end_frame, avg_confidence))
                start_frame = idx
                prev_class_index = class_index


def load_checkpoint_jcm(filename: str,
                        num_classes: tuple,
                        input_dim: int,
                        device: torch.device,
                        init_lr: float = None):
    """Loads a trained VA-JCM model and its states."""
    checkpoint = torch.load(filename, map_location='cpu')
    hyperparameters = checkpoint['hyperparameters']

    use_sgd = hyperparameters[HyperParamType.USE_SGD]
    aug = hyperparameters[HyperParamType.ENABLE_AUGMENT]
    va_on = hyperparameters[HyperParamType.ENABLE_VA]
    dropouts = hyperparameters[HyperParamType.DROPOUTS]
    rnn_hidden_dims = hyperparameters[HyperParamType.RNN_HIDDEN_DIMS]
    fc_hidden_dims = hyperparameters[HyperParamType.FC_HIDDEN_DIMS]
    lr_decay = hyperparameters[HyperParamType.LR_DECAY]
    lr_decay_patience = hyperparameters[HyperParamType.LR_DECAY_PATIENCE]

    try:
        rnn_type = hyperparameters[HyperParamType.RNN_TYPE]
        activation_type = hyperparameters[HyperParamType.ACTIVATION_TYPE]
    except IndexError:  # previously trained model
        rnn_type = RNNType.LSTM if hyperparameters[HyperParamType.USE_LSTM] else RNNType.GRU
        activation_type = ActivationType.SELU

    try:
        use_layer_norm = hyperparameters[HyperParamType.USE_LAYER_NORM]
    except IndexError:  # previously trained model
        use_layer_norm = True if rnn_type == RNNType.SRU else False

    num_action_classes, num_subject_classes, num_age_classes = num_classes
    model = JCM(joint_dimensions=input_dim,
                num_action_classes=num_action_classes,
                num_subject_classes=num_subject_classes,
                num_age_classes=num_age_classes,
                rnn_type=rnn_type,
                activation_type=activation_type,
                subnet1_dropout=dropouts[0],
                subnet2_dropout=dropouts[1],
                subnet3_dropout=dropouts[2],
                rnn_hidden_dim1=rnn_hidden_dims[0],
                rnn_hidden_dim2=rnn_hidden_dims[1],
                rnn_hidden_dim3=rnn_hidden_dims[2],
                rnnfc_hidden_dim1=fc_hidden_dims[0],
                rnnfc_hidden_dim2=fc_hidden_dims[1],
                rnnfc_hidden_dim3=fc_hidden_dims[2],
                view_adaptive=va_on,
                layer_norm=use_layer_norm,
                enable_augmentation=aug,
                device=device)
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    if init_lr is not None:
        hyperparameters[HyperParamType.INIT_LR] = init_lr
        load_optim_dict = False
    else:
        init_lr = hyperparameters[HyperParamType.INIT_LR]
        load_optim_dict = True
    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, nesterov=False, weight_decay=1e-7)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=lr_decay_patience,
                                                           factor=lr_decay,
                                                           verbose=True)
    start_epoch = checkpoint['epoch'] + 1
    if load_optim_dict:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    loss_plotter: PlotLosses = checkpoint['loss_plotter']
    train_indices = checkpoint['train_indices']
    test_indices = checkpoint['test_indices']
    print("Loaded checkpoint at epoch {}".format(start_epoch))
    return model, optimizer, scheduler, start_epoch, loss_plotter, hyperparameters, train_indices, test_indices


def load_checkpoint(filename: str,
                    num_classes: int,
                    input_dim: int,
                    device: torch.device,
                    new_regression_lambda: float = None,
                    init_lr: float = None):
    """Loads a trained VA-JCR model and its states."""
    checkpoint = torch.load(filename, map_location='cpu')
    hyperparameters = checkpoint['hyperparameters']

    use_sgd = hyperparameters[HyperParamType.USE_SGD]
    aug = hyperparameters[HyperParamType.ENABLE_AUGMENT]
    va_on = hyperparameters[HyperParamType.ENABLE_VA]
    dropouts = hyperparameters[HyperParamType.DROPOUTS]
    rnn_hidden_dims = hyperparameters[HyperParamType.RNN_HIDDEN_DIMS]
    fc_hidden_dims = hyperparameters[HyperParamType.FC_HIDDEN_DIMS]
    if new_regression_lambda is None:
        new_regression_lambda = hyperparameters[HyperParamType.REGRESS_LAMBDA]
    else:
        hyperparameters[HyperParamType.REGRESS_LAMBDA] = new_regression_lambda
    lr_decay = hyperparameters[HyperParamType.LR_DECAY]
    lr_decay_patience = hyperparameters[HyperParamType.LR_DECAY_PATIENCE]

    try:
        rnn_type = hyperparameters[HyperParamType.RNN_TYPE]
        activation_type = hyperparameters[HyperParamType.ACTIVATION_TYPE]
    except IndexError:  # previously trained model
        rnn_type = RNNType.LSTM if hyperparameters[HyperParamType.USE_LSTM] else RNNType.GRU
        activation_type = ActivationType.SELU

    try:
        use_layer_norm = hyperparameters[HyperParamType.USE_LAYER_NORM]
    except IndexError:  # previously trained model
        use_layer_norm = True if rnn_type == RNNType.SRU else False

    model = JCR(joint_dimensions=input_dim,
                num_classes=num_classes,
                rnn_type=rnn_type,
                activation_type=activation_type,
                subnet1_dropout=dropouts[0],
                subnet2_dropout=dropouts[1],
                subnet3_dropout=dropouts[2],
                rnn_hidden_dim1=rnn_hidden_dims[0],
                rnn_hidden_dim2=rnn_hidden_dims[1],
                rnn_hidden_dim3=rnn_hidden_dims[2],
                rnnfc_hidden_dim1=fc_hidden_dims[0],
                rnnfc_hidden_dim2=fc_hidden_dims[1],
                rnnfc_hidden_dim3=fc_hidden_dims[2],
                view_adaptive=va_on,
                enable_regression=(new_regression_lambda > 0),
                layer_norm=use_layer_norm,
                enable_augmentation=aug,
                device=device)
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    if init_lr is not None:
        hyperparameters[HyperParamType.INIT_LR] = init_lr
        load_optim_dict = False
    else:
        init_lr = hyperparameters[HyperParamType.INIT_LR]
        load_optim_dict = True
    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, nesterov=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=lr_decay_patience,
                                                           factor=lr_decay,
                                                           verbose=True)
    start_epoch = checkpoint['epoch'] + 1
    if load_optim_dict:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    loss_plotter: PlotLosses = checkpoint['loss_plotter']
    print("Loaded checkpoint at epoch {}".format(start_epoch))
    return model, optimizer, scheduler, start_epoch, loss_plotter, hyperparameters


def chunks(l: list, n: int):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
