"""Trains VA-JCM models for action and attributes predictions."""
import os
import torch
from torch.backends import cudnn
from livelossplot import PlotLosses
from network.jcr import JCM
from network.jcm_trainer import *
from dataset.skeleton_multitask import deserialize_dataset_multitask
from global_configs import RNNType, ActivationType
from utils.misc import load_checkpoint_jcm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_start_method('spawn', force=True)


if __name__ == '__main__':  # Must run with main for multiprocessing
    # Hardware
    use_gpu = torch.cuda.is_available()     # True      # Training with GPU, application with CPU
    cudnn.enabled = use_gpu
    cudnn.benchmark = use_gpu
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    gpu_count = 1
    use_multiprocess_data_loading = False   # Only set True for datasets not being loaded entirely to RAM
    preload = True
    downsample_factor = 2

    # Dataset
    skeleton_dataset = deserialize_dataset_multitask('D:/Illumine/Y4/METR4901/dataset/'
                                                     'JL_Dataset_translated_crossSample_new.skeldat',
                                                     preload,
                                                     device,
                                                     downsample_factor)
    num_action_classes = skeleton_dataset.label_size
    num_subject_classes = skeleton_dataset.subject_label_size
    num_age_classes = skeleton_dataset.age_label_size
    input_dim = skeleton_dataset.get_joint_number() * 3  # 3D coordinates
    if skeleton_dataset.has_interaction:
        input_dim *= 2
    translated = skeleton_dataset.is_translated
    edged = skeleton_dataset.is_edged
    rotated = skeleton_dataset.is_rotated
    filtered = skeleton_dataset.is_filtered
    # print('Sequence lengths (%d, %d)' % (skeleton_dataset.get_min_seq_len(), skeleton_dataset.get_max_seq_len()))
    total_epoches = 4000

    # Hyperparameters. Comment below to resume training
    use_sgd = False
    enable_augment = True
    enable_va = True
    batch_size = 1  # DO NOT MODIFY THIS FOR NOW
    dropouts = (0.25, 0.25, 0.25)
    rnn_hidden_dims = (100, 110, 100)
    fc_hidden_dims = (100, 110, 100)
    lambda_weight = 0.6
    init_lr = 3e-4
    lr_decay = 0.5     # Note: final learning rate no less than minimum positive number represented by float64
    lr_decay_patience = 5
    rnn_type = RNNType.SRU
    use_lstm = True if rnn_type == RNNType.LSTM else False
    activation_type = ActivationType.SELU
    truncated_length = 5000  # setting this larger than sequence length will result in ordinary BPTT
    use_focal_loss = True
    layer_norm = True
    
    hyperparameters = [use_sgd, use_lstm, enable_augment, enable_va, batch_size, dropouts,
                       rnn_hidden_dims, fc_hidden_dims, lambda_weight, init_lr, lr_decay,
                       lr_decay_patience, rnn_type, activation_type, truncated_length,
                       use_focal_loss, layer_norm]

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
                view_adaptive=enable_va,
                layer_norm=layer_norm,
                enable_augmentation=enable_augment)

    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.99, nesterov=False, weight_decay=1e-7)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=lr_decay,
                                                           patience=lr_decay_patience,
                                                           verbose=True)
    last_epoch = 0
    loss_plotter = PlotLosses()

    '''
    # Uncomment below to resume training
    model, optimizer, scheduler, last_epoch, loss_plotter, hyperparameters = \
        load_checkpoint_jcm('./trained/'
                            'G3D_Dataset_Epoch18_Lambda0.1_SRU_SELU_Adam0.0005_dp222_decayFact0.5_decayPati25_tl64'
                            '_VA_LN_FL_transTrue_edgFalse_rotFalse_filtFalse'
                            '.tar',
                            num_classes,
                            input_dim,
                            device,
                            new_regression_lambda=0.1,
                            init_lr=5e-4,
                            deprecated=False)
    '''
    # Run the trainer
    train_jcm(model, skeleton_dataset, optimizer, scheduler, hyperparameters, total_epoches,
              len(skeleton_dataset.training_set) // 4,  # Log interval
              10,  # Display interval
              10,  # Save interval
              loss_plotter, last_epoch,
              device, gpu_count,
              use_multiprocess_data_loading,
              preload,
              2,    # attempt
              # Extra info
              'trans' + str(translated),
              'edg' + str(edged),
              'rot' + str(rotated),
              'filt' + str(filtered),
              'downsample%d' % downsample_factor
              )
