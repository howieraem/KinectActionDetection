"""Trains VA-JCR models for action detections and forecasts."""
import os
import torch
from torch.backends import cudnn
from livelossplot import PlotLosses
from network.jcr import JCR
from network.jcr_trainer import *
from dataset.skeleton_abstract import deserialize_dataset
from global_configs import RNNType, ActivationType
from utils.misc import load_checkpoint


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
    new_downsample_factor = None

    # Dataset
    skeleton_dataset = deserialize_dataset('./dataset/'
                                           'OAD_Dataset_translated.skeldat',
                                           preload,
                                           device,
                                           new_downsample_factor=new_downsample_factor)
    num_classes = skeleton_dataset.label_size
    input_dim = skeleton_dataset.get_joint_number() * 3  # 3D coordinates
    if skeleton_dataset.has_interaction:
        input_dim *= 2
    translated = skeleton_dataset.is_translated
    edged = skeleton_dataset.is_edged
    rotated = skeleton_dataset.is_rotated
    filtered = skeleton_dataset.is_filtered

    # Hyperparameters
    total_epoches = 4000
    gamma = 1.0
    acc_thres = 0.8
    ignore_background_class = False

    # ---------------------------------------- Comment below to resume training ----------------------------------------
    use_sgd = False
    enable_augment = False  # DO NOT SET TRUE IF NOT FOR REAL-WORLD DEMO   # and new_downsample_factor == 1
    enable_va = True and not skeleton_dataset.is_rotated
    batch_size = 1  # DO NOT MODIFY THIS FOR NOW
    dropouts = (0.25, 0.25, 0.25)
    rnn_hidden_dims = (100, 110, 100)
    fc_hidden_dims = (100, 110, 100)
    regress_lambda = 0.1
    init_lr = 3e-4
    lr_decay = 0.5     # Note: final learning rate no less than minimum positive number represented by float64
    lr_decay_patience = 10
    rnn_type = RNNType.LSTM
    use_lstm = True if rnn_type == RNNType.LSTM else False
    activation_type = ActivationType.ReLU
    truncated_length = 5000  # setting this larger than sequence length will result in ordinary BPTT
    use_focal_loss = False
    layer_norm = True  # and (rnn_type != RNNType.OTHER)
    
    hyperparameters = [use_sgd, use_lstm, enable_augment, enable_va, batch_size, dropouts,
                       rnn_hidden_dims, fc_hidden_dims, regress_lambda, init_lr, lr_decay,
                       lr_decay_patience, rnn_type, activation_type, truncated_length,
                       use_focal_loss, layer_norm]

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
                view_adaptive=enable_va,
                enable_regression=(regress_lambda > 0),
                layer_norm=layer_norm,
                enable_augmentation=enable_augment,
                device=device)

    if use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.99, nesterov=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=lr_decay,
                                                           patience=lr_decay_patience,
                                                           verbose=True)
    last_epoch = 0
    loss_plotter = PlotLosses()

    '''
    # --------------------------------------- Uncomment below to resume training ---------------------------------------
    model, optimizer, scheduler, last_epoch, loss_plotter, hyperparameters = \
        load_checkpoint('./trained/'
                        'G3D_Dataset_lambda9.0_SRU_ReLU_Adam0.0003_dp222_decayFact0.5_decayPati10_tl64_VA_LN_FL_aug'
                        '_transTrue_edgFalse_rotFalse_filtFalse_gamma1.0_Epoch28'
                        '.tar',
                        num_classes,
                        input_dim,
                        device,
                        new_regression_lambda=10.0,
                        init_lr=3e-4,
                        deprecated=False)
    '''
    # ---------------------------------------------- Do NOT comment below ----------------------------------------------
    # Run the trainer
    train_jcr(model, skeleton_dataset, optimizer, scheduler, hyperparameters, total_epoches,
              len(skeleton_dataset.training_set) // 4,  # Log interval
              10,  # Display interval
              20,  # Save interval
              loss_plotter, last_epoch,
              device, gpu_count,
              use_multiprocess_data_loading,
              preload,
              gamma,
              acc_thres,
              ignore_background_class,
              # Extra info
              'trans' + str(translated),
              'edg' + str(edged),
              'rot' + str(rotated),
              'filt' + str(filtered),
              )
