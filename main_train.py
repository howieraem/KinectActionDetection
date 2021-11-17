from torch.backends import cudnn
from jcr.rnn import *
from jcr.trainer import *
from dataset.skeleton import *


if __name__ == '__main__':  # Must run with main for multiprocessing
    USE_GPU = True      # Training with GPU, application with CPU
    USE_MULTIPROCESS_DATA_LOADING = False   # Warning: set True may cause errors
    cudnn.enabled = USE_GPU
    cudnn.benchmark = USE_GPU

    train_ratio = 0.9   # zero means pure testing
    skeleton_dataset = SkeletonDatasetCornell60('D:/Illumine/Y4/METR4901/dataset/Cornell60', train_ratio)
    # print('Maximum sequence length in the dataset: ', skeleton_dataset.get_maximum_sequence_length())
    num_classes = skeleton_dataset.label_size()
    input_dim = skeleton_dataset.get_joint_number() * 3  # 3D coordinates
    batch_size = 1

    if USE_GPU:
        jcr_model = JcrGRU(input_dim, num_classes, 1)
        jcr_model.cuda(0)
    else:
        jcr_model = JcrGRU(input_dim, num_classes, 1, use_gpu=False)
    epoches = 2000
    init_learning_rate = 1e-5
    optimizer = torch.optim.Adam(jcr_model.parameters(), lr=init_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    sigma = 5.
    lambda_weight = 0.
    last_epoch = 0
    loss_plotter = PlotLosses()
    """
    jcr_model, optimizer, scheduler, last_epoch, loss_plotter = \
        load_checkpoint('Cornell_60_Dataset_Epoch_450_Lambda_0.0_GRU_2e-05_fullPlot_batch2.tar',
                        jcr_model, optimizer, scheduler)"""

    train_jcr(jcr_model, skeleton_dataset, batch_size,
              optimizer, scheduler,
              sigma, lambda_weight,
              epoches,
              epoches / 200,    # Log interval
              epoches / 40,     # Display interval
              epoches / 40,     # Save interval
              loss_plotter,
              last_epoch,
              USE_GPU,
              USE_MULTIPROCESS_DATA_LOADING,
              'GRU',                # Extra info
              init_learning_rate,   # Extra info
              'fullPlot',           # Extra info
              'batch' + str(batch_size)     # Extra info
              )
