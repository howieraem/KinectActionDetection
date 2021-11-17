import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torch.backends import cudnn
from jcr.rnn import JcrGRU
from jcr.trainer import load_checkpoint
from jcr.evaluator import evaluate_dataset
from dataset.skeleton import *


if __name__ == '__main__':  # Must run with main for multiprocessing
    USE_GPU = True      # Training with GPU, application with CPU
    USE_MULTIPROCESS_DATA_LOADING = False   # Set true for G3D dataset
    cudnn.enabled = USE_GPU
    cudnn.benchmark = USE_GPU

    skeleton_dataset = SkeletonDatasetCornell60('D:/Illumine/Y4/METR4901/dataset/Cornell60')
    # print('Maximum sequence length in the dataset: ', skeleton_dataset.get_max_seq_len())
    labels = list(skeleton_dataset.get_labels())
    labels.insert(len(labels), 'unknown')
    num_classes = skeleton_dataset.label_size()
    input_dim = skeleton_dataset.get_joint_number() * 3  # 3D coordinates

    if USE_GPU:
        jcr_model = JcrGRU(input_dim, num_classes, 1)
        jcr_model.cuda(0)
    else:
        jcr_model = JcrGRU(input_dim, num_classes, 1, use_gpu=False)
    sigma = 5.

    jcr_model, _, _, _, _ = \
        load_checkpoint('Cornell_60_Dataset_Epoch_300_Lambda_0.0_GRU_1e-05_fullPlot_batch1.tar',
                        jcr_model)
    confusion_mat, f1, sl, el, acc = evaluate_dataset(jcr_model, skeleton_dataset, USE_GPU, sigma)
    with np.printoptions(precision=3, suppress=True):
        print(f1)
        print(sl, el, acc)

    df_cm = pd.DataFrame(confusion_mat, index=[label for label in labels],
                         columns=[label for label in labels])
    plt.figure(figsize=(13, 9))
    sn.heatmap(df_cm, annot=True)
    plt.show()
