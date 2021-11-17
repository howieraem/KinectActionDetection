"""Visualize basic statistics of untrimmed datasets."""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from dataset.skeleton_abstract import deserialize_dataset


np.set_printoptions(suppress=True, threshold=121)


def count_actions(y_seq, label_size):
    ret1 = np.zeros((label_size, 1), dtype=np.int)
    ret2 = np.zeros((label_size, 1), dtype=np.int)
    prev_class_index = label_size - 1
    start_frame = 0
    sequence_end_index = len(y_seq) - 1
    for i, class_index in enumerate(y_seq):
        if class_index != prev_class_index or i == sequence_end_index:
            end_frame = i - 1
            ret1[prev_class_index] += 1
            ret2[prev_class_index] += (end_frame - start_frame + 1)
            start_frame = i
            prev_class_index = class_index
    return ret1, ret2


if __name__ == '__main__':
    dat = deserialize_dataset('./dataset/'
                              'G3D_Dataset_translated.skeldat',
                              False)
    dat.downsample_factor = 1
    num_classes = dat.label_size
    instance_counts = np.zeros((num_classes, 1), dtype=np.int)
    frame_counts = np.zeros((num_classes, 1), dtype=np.int)
    for (_, label, _) in dat:
        ic, fc = count_actions(label, num_classes)
        instance_counts += ic
        frame_counts += fc
    labels = np.array(dat.get_labels(), dtype=object)[:, np.newaxis]
    ret = np.concatenate((labels, instance_counts, frame_counts), axis=-1)
    ret = np.insert(ret, 0, ['Actions', 'Interval Count', 'Frame Count'], axis=0)
    print(sum(instance_counts[:-1]))
    # print(ret[-1])
    df = pd.DataFrame(ret[1:-1, 1:], index=ret[1:-1, 0], columns=ret[0, 1:])
    print(df)

    plt.figure()  # Create matplotlib figure
    df.plot(kind='bar', secondary_y='Frame Count', rot=75, fontsize=10, figsize=(16, 6))
    plt.show()
