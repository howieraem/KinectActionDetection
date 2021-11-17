"""Visualize basic statistics of trimmed multi-attribute datasets."""
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pylab as plt
from dataset.skeleton_multitask import deserialize_dataset_multitask


np.set_printoptions(suppress=True, threshold=121)


if __name__ == '__main__':
    dat = deserialize_dataset_multitask('./dataset/'
                                        'JL_Dataset_translated_crossSample12.skeldat',
                                        False)
    dat.downsample_factor = 1
    num_actions, num_subjects = dat.label_size, dat.subject_label_size
    action_instance_counts = np.zeros((num_actions, 1), dtype=np.int)
    action_frame_counts = np.zeros((num_actions, 1), dtype=np.int)
    subject_instance_counts = np.zeros((num_subjects, 1), dtype=np.int)
    subject_frame_counts = np.zeros((num_subjects, 1), dtype=np.int)
    action_subject_counts = np.zeros((num_actions, num_subjects), dtype=np.int)
    for (_, label, subject, _) in dat:
        # ic, fc = count_actions(label, num_classes)
        label_idx, subject_idx = label[0].item(), subject[0].item()
        action_instance_counts[label_idx] += 1
        action_frame_counts[label_idx] += len(label)
        subject_instance_counts[subject_idx] += 1
        subject_frame_counts[subject_idx] += len(label)
        action_subject_counts[label_idx, subject_idx] += 1
    action_labels = np.array(dat.get_labels(), dtype=object)[:, np.newaxis]
    subject_labels = np.arange(1, num_subjects+1)[:, np.newaxis]

    action_ret = np.concatenate((action_labels, action_instance_counts, action_frame_counts), axis=-1)
    action_ret = np.insert(action_ret, 0, ['Actions', 'Interval Count', 'Frame Count'], axis=0)
    # print(sum(instance_counts))
    # print(ret)
    action_df = pd.DataFrame(action_ret[1:, 1:], index=action_ret[1:, 0], columns=action_ret[0, 1:])
    # print(df)
    fig1 = plt.figure()
    ax1 = action_df.plot(kind='bar', secondary_y='Frame Count', rot=75, fontsize=10, figsize=(12, 5))
    ax1.set_xlabel('Actions')
    plt.show()

    subject_ret = np.concatenate((subject_labels,
                                  subject_instance_counts,
                                  subject_frame_counts), axis=-1).astype(object)
    subject_ret = np.insert(subject_ret, 0, ['Subjects', 'Interval Count', 'Frame Count'], axis=0)
    subject_df = pd.DataFrame(subject_ret[1:, 1:], index=subject_ret[1:, 0], columns=subject_ret[0, 1:])
    fig2 = plt.figure()
    ax2 = subject_df.plot(kind='bar', secondary_y='Frame Count', rot=45, fontsize=12, figsize=(15, 6.5))
    ax2.set_xlabel('Subjects')
    plt.show()

    fig3 = plt.figure(figsize=(16, 4))
    hm = pd.DataFrame(action_subject_counts,
                      index=list(action_labels.squeeze()),
                      columns=list(subject_labels.squeeze()))
    heatmap = sn.heatmap(hm, annot=True, square=True, annot_kws={'size': 9},
                         xticklabels=True, yticklabels=True,
                         cbar=False,
                         cmap='Blues')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=0, ha='right', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=75, ha='right', fontsize=10)
    plt.xlabel('Subjects', fontsize=10)
    plt.ylabel('Actions', fontsize=10)
    plt.gcf().subplots_adjust(left=0.3, bottom=0.15)
    plt.show()
