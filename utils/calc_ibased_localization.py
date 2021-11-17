"""
A supplementary module to calculate I-based localization scores.

TODO: integrate this to the evaluation module.
"""
import os
import math
import numpy as np


PRED = './pred_files/'
TRUE = './true_files/'
NULL_CLASS = 10


def iou(prop, gt):
    l_p, s_p, e_p = prop
    l_g, s_g, e_g = gt
    if l_p != l_g:
        return 0
    denominator = max(e_p, e_g) - min(s_p, s_g)
    if not denominator:     # avoid division by zero, i.e. one-frame prediction
        return 0
    return (min(e_p, e_g) - max(s_p, s_g)) / denominator
    
    
def get_localization_scores(predicted_start: int, predicted_end: int, true_start: int, true_end: int):
    """
    exp(-abs(t_pred_start-t_start)/(t_end-t_start))
    exp(-abs(t_pred_end-t_end)/(t_end-t_start))
    :param predicted_start:
    :param predicted_end:
    :param true_start:
    :param true_end:
    """
    if true_end - true_start <= 0:
        return 0, 0
    base = math.exp(1 / (true_start - true_end))
    return base ** abs(predicted_start - true_start), base ** abs(predicted_end - true_end)


def get_folders_and_files(directory: str):
    filenames = os.listdir(directory)
    folders = []
    files = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(directory), filename)):  # check if it is a folder
            folders.append(filename)
        elif os.path.isfile(os.path.join(os.path.abspath(directory), filename)):  # check if it is a file
            files.append(filename)
    return sorted(folders), sorted(files)
    
    
def main():
    _, txts = get_folders_and_files(PRED)
    a_sl, a_el = 0, 0
    v_count = 0
    for txt in txts:
        if not txt.endswith('.txt'):
            continue
        preds = np.loadtxt(PRED + txt, delimiter=',', dtype=object)
        trues = np.loadtxt(TRUE + txt, delimiter=',', dtype=object)
        if len(preds.shape) == 1:
            preds = preds[None, ...]
        if len(trues.shape) == 1:
            trues = trues[None, ...]
        preds = preds[:, :-1].astype(np.int)
        trues = trues[:, :-1].astype(np.int)
        true_count = 0
        v_sl, v_el = 0, 0
        for true in trues:
            if true[0] == NULL_CLASS:
                continue
            sl_scores = []
            el_scores = []
            for pred in preds:
                theta = iou(pred, true)
                if theta <= 0:
                    continue
                sl, el = get_localization_scores(pred[1], pred[2], true[1], true[2])
                sl_scores.append(sl)
                el_scores.append(el)
            if len(sl_scores) != 0:
                v_sl += np.mean(sl_scores)
            if len(el_scores) != 0:
                v_el += np.mean(el_scores)
            true_count += 1
        a_sl += v_sl / true_count
        a_el += v_el / true_count
        v_count += 1
    a_sl /= v_count
    a_el /= v_count
    print(a_sl, a_el)
    _, files = get_folders_and_files('./')
    for file in files:
        if file.endswith('.log'):
            with open(file, 'a') as f:
                f.write('New SL: %.3f\n' % a_sl) 
                f.write('New EL: %.3f\n' % a_el)


if __name__ == '__main__':
    main()
