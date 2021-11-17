"""Test inference speed on CPU with the 9 unused sequences from OAD dataset."""
import time
import torch
from torch.autograd import Variable
from numpy import mean
from utils.misc import load_checkpoint, get_folders_and_files
from dataset.skeleton_abstract import deserialize_dataset


OAD_SPEED_INDICES = [5, 6, 11, 12, 30, 31, 46, 47, 48]
DEVICE = torch.device('cpu')
MODEL_PATH = './validation/OAD_VA+LN+SRU.tar'


if __name__ == '__main__':
    import os
    dat = deserialize_dataset('./dataset/'
                              'OAD_Dataset_translated.skeldat',
                              False)
    model_filenames = get_folders_and_files(MODEL_PATH)[1]
    for model_filename in model_filenames:
        if not model_filename.endswith('.tar'):
            continue
        model = load_checkpoint(MODEL_PATH + model_filename,
                                num_classes=dat.label_size,
                                input_dim=dat.get_joint_number()*3,
                                device=DEVICE)[0]
        framerates = []
        model.eval()
        with torch.no_grad():
            for idx in OAD_SPEED_INDICES:
                seq, _, _ = dat[idx]
                seq = Variable(seq)
                t0 = time.time() * 1000
                for frame in seq:
                    model(frame[None, None, ...])
                dt = (time.time() * 1000 - t0) // 1000
                framerates.append(len(seq) / dt)
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_va_parameters = sum(p.numel() for p in model.va_net.parameters() if p.requires_grad)
        if not model.enable_va:
            num_parameters -= num_va_parameters
        print(model_filename)
        print('\t# Parameters: %.1f K' % (num_parameters / 1000))
        print('\tFramerate: %d fps' % mean(framerates))
    os.system('pause')
