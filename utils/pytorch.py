import torch
import copy
from utils.misc import deprecated


def unprocessed_collate(batch):
    """
    A dummy function to prevent Pytorch's data loader from converting and stacking batch data.
    :param batch:
    :return:
    """
    return batch    # List of data tuples (sequence, timeline, label)


@deprecated
def custom_collate(batch):
    """This helper function only works for batch training many-to-one RNN."""
    data = [item[0] for item in batch]
    start = [item[1] for item in batch]
    end = [item[2] for item in batch]
    target = [item[3] for item in batch]
    target = torch.LongTensor(target)
    return [data, start, end, target]


@deprecated
def pad_tensor(vec, pad, dim):
    """
    Warning: DO NOT use this function to pad sequence, otherwise the model will not learn probably

    :param vec: tensor to pad
    :param pad: the size to pad to
    :param dim: dimension to pad
    :return: a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec.float(), torch.zeros(*pad_size)], dim=dim)


@deprecated
class PadCollate:
    """
    A variant of callate_fn that pads according to the longest sequence in
    a batch of sequences. Warning: DO NOT use this helper for torch data loader,
    or the model will not learn probably.
    """

    def __init__(self, dim=0):
        """
        :param dim: the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        :param batch: list of (tensor, label)

        :returns
            xs: a tensor of all examples in 'batch' after padding
            ys: a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))

        # pad according to max_len
        batch = map(lambda x: (pad_tensor(x[0], pad=max_len, dim=self.dim), x[1]), batch)
        temp = copy.deepcopy(batch)

        # stack all
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.LongTensor(list(map(lambda x: x[1], temp)))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


def get_gaussian_confidence(t, event_time, sigma):
    """Event can be either start or end of an action. Time is actually frame index for actual sequences."""
    return torch.exp(-0.5 * torch.pow((t - event_time) / sigma, 2.))


def get_confidence_matrix(label_tensor: torch.tensor, num_classes: int, sigma: float):
    seq_len = len(label_tensor)
    confidence_mat = torch.zeros(seq_len, num_classes + 1, 2)
    last_label_idx = num_classes    # denotes unknown action
    start_frames = []
    end_frames = []
    for frame_idx, label_idx in enumerate(label_tensor, 0):
        assert label_idx <= num_classes, 'Unexpected index of action class found.'
        if label_idx != last_label_idx and label_idx != num_classes:     # transition of action class
            if last_label_idx == num_classes:
                # start of an action
                start_frames.append((label_idx, frame_idx))
            elif frame_idx != seq_len - 1:
                # end of an action
                end_frames.append((label_idx, frame_idx))
            last_label_idx = label_idx
        elif frame_idx == seq_len - 1 and label_idx != num_classes:
            # end of an action is the same as end of the sequence (trimmed action sequence)
            end_frames.append((label_idx, frame_idx))
    for start_frame in start_frames:
        confidence_mat[:, start_frame[0], 0] += get_gaussian_confidence(torch.FloatTensor(range(0, seq_len)),
                                                                        start_frame[1],
                                                                        sigma)
    for end_frame in end_frames:
        confidence_mat[:, end_frame[0], 1] += get_gaussian_confidence(torch.FloatTensor(range(0, seq_len)),
                                                                      end_frame[1],
                                                                      sigma)
    confidence_mat = torch.clamp(confidence_mat, min=0, max=1)
    return confidence_mat


@deprecated
def get_confidence_matrix_old(seq_len: int, timeline: torch.tensor, label_idx: int, num_classes: int, sigma: float):
    # This version does not work with multi-class sequence
    timeline = timeline.squeeze()
    confidence_mat = torch.zeros(seq_len, num_classes + 1, 2)
    start = int(timeline[0].long().numpy())
    end = int(timeline[-1].long().numpy())
    confidence_mat[:, label_idx, :] = timeline.expand(2, seq_len).permute(1, 0)     # TODO timeline == seq_len
    confidence_mat[:, label_idx, 0] = get_gaussian_confidence(confidence_mat[:, label_idx, 0],
                                                              start,
                                                              sigma)
    confidence_mat[:, label_idx, 1] = get_gaussian_confidence(confidence_mat[:, label_idx, 1],
                                                              end,
                                                              sigma)
    return confidence_mat


def is_model_on_gpu(model: torch.nn.Module):
    return next(model.parameters()).is_cuda
