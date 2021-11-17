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


def get_confidence_matrix(seq_len: int, timeline: torch.tensor, label_idx: int, num_classes: int, sigma: float):
    assert len(timeline) <= seq_len, 'Sequence must cover the whole action.'
    timeline = timeline.squeeze()
    confidence_mat = torch.zeros(seq_len, num_classes + 1, 2)
    start = int(timeline[0].long().numpy())
    end = int(timeline[-1].long().numpy())
    confidence_mat[:, label_idx, :] = timeline.expand(2, seq_len).permute(1, 0)
    confidence_mat[:, label_idx, 0] = get_gaussian_confidence(confidence_mat[:, label_idx, 0],
                                                              start,
                                                              sigma)
    confidence_mat[:, label_idx, 1] = get_gaussian_confidence(confidence_mat[:, label_idx, 1],
                                                              end,
                                                              sigma)
    return confidence_mat


def is_model_on_gpu(model: torch.nn.Module):
    return next(model.parameters()).is_cuda
