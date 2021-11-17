import random
import math
import numpy as _np
import torch as _torch
import copy
import torch.nn.functional as _F
import transforms3d
from global_configs import *


def get_data_at_observation_levels(seq):
    """Returns data at 10 uniformly sampled indices."""
    if len(seq.shape) == 3:
        indices = ((seq.shape[1] - 1) * _np.arange(0, 1.1, 0.1)).astype(_np.int16)
        return seq[:, indices]
    indices = ((len(seq) - 1) * _np.arange(0, 1.1, 0.1)).astype(_np.int16)
    return seq[indices]


def convert_skeleton_ni_to_ms1(unflattened_ni_joint_array: _np.ndarray, is_cad: bool) -> _np.ndarray:
    """Converts 15-joint skeleton to 20-joint skeleton by re-ordering and zero-padding."""
    zero_padded = _np.zeros((SensorJointNumber.KINECT_V1, 3), dtype=_np.float32)
    conversion_dict = CAD_2_MS1_JOINT_IDX_CONVERT if is_cad else NI_2_MS1_JOINT_IDX_CONVERT
    for ni_joint_idx in range(SensorJointNumber.OPENNI):
        ms1_joint_idx = conversion_dict[ni_joint_idx]
        zero_padded[ms1_joint_idx] = unflattened_ni_joint_array[ni_joint_idx]
    return zero_padded


def preprocess_skeleton_frame(joint_collection, is_15_joint: bool, is_cad: bool = False,
                              to_edge: bool = True,
                              to_rotate: bool = False,
                              is_25_joint: bool = False) -> _np.ndarray:
    """Applies re-ordering, orientation standardzing or edging to a skeleton frame."""
    assert is_15_joint + is_25_joint <= 1
    original_length = len(joint_collection)
    if not isinstance(joint_collection, _np.ndarray):
        joint_collection = _np.array(joint_collection, dtype=_np.float32)
    ret = joint_collection.reshape((len(joint_collection) // 3, 3)).transpose((0, 1))   # unflatten
    if to_rotate:
        ret = standardize_x_axis_framewise(ret)
    if is_15_joint:
        ret = convert_skeleton_ni_to_ms1(ret, is_cad)
    if to_edge:
        ret = joints_to_edges(ret, is_15_joint, is_25_joint)
    return ret.reshape(original_length)     # flatten


def standardize_x_axis_framewise(unflattened_joint_array: _np.ndarray) -> _np.ndarray:
    """
    Standardizes the skeleton orientation by keeping the angle, between yz plane and the edge connecting
    two shoulder joints, to 0 for every frame.
    """
    x_shoulder_right, _, z_shoulder_right = unflattened_joint_array[JointTypeMS1.SHOULDER_RIGHT]
    x_shoulder_left, _, z_shoulder_left = unflattened_joint_array[JointTypeMS1.SHOULDER_LEFT]
    angle = math.atan2(z_shoulder_right - z_shoulder_left, x_shoulder_right - x_shoulder_left)
    rot_mat = get_rotate_mat(0, -angle, 0)
    return _np.matmul(unflattened_joint_array, rot_mat).astype(_np.float32)


def standardize_coordinate_origin_sequence(sequence: _np.ndarray,
                                           is_25_joint: bool = False,
                                           use_hip: bool = False) -> _np.ndarray:
    """Fixes the origin of world coordinates to a skeleton joint-based location in every frame."""
    sequence_array = sequence.copy()
    joints_per_skeleton = SensorJointNumber.KINECT_V2 if is_25_joint else SensorJointNumber.KINECT_V1
    channels = sequence_array.shape[-1]
    if channels == joints_per_skeleton * 3:
        if use_hip:
            hip_centers = _np.tile(sequence_array[:, 3 * JointTypeMS1.HIP_CENTER:3 * JointTypeMS1.SPINE],
                                   joints_per_skeleton)
            hip_lefts = _np.tile(sequence_array[:, 3 * JointTypeMS1.HIP_LEFT:3 * JointTypeMS1.KNEE_LEFT],
                                 joints_per_skeleton)
            hip_rights = _np.tile(sequence_array[:, 3 * JointTypeMS1.HIP_RIGHT:3 * JointTypeMS1.KNEE_RIGHT],
                                  joints_per_skeleton)
            return sequence_array - (hip_centers + hip_lefts + hip_rights) / 3
        return sequence_array - _np.tile(sequence_array[:, 3 * JointTypeMS1.SPINE:3 * JointTypeMS1.SHOULDER_CENTER],
                                         joints_per_skeleton)
    else:
        first_person_array = sequence_array[:, :joints_per_skeleton*3]
        second_person_array = sequence_array[:, joints_per_skeleton*3:]
        if use_hip:
            hip_centers1 = _np.tile(first_person_array[:, 3 * JointTypeMS1.HIP_CENTER:3 * JointTypeMS1.SPINE],
                                    joints_per_skeleton)
            hip_lefts1 = _np.tile(first_person_array[:, 3 * JointTypeMS1.HIP_LEFT:3 * JointTypeMS1.KNEE_LEFT],
                                  joints_per_skeleton)
            hip_rights1 = _np.tile(first_person_array[:, 3 * JointTypeMS1.HIP_RIGHT:3 * JointTypeMS1.KNEE_RIGHT],
                                   joints_per_skeleton)
            hip_centers2 = _np.tile(second_person_array[:, 3 * JointTypeMS1.HIP_CENTER:3 * JointTypeMS1.SPINE],
                                    joints_per_skeleton)
            hip_lefts2 = _np.tile(second_person_array[:, 3 * JointTypeMS1.HIP_LEFT:3 * JointTypeMS1.KNEE_LEFT],
                                  joints_per_skeleton)
            hip_rights2 = _np.tile(second_person_array[:, 3 * JointTypeMS1.HIP_RIGHT:3 * JointTypeMS1.KNEE_RIGHT],
                                   joints_per_skeleton)
            first_person_array -= (hip_centers1 + hip_lefts1 + hip_rights1) / 3
            second_person_array -= (hip_centers2 + hip_lefts2 + hip_rights2) / 3
            return _np.concatenate((first_person_array, second_person_array), axis=-1)
        first_person_array -= _np.tile(first_person_array[:, 3 * JointTypeMS1.SPINE:3 * JointTypeMS1.SHOULDER_CENTER],
                                       joints_per_skeleton)
        second_person_array -= _np.tile(second_person_array[:, 3 * JointTypeMS1.SPINE:3 * JointTypeMS1.SHOULDER_CENTER],
                                        joints_per_skeleton)
        return _np.concatenate((first_person_array, second_person_array), axis=-1)


def causal_savitzky_golay_filter(skeleton_sequence: _np.ndarray) -> _np.ndarray:
    """
    Applies the filter: X_f(0) = 0.086*X(-4) - 0.143*X(-3) - 0.086*X(-2) + 0.257*X(-1) + 0.886*X(0), to a skeleton
    sequence. Run this in the __getitem__ method of the SkeletonDataset class (as PyTorch's DataLoader will
    unsqueeze the first dimension for batching) if for training.
    """
    ss1 = _np.zeros((skeleton_sequence.shape[0], skeleton_sequence.shape[1]))
    ss2 = _np.zeros((skeleton_sequence.shape[0], skeleton_sequence.shape[1]))
    ss3 = _np.zeros((skeleton_sequence.shape[0], skeleton_sequence.shape[1]))
    ss4 = _np.zeros((skeleton_sequence.shape[0], skeleton_sequence.shape[1]))
    causal_data = skeleton_sequence[:-1]
    ss1[1:] = causal_data
    ss2[2:] = causal_data[:-1]
    ss3[3:] = causal_data[:-2]
    ss4[4:] = causal_data[:-3]
    return (0.086*ss4 - 0.143*ss3 - 0.086*ss2 + 0.257*ss1 + 0.886*skeleton_sequence).astype(_np.float32)


def joints_to_edges(unflattened_joint_array: _np.ndarray, is_from_ni: bool = False,
                    is_25_joints: bool = False) -> _np.ndarray:
    """
    Converts ordered joint array to edge features according to the paper:
        H. Wang and L. Wang, "Beyond joints: Learning representations from primitive geometries for
        skeleton-based action recognition and detection," IEEE Transactions on Image Processing,
        vol. 27, no. 9, pp. 4382-4394, 2018.
    """
    skeleton_edge_array = _np.zeros((SensorJointNumber.KINECT_V1, 3), dtype=_np.float32) if not is_25_joints \
        else _np.zeros((SensorJointNumber.KINECT_V2, 3), dtype=_np.float32)
    if not is_from_ni:
        skeleton_edge_array[0] = unflattened_joint_array[JointTypeMS1.SPINE] - \
            unflattened_joint_array[JointTypeMS1.HIP_CENTER]
        skeleton_edge_array[7] = unflattened_joint_array[JointTypeMS1.WRIST_RIGHT] - \
            unflattened_joint_array[JointTypeMS1.ELBOW_RIGHT]
        skeleton_edge_array[8] = unflattened_joint_array[JointTypeMS1.HAND_RIGHT] - \
            unflattened_joint_array[JointTypeMS1.WRIST_RIGHT]
        skeleton_edge_array[9] = unflattened_joint_array[JointTypeMS1.WRIST_LEFT] - \
            unflattened_joint_array[JointTypeMS1.ELBOW_LEFT]
        skeleton_edge_array[10] = unflattened_joint_array[JointTypeMS1.HAND_LEFT] - \
            unflattened_joint_array[JointTypeMS1.WRIST_LEFT]
        skeleton_edge_array[11] = unflattened_joint_array[JointTypeMS1.HIP_RIGHT] - \
            unflattened_joint_array[JointTypeMS1.HIP_CENTER]
        skeleton_edge_array[12] = unflattened_joint_array[JointTypeMS1.HIP_LEFT] - \
            unflattened_joint_array[JointTypeMS1.HIP_CENTER]
        skeleton_edge_array[15] = unflattened_joint_array[JointTypeMS1.ANKLE_RIGHT] - \
            unflattened_joint_array[JointTypeMS1.KNEE_RIGHT]
        skeleton_edge_array[16] = unflattened_joint_array[JointTypeMS1.FOOT_RIGHT] - \
            unflattened_joint_array[JointTypeMS1.ANKLE_RIGHT]
        skeleton_edge_array[17] = unflattened_joint_array[JointTypeMS1.ANKLE_LEFT] - \
            unflattened_joint_array[JointTypeMS1.KNEE_LEFT]
        skeleton_edge_array[18] = unflattened_joint_array[JointTypeMS1.FOOT_LEFT] - \
            unflattened_joint_array[JointTypeMS1.ANKLE_LEFT]
    else:
        skeleton_edge_array[8] = unflattened_joint_array[JointTypeMS1.HAND_RIGHT] - \
            unflattened_joint_array[JointTypeMS1.ELBOW_RIGHT]
        skeleton_edge_array[10] = unflattened_joint_array[JointTypeMS1.HAND_LEFT] - \
            unflattened_joint_array[JointTypeMS1.ELBOW_LEFT]
        skeleton_edge_array[11] = unflattened_joint_array[JointTypeMS1.HIP_RIGHT] - \
            unflattened_joint_array[JointTypeMS1.SPINE]
        skeleton_edge_array[12] = unflattened_joint_array[JointTypeMS1.HIP_LEFT] - \
            unflattened_joint_array[JointTypeMS1.SPINE]
        skeleton_edge_array[16] = unflattened_joint_array[JointTypeMS1.FOOT_RIGHT] - \
            unflattened_joint_array[JointTypeMS1.KNEE_RIGHT]
        skeleton_edge_array[18] = unflattened_joint_array[JointTypeMS1.FOOT_LEFT] - \
            unflattened_joint_array[JointTypeMS1.KNEE_LEFT]
    skeleton_edge_array[1] = unflattened_joint_array[JointTypeMS1.SHOULDER_CENTER] - \
        unflattened_joint_array[JointTypeMS1.SPINE]
    skeleton_edge_array[2] = unflattened_joint_array[JointTypeMS1.HEAD] - \
        unflattened_joint_array[JointTypeMS1.SHOULDER_CENTER]
    skeleton_edge_array[3] = unflattened_joint_array[JointTypeMS1.SHOULDER_RIGHT] - \
        unflattened_joint_array[JointTypeMS1.SHOULDER_CENTER]
    skeleton_edge_array[4] = unflattened_joint_array[JointTypeMS1.SHOULDER_LEFT] - \
        unflattened_joint_array[JointTypeMS1.SHOULDER_CENTER]
    skeleton_edge_array[5] = unflattened_joint_array[JointTypeMS1.ELBOW_RIGHT] - \
        unflattened_joint_array[JointTypeMS1.SHOULDER_RIGHT]
    skeleton_edge_array[6] = unflattened_joint_array[JointTypeMS1.ELBOW_LEFT] - \
        unflattened_joint_array[JointTypeMS1.SHOULDER_LEFT]
    skeleton_edge_array[13] = unflattened_joint_array[JointTypeMS1.KNEE_RIGHT] - \
        unflattened_joint_array[JointTypeMS1.HIP_RIGHT]
    skeleton_edge_array[14] = unflattened_joint_array[JointTypeMS1.KNEE_LEFT] - \
        unflattened_joint_array[JointTypeMS1.HIP_LEFT]
    if is_25_joints:
        skeleton_edge_array[20] = unflattened_joint_array[JointTypeMS2.HAND_TIP_RIGHT] - \
            unflattened_joint_array[JointTypeMS2.HAND_RIGHT]
        skeleton_edge_array[21] = unflattened_joint_array[JointTypeMS2.THUMB_RIGHT] - \
            unflattened_joint_array[JointTypeMS2.HAND_RIGHT]
        skeleton_edge_array[22] = unflattened_joint_array[JointTypeMS2.HAND_TIP_LEFT] - \
            unflattened_joint_array[JointTypeMS2.HAND_LEFT]
        skeleton_edge_array[23] = unflattened_joint_array[JointTypeMS2.THUMB_LEFT] - \
            unflattened_joint_array[JointTypeMS2.HAND_LEFT]
    return skeleton_edge_array


def unprocessed_collate(batch):
    """A dummy function to prevent Pytorch's data loader from converting and stacking batch data."""
    return batch    # List of data tuples (sequence, label, ...)


def get_gaussian_confidence(t, event_time, sigma):
    """Event can be either start or end of an action. Time is the frame index for actual sequences."""
    return _torch.exp(-0.5 * _torch.pow((t - event_time) / sigma, 2.))


def get_regression_matrix(label_tensor: _torch.Tensor, num_classes: int, sigma: float, forcast_t: int):
    """
    Generates target values for temporal regression of action start/end confidences, according to the paper:
        Y. Li, C. Lan, J. Xing, W. Zeng, C. Yuan, and J. Liu, "Online human action detection using joint
        classification-regression recurrent neural networks," in European Conference on Computer Vision,
        2016, pp. 203-220: Springer.

    Note that the background class ~ is at the very last index and should have all zeros for its regression target
    as its start/end is not of interest.
    """
    seq_len = len(label_tensor)
    confidence_mat = _torch.zeros(seq_len, num_classes, 2)   # assuming the 'unknown' class is already included
    last_label_idx = num_classes - 1    # denotes unknown action
    start_frames = []
    end_frames = []
    for frame_idx, label_idx in enumerate(label_tensor, 0):
        # assert label_idx < num_classes, 'Unexpected index of action class found.'
        if label_idx == num_classes - 1:    # don't record unknown action
            if last_label_idx != num_classes - 1:
                end_frames.append((last_label_idx, frame_idx))
                last_label_idx = label_idx
            continue
        if label_idx != last_label_idx:
            # start of an action
            start_frames.append((label_idx, frame_idx + forcast_t))
            if last_label_idx != num_classes - 1:     # and frame_idx != seq_len - 1
                # end of the last action
                end_frames.append((last_label_idx, frame_idx))
        elif frame_idx == seq_len - 1 and last_label_idx != num_classes - 1:
            # when the end of an action is same as the end of sequence
            end_frames.append((label_idx, frame_idx))
        last_label_idx = label_idx
    for start_frame in start_frames:
        confidence_mat[:, start_frame[0], 0] += get_gaussian_confidence(_torch.FloatTensor(range(0, seq_len)),
                                                                        start_frame[1],
                                                                        sigma)
    for end_frame in end_frames:
        confidence_mat[:, end_frame[0], 1] += get_gaussian_confidence(_torch.FloatTensor(range(0, seq_len)),
                                                                      end_frame[1],
                                                                      sigma)
    confidence_mat = _torch.clamp(confidence_mat, min=0, max=1)
    return confidence_mat


class PadCollate:
    """
    A variant of callate_fn that pads according to the longest sequence in a batch of sequences.
    This is for variable-length sequence batch training in LSTM/GRU. SRU/IndRNN not tested yet.
    """

    def __init__(self, dim=0, enable_augment=True):
        """
        :param dim: the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.enable_augment = enable_augment

    @staticmethod
    def pad_tensor(vec, pad, dim, value=0):
        """
        :param vec: tensor to pad
        :param pad: the size to pad to
        :param dim: dimension/axis to pad
        :param value: the padding value
        :return: a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return _torch.cat((vec.float(), value * _torch.ones(*pad_size)), dim=dim)

    def pad_collate(self, batch):
        """
        :param batch: list of (tensor, label)

        :returns
            xs: a tensor of all examples in 'batch' after padding
            ys: a LongTensor of all labels in batch
        """

        # find longest sequence
        lengths = list(map(lambda x: x[0].shape[self.dim], batch))
        max_len = max(lengths)
        lengths = _torch.LongTensor(lengths)
        lengths, perm_indices = lengths.sort(0, descending=True)

        # pad according to max_len
        batch = map(lambda x: (self.pad_tensor(x[0], pad=max_len, dim=self.dim),
                               self.pad_tensor(x[1], pad=max_len, dim=self.dim, value=-100),
                               self.pad_tensor(x[2], pad=max_len, dim=self.dim)), batch)
        temp1 = copy.deepcopy(batch)
        temp2 = copy.deepcopy(batch)

        # stack all and sort
        xs = _torch.stack(tuple(map(lambda x: x[0], batch)), dim=0)
        ys = _torch.stack(tuple(map(lambda x: x[1], temp1)), dim=0)
        zs = _torch.stack(tuple(map(lambda x: x[2], temp2)), dim=0)
        return xs[perm_indices], ys.long(), zs, lengths, perm_indices

    # deprecated
    def __call__(self, batch):
        """For the collate_fn argument of PyTorch's dataloader."""
        return self.pad_collate(batch)[:2]


# deprecated('Use rand_resample() to achieve either downsampling or upsampling instead.')
def rand_downsample(tensor: _torch.Tensor):
    if tensor.dim() == 3:
        tensor = tensor.squeeze()
    seq_len = tensor.shape[0]
    n_segments = random.randint(seq_len // 3, seq_len)
    downsampled = _torch.FloatTensor(n_segments, tensor.shape[1])
    seg_len = seq_len // n_segments
    for i in range(n_segments):
        frame_idx = random.randint(i * seg_len, (i + 1) * seg_len - 1)
        downsampled[i] = tensor[frame_idx]
    return downsampled


def logit_to_prob(logit):
    odd = _torch.exp(logit)
    return odd / (1 + odd)


def is_model_on_gpu(model: _torch.nn.Module):
    return next(model.parameters()).is_cuda


# deprecated('Not used in the current RNN-based method')
def seq_to_img(seq: _torch.Tensor, c_min, c_max, side_len=IMG_SIDE_LEN, use_gpu=True):
    """Encode a (trimmed) sequence into a pseudo-color image for CNN classifications."""
    if seq.dim() == 2:
        seq = seq.unsqueeze(0)
    h = seq.shape[1]
    w = seq.shape[2] // 3
    min_tensor, max_tensor = _torch.Tensor(c_min), _torch.Tensor(c_max)
    if use_gpu:
        min_tensor = min_tensor.cuda()
        max_tensor = max_tensor.cuda()
    temp = seq.view((h, w, 3))
    raw = (255 * ((temp - min_tensor) / (max_tensor - min_tensor))).floor()
    return _resize_img(raw, side_len)


# deprecated('Not used in the current RNN-based method')
def _resize_img(img: _torch.Tensor, side_lenth):
    """Resizes an image of PyTorch tensor format."""
    # assert img.dim() == 3
    return _F.interpolate(img.permute((2, 0, 1)).unsqueeze(0),
                          size=(side_lenth, side_lenth),
                          mode='bilinear',
                          align_corners=True)


# deprecated('Not used in the current RNN-based method')
def tensor_as_cv2_img(img: _torch.Tensor):
    """Converts PyTorch tensor to BGR OpenCV image."""
    return img.squeeze().permute(1, 2, 0).byte().numpy()     # OpenCV interprets image in dimensions (H, W, ch)


# deprecated('Not used in the current RNN-based method')
def sliding_window(seq: _torch.Tensor, window_size: int):
    """Returns windowed slices of a long sequence."""
    length = len(seq)
    if window_size > length:
        window_size = length
    ret = _torch.stack([seq[i:i+window_size] for i in range(length - window_size + 1)])

    # shuffle the cropped windows
    r = _torch.randperm(len(ret))
    ret = ret[r][:]
    return ret


def get_rotate_mat(alpha, beta, gamma) -> _np.ndarray:
    return transforms3d.euler.euler2mat(alpha, beta, gamma, axes='sxyz')


def get_rotate_tensor(alpha, beta, gamma) -> _torch.Tensor:
    rx = _torch.Tensor([[1, 0, 0],
                       [0, math.cos(alpha), -math.sin(alpha)],
                       [0, math.sin(alpha), math.cos(alpha)]])
    ry = _torch.Tensor([[math.cos(beta), 0, math.sin(beta)],
                       [0, 1, 0],
                       [-math.sin(beta), 0, math.cos(beta)]])
    rz = _torch.Tensor([[math.cos(gamma), -math.sin(gamma), 0],
                       [math.sin(gamma), math.cos(gamma), 0],
                       [0, 0, 1]])
    return rz.mm(ry).mm(rx)


def get_rotate_tensors(angles: _torch.Tensor) -> _torch.Tensor:
    """Produces batched rotation matrices from a batch of XYZ Euler angles."""
    seq_len = angles.shape[0]
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]

    cosz = _torch.cos(z)
    sinz = _torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = _torch.stack((cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones), dim=1).reshape((seq_len, 3, 3))

    cosy = _torch.cos(y)
    siny = _torch.sin(y)

    ymat = _torch.stack((cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy), dim=1).reshape((seq_len, 3, 3))

    cosx = _torch.cos(x)
    sinx = _torch.sin(x)

    xmat = _torch.stack((ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx), dim=1).reshape((seq_len, 3, 3))

    return zmat @ ymat @ xmat


def get_translate_mat(dx, dy, dz) -> _np.ndarray:
    return _np.array([dx, dy, dz])


def get_translate_tensor(dx, dy, dz) -> _torch.Tensor:
    return _torch.Tensor([dx, dy, dz])


def get_scale_mat(scale) -> _np.ndarray:
    return transforms3d.zooms.zfdir2mat(scale, direction=None)


def get_scale_tensor(scale) -> _torch.Tensor:
    return scale * _torch.eye(3)


def affine_transform_sequence(tensor: _torch.Tensor,
                              alpha: float = 0,
                              beta: float = 0,
                              gamma: float = 0,
                              dx: float = 0,
                              dy: float = 0,
                              dz: float = 0,
                              scale: float = 1.0,
                              reflect_yz: bool = False):
    dim = tensor.dim()
    batched = False
    if dim == 1:
        tensor = tensor.unsqueeze(0)
    elif dim == 3:
        batched = True
        tensor = tensor.squeeze(0)
    device = tensor.device
    r, s, t = get_rotate_tensor(alpha, beta, gamma).to(device), \
        get_scale_tensor(scale).to(device), \
        get_translate_tensor(dx, dy, dz).expand((tensor.shape[0], 3)).to(device)
    unflattened = tensor.view((tensor.shape[0], tensor.shape[1] // 3, 3)).transpose(1, 2).permute((2, 0, 1))
    rotated = _torch.matmul(unflattened, r)
    scaled = _torch.matmul(rotated, s)
    translated = scaled + t
    if reflect_yz:
        f = _torch.eye(3)
        f[0, 0] = -1
        translated = _torch.matmul(translated, f.to(device))
    translated = translated.permute((1, 0, 2)).contiguous()
    ret = translated.view((tensor.shape[0], tensor.shape[1]))   # flatten
    if batched:
        ret = ret.unsqueeze(0)
    return ret


def affine_transform_sequence_mat(mat: _np.ndarray,
                                  alpha: float = random.uniform(-MAX_PITCH_YAW_ANGLE, MAX_PITCH_YAW_ANGLE),
                                  beta: float = random.uniform(-MAX_PITCH_YAW_ANGLE, MAX_PITCH_YAW_ANGLE),
                                  gamma: float = random.uniform(-MAX_PITCH_YAW_ANGLE, MAX_PITCH_YAW_ANGLE),
                                  dx: float = random.uniform(-MAX_X_TRANSLATION, MAX_X_TRANSLATION),
                                  dy: float = random.uniform(-MAX_Y_TRANSLATION, MAX_Y_TRANSLATION),
                                  dz: float = random.uniform(0, MAX_Z_TRANSLATION),
                                  scale: float = random.uniform(MIN_SCALE, MAX_SCALE)):
    r, s, t = get_rotate_mat(alpha, beta, gamma), \
              get_scale_mat(scale), \
              _np.expand_dims(get_translate_mat(dx, dy, dz), 0).repeat(mat.shape[0], axis=0)
    unflattened = mat.reshape((mat.shape[0], mat.shape[1] // 3, 3)).transpose((1, 0, 2))
    rotated = _np.matmul(unflattened, r)
    scaled = _np.matmul(rotated, s)
    translated = scaled + t
    return translated.transpose((1, 0, 2)).reshape(mat.shape[0], mat.shape[1])


def rotate_framewise(sequence: _torch.Tensor,
                     angles: _torch.Tensor):
    sequence = sequence.squeeze(0)
    angles = angles.squeeze(0)
    rotations = get_rotate_tensors(angles)
    unflattened = sequence.view((sequence.shape[0], sequence.shape[1] // 3, 3)).transpose(1, 2)
    rotated = _torch.matmul(rotations, unflattened)
    return rotated.view(sequence.shape)  # flatten


def rand_resample(tensor: _torch.Tensor, lower_factor=DOWNSAMPLE_THRES, upper_factor=UPSAMPLE_THRES):
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    _, seq_len, channels = tensor.shape
    new_len = random.randint(int(seq_len * lower_factor), int(seq_len * upper_factor))
    ret = _F.interpolate(tensor.view((seq_len, channels // 3, 3)).permute((1, 2, 0)),
                         size=new_len,
                         mode='linear',
                         align_corners=True)
    ret = ret.permute((2, 0, 1)).contiguous()
    return ret.view((new_len, channels))


def world_coords_to_image(coords_3d, focal_length: float, width: int, height: int) -> tuple:
    assert len(coords_3d) == 3, 'Dimension should be 3.'
    if coords_3d[2] == 0:
        return -1, -1
    u = int(round(focal_length * coords_3d[0] / coords_3d[2] + width / 2))
    v = int(round(focal_length * (-1 * coords_3d[1]) / coords_3d[2] + height / 2))
    return u, v


def world_coords_array_to_image(joint_3d_coords: _np.ndarray, focal_length, width, height) -> _np.ndarray:
    assert joint_3d_coords.shape[1] == 3
    coords_3d = joint_3d_coords.copy()
    ret = -_np.ones((coords_3d.shape[0], 2), dtype=_np.uint16)
    non_zero_idx = (coords_3d[:, 2] != 0)
    ret[non_zero_idx, 0] = focal_length * coords_3d[non_zero_idx, 0] / coords_3d[non_zero_idx, 2] + width // 2
    ret[non_zero_idx, 1] = focal_length * (-1 * coords_3d[non_zero_idx, 1]) / coords_3d[non_zero_idx, 2] + height // 2
    return ret
