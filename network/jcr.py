"""Code for VA-JCR and VA-JCM architectures."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Union
from sru import SRU
from .layer_norm_rnn import LayerNormLSTM, LayerNormGRU
from .indrnn import IndRNN, LayerNormIndRNN
from .misc_layers import GaussianNoise
from utils.processing import rotate_framewise
from global_configs import RNNType, ActivationType


__all__ = ['JCR', 'JCM']


def init_rnn_hidden(is_lstm: bool, num_layers: int, batch_size: int, hidden_dim: int, device: torch.device):
    """Initializes the hidden state for a recurrent layer to store."""
    if is_lstm:
        h0 = Variable(torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32).to(device))
        c0 = Variable(torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32).to(device))
        return h0, c0
    # Below is for GRU or SRU
    h0 = Variable(torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32).to(device))
    return h0


def detach_hidden_inplace(hidden, is_lstm):
    """Truncates the computation graph."""
    if is_lstm:
        hidden[0].detach_()
        hidden[1].detach_()
    else:
        hidden.detach_()


class IndRNNBlock(nn.Module):
    """
    Neural network module with a weight (FC) layer before each IndRNN recurrent layer to form a complete IndRNN
    block.
    """
    def __init__(self, input_size, rnn_hidden_dim, num_rnn_layers, layer_norm=False):
        super(IndRNNBlock, self).__init__()
        if layer_norm:
            self.rnn_layer = LayerNormIndRNN(input_size, rnn_hidden_dim, num_rnn_layers, batch_first=False,
                                             nonlinearity='relu')
        else:
            self.rnn_layer = IndRNN(input_size, rnn_hidden_dim, num_rnn_layers, batch_first=False, nonlinearity='relu')
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x, hidden):
        x = self.fc(x)
        return self.rnn_layer(x, hidden)


class RNNSubnet(nn.Module):
    """
    Neural network module with one recurrent layer/block and one (nonlinear) FC layer after.
    """
    def __init__(self, input_size: int, rnn_type: RNNType,
                 rnn_hidden_dim: int, num_rnn_layers: int,
                 fc_hidden_dim: int, fc_dropout: float,
                 fc_activation, device: torch.device,
                 layer_norm: bool = True,
                 input_normalized: bool = False):
        super(RNNSubnet, self).__init__()
        self.is_lstm = (rnn_type == RNNType.LSTM)
        if rnn_type == RNNType.LSTM:
            if layer_norm:
                self.rnn = LayerNormLSTM(input_size, rnn_hidden_dim, num_rnn_layers)
            else:
                self.rnn = nn.LSTM(input_size, rnn_hidden_dim, num_rnn_layers, batch_first=False)
        elif rnn_type == RNNType.GRU:
            if layer_norm:
                self.rnn = LayerNormGRU(input_size, rnn_hidden_dim, num_rnn_layers)
            else:
                self.rnn = nn.GRU(input_size, rnn_hidden_dim, num_rnn_layers, batch_first=False)
        elif rnn_type == RNNType.SRU:
            num_rnn_layers *= 2
            self.rnn = SRU(input_size, rnn_hidden_dim, num_rnn_layers,
                           use_tanh=True,
                           layer_norm=layer_norm,
                           dropout=0.1,
                           is_input_normalized=input_normalized)
        else:
            self.rnn = IndRNNBlock(input_size, rnn_hidden_dim, num_rnn_layers, layer_norm=layer_norm)
        self.hidden = init_rnn_hidden(self.is_lstm, num_rnn_layers, 1, rnn_hidden_dim, device)
        self.fc = nn.Linear(rnn_hidden_dim, fc_hidden_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.activation_func = fc_activation
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)
        # nn.utils.weight_norm(self.fc, 'weight')
        # self.noise = GaussianNoise()    # add random noise if self.training

    def forward(self, x: torch.Tensor):
        detach_hidden_inplace(self.hidden, self.is_lstm)
        # x = self.noise(x)
        y, self.hidden = self.rnn(x, self.hidden)
        # y = self.noise(y)
        y = self.dropout(self.activation_func(self.fc(y), inplace=True))
        return y


class VASubnet(nn.Module):
    """
    Neural network module that learns Euler angles of the optimal viewpoint.
    """
    def __init__(self, input_size: int, rnn_type: RNNType,
                 rnn_hidden_dim: int, num_rnn_layers: int,
                 fc_dropout: float, device: torch.device,
                 layer_norm: bool = True):
        super(VASubnet, self).__init__()
        self.is_lstm = (rnn_type == RNNType.LSTM)
        if rnn_type == RNNType.LSTM:
            if layer_norm:
                self.rnn = LayerNormLSTM(input_size, rnn_hidden_dim, num_rnn_layers)
            else:
                self.rnn = nn.LSTM(input_size, rnn_hidden_dim, num_rnn_layers, batch_first=False)
        elif rnn_type == RNNType.GRU:
            if layer_norm:
                self.rnn = LayerNormGRU(input_size, rnn_hidden_dim, num_rnn_layers)
            else:
                self.rnn = nn.GRU(input_size, rnn_hidden_dim, num_rnn_layers, batch_first=False)
        elif rnn_type == RNNType.SRU:
            num_rnn_layers *= 2
            self.rnn = SRU(input_size, rnn_hidden_dim, num_rnn_layers,
                           use_tanh=True,
                           layer_norm=layer_norm,
                           dropout=0.1)
        else:
            # self.rnn = IndRNN(input_size, rnn_hidden_dim, num_rnn_layers, batch_first=False, nonlinearity='relu')
            self.rnn = IndRNNBlock(input_size, rnn_hidden_dim, num_rnn_layers)
        self.hidden = init_rnn_hidden(self.is_lstm, num_rnn_layers, 1, rnn_hidden_dim, device)
        self.fc = nn.Linear(rnn_hidden_dim, 3)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc.weight.data.fill_(0)
        self.fc.bias.data.fill_(0)
        # nn.utils.weight_norm(self.fc, 'weight')

    def forward(self, x):
        detach_hidden_inplace(self.hidden, self.is_lstm)
        x = x.permute((1, 0, 2)).contiguous()
        y, self.hidden = self.rnn(x, self.hidden)
        y = self.dropout(self.fc(y))
        return y.permute((1, 0, 2))


def set_subnet_weight_norm(subnet: Union[RNNSubnet, VASubnet]):
    nn.utils.weight_norm(subnet.rnn, 'weight_hh_l0')
    nn.utils.weight_norm(subnet.rnn, 'weight_ih_l0')


class JCR(nn.Module):
    """
    Implements the VA-JCR architecture inspired by the following papers:
        - Y. Li, C. Lan, J. Xing, W. Zeng, C. Yuan, and J. Liu, "Online human action detection using joint
            classification-regression recurrent neural networks," in European Conference on Computer Vision,
            2016, pp. 203-220: Springer.

        - P. Zhang, C. Lan, J. Xing, W. Zeng, J. Xue, and N. Zheng, "View Adaptive Neural Networks for High
            Performance Skeleton-based Human Action Recognition," arXiv preprint arXiv:1804.07453, 2018.
    """
    def __init__(self, joint_dimensions, num_classes,
                 rnn_type: RNNType = RNNType.GRU, activation_type: ActivationType = ActivationType.SELU,
                 rnn_hidden_dim1: int = 100, rnnfc_hidden_dim1: int = 100, subnet1_dropout: float = 0.25,
                 rnn_hidden_dim2: int = 110, rnnfc_hidden_dim2: int = 110, subnet2_dropout: float = 0.25,
                 rnn_hidden_dim3: int = 100, rnnfc_hidden_dim3: int = 100, subnet3_dropout: float = 0.25,
                 view_adaptive: bool = True, view_adaptive_rnn_dim: int = 100, view_adaptive_dropout: float = 0.25,
                 enable_regression: bool = True, regression_features: int = 10, layer_norm: bool = True,
                 enable_augmentation: bool = True, device=torch.device('cuda:0')):
        super(JCR, self).__init__()
        self.aug = enable_augmentation
        self.rnn_type = rnn_type
        if activation_type == ActivationType.ReLU:
            activation_func = F.relu
        elif activation_type == ActivationType.ELU:
            activation_func = F.elu
        elif activation_type == ActivationType.SELU:
            activation_func = F.selu
        else:
            raise NotImplementedError
        self.activation_func = activation_func
        self.num_classes = num_classes
        self.regression_features = regression_features if enable_regression else 0
        self.enable_va = view_adaptive
        self.va_net = VASubnet(joint_dimensions, rnn_type, view_adaptive_rnn_dim, 1,
                               view_adaptive_dropout, device,
                               layer_norm=layer_norm)
        self.subnet1 = RNNSubnet(joint_dimensions, rnn_type, rnn_hidden_dim1, 1, rnnfc_hidden_dim1,
                                 subnet1_dropout, activation_func, device,
                                 layer_norm=layer_norm)
        self.subnet2 = RNNSubnet(rnnfc_hidden_dim1, rnn_type, rnn_hidden_dim2, 1, rnnfc_hidden_dim2,
                                 subnet2_dropout, activation_func, device,
                                 layer_norm=layer_norm,
                                 input_normalized=layer_norm)
        self.subnet3 = RNNSubnet(rnnfc_hidden_dim2, rnn_type, rnn_hidden_dim3, 1, rnnfc_hidden_dim3,
                                 subnet3_dropout, activation_func, device,
                                 layer_norm=layer_norm,
                                 input_normalized=layer_norm)
        self.main_net = nn.Sequential(self.subnet1, self.subnet2, self.subnet3)
        self.fc1 = nn.Linear(rnnfc_hidden_dim3, num_classes)
        self.fc2 = nn.Linear(rnnfc_hidden_dim3, regression_features * num_classes)
        self.fc3 = nn.Linear(regression_features, 2)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)
        nn.utils.weight_norm(self.fc1, 'weight')
        # nn.utils.weight_norm(self.fc2, 'weight')
        # nn.utils.weight_norm(self.fc3, 'weight')
        # self.noise = GaussianNoise(stddev=0.15)    # add random noise if self.training

    # deprecated
    def set_rnn_weight_norm(self):
        """Enables weight normalization for all recurrent layers. Not recommended."""
        if self.rnn_type == RNNType.LSTM or self.rnn_type == RNNType.GRU:
            set_subnet_weight_norm(self.subnet1)
            set_subnet_weight_norm(self.subnet2)
            set_subnet_weight_norm(self.subnet3)
            set_subnet_weight_norm(self.va_net)

    def forward(self, x: torch.Tensor):
        """
        :param x: in shape (N, L, C), i.e. batch first
        """
        if self.aug:
            # x = self.noise(x)
            pass
        if self.enable_va:
            angles = self.va_net(x)
            x = rotate_framewise(x, angles).unsqueeze(0)
        y = self.main_net(x.permute((1, 0, 2)).contiguous()).permute((1, 0, 2))
        yl = self.fc1(y)
        ys = F.softmax(yl, dim=-1).permute((1, 2, 0))
        if self.regression_features > 0:
            r = F.relu(self.fc2(y[0]), inplace=True)
            r = r.reshape(r.shape[0], self.num_classes, self.regression_features)
            r = r * ys  # soft-selector
            r = F.relu(self.fc3(r), inplace=True)
        else:
            r = x.new_zeros((x.shape[1], self.num_classes, 2))
        return yl[0], ys[..., -1], r


class JCM(nn.Module):
    """
        Implements the VA-JCM architecture inspired by the following papers:
            - Y. Li, C. Lan, J. Xing, W. Zeng, C. Yuan, and J. Liu, "Online human action detection using joint
                classification-regression recurrent neural networks," in European Conference on Computer Vision,
                2016, pp. 203-220: Springer.

            - P. Zhang, C. Lan, J. Xing, W. Zeng, J. Xue, and N. Zheng, "View Adaptive Neural Networks for High
                Performance Skeleton-based Human Action Recognition," arXiv preprint arXiv:1804.07453, 2018.

            - H. Wang and L. Wang, "Learning content and style: Joint action recognition and person identification
                from human skeletons," Pattern Recognition, vol. 81, pp. 23-35, 2018.
    """
    def __init__(self, joint_dimensions, num_action_classes, num_subject_classes, num_age_classes,
                 rnn_type: RNNType = RNNType.GRU, activation_type: ActivationType = ActivationType.SELU,
                 rnn_hidden_dim1: int = 100, rnnfc_hidden_dim1: int = 100, subnet1_dropout: float = 0.25,
                 rnn_hidden_dim2: int = 110, rnnfc_hidden_dim2: int = 110, subnet2_dropout: float = 0.25,
                 rnn_hidden_dim3: int = 100, rnnfc_hidden_dim3: int = 100, subnet3_dropout: float = 0.25,
                 view_adaptive: bool = True, view_adaptive_rnn_dim: int = 100, view_adaptive_dropout: float = 0.25,
                 layer_norm: bool = True, enable_augmentation: bool = True, device=torch.device('cuda:0')):
        super(JCM, self).__init__()
        self.aug = enable_augmentation
        self.rnn_type = rnn_type
        if activation_type == ActivationType.ReLU:
            activation_func = F.relu
        elif activation_type == ActivationType.ELU:
            activation_func = F.elu
        elif activation_type == ActivationType.SELU:
            activation_func = F.selu
        else:
            raise NotImplementedError
        self.activation_func = activation_func
        self.num_classes = num_action_classes
        self.enable_va = view_adaptive
        self.va_net = VASubnet(joint_dimensions, rnn_type, view_adaptive_rnn_dim, 1,
                               view_adaptive_dropout, device,
                               layer_norm=layer_norm)
        self.subnet1 = RNNSubnet(joint_dimensions, rnn_type, rnn_hidden_dim1, 1, rnnfc_hidden_dim1,
                                 subnet1_dropout, activation_func, device,
                                 layer_norm=layer_norm)
        self.subnet2 = RNNSubnet(rnnfc_hidden_dim1, rnn_type, rnn_hidden_dim2, 1, rnnfc_hidden_dim2,
                                 subnet2_dropout, activation_func, device,
                                 layer_norm=layer_norm,
                                 input_normalized=layer_norm)
        self.subnet3 = RNNSubnet(rnnfc_hidden_dim2, rnn_type, rnn_hidden_dim3, 1, rnnfc_hidden_dim3,
                                 subnet3_dropout, activation_func, device,
                                 layer_norm=layer_norm,
                                 input_normalized=layer_norm)
        self.main_net = nn.Sequential(self.subnet1, self.subnet2, self.subnet3)
        self.fc1 = nn.Linear(rnnfc_hidden_dim3, num_action_classes)
        self.fc2 = nn.Linear(rnnfc_hidden_dim3, num_subject_classes)
        self.fc3 = nn.Linear(rnnfc_hidden_dim3, num_age_classes)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)
        nn.utils.weight_norm(self.fc1, 'weight')
        nn.utils.weight_norm(self.fc2, 'weight')
        nn.utils.weight_norm(self.fc3, 'weight')
        # self.noise = GaussianNoise(stddev=0.15)    # add random noise if self.training

    # deprecated
    def set_rnn_weight_norm(self):
        """Enables weight normalization for all recurrent layers. Not recommended."""
        if self.rnn_type == RNNType.LSTM or self.rnn_type == RNNType.GRU:
            set_subnet_weight_norm(self.subnet1)
            set_subnet_weight_norm(self.subnet2)
            set_subnet_weight_norm(self.subnet3)
            set_subnet_weight_norm(self.va_net)

    def forward(self, x: torch.Tensor):
        """
        :param x: in shape (N, L, C), i.e. batch first.
        """
        if self.aug:
            # x = self.noise(x)
            pass
        if self.enable_va:
            angles = self.va_net(x)
            x = rotate_framewise(x, angles).unsqueeze(0)
        y = self.main_net(x.permute((1, 0, 2)).contiguous()).permute((1, 0, 2))
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y1s = F.softmax(y1, dim=-1)
        y2s = F.softmax(y2, dim=-1)
        y3s = F.softmax(y3, dim=-1)
        return (y1[0], y2[0], y3[0]), (y1s[0], y2s[0], y3s[0])
