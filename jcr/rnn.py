# TODO solve variable sequence length problem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


"""
Note:
    Input and output formats are both (batch, seq_len, features) because of batch_first=True
    for CUDA GPU training
"""


class _JcrLSTMSubnet(nn.Module):
    def __init__(self, input_size, batch_size, num_layers, hidden_dim, dropout_ratio, use_gpu: bool):
        super(_JcrLSTMSubnet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)     # Might add batch norm
        self.dropout_ratio = dropout_ratio
        self.hidden = self.init_lstm_hidden(num_layers, batch_size, hidden_dim, use_gpu)

    @staticmethod
    def init_lstm_hidden(num_layers, batch_size, hidden_dim, use_gpu: bool):
        if use_gpu:
            h0 = Variable(torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32).cuda(0))
            c0 = Variable(torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32).cuda(0))
        else:
            h0 = Variable(torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32))
            c0 = Variable(torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32))
        return h0, c0

    def forward(self, input_data):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        out, self.hidden = self.lstm(input_data, self.hidden)
        out = F.dropout(F.relu(self.fc(out)), self.dropout_ratio, training=True)
        out = self.dp(out)
        return out


class _JcrGRUSubnet(nn.Module):
    def __init__(self, input_size, batch_size, num_layers, hidden_dim, dropout_ratio, use_gpu: bool):
        super(_JcrGRUSubnet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)     # Might add batch norm
        self.dropout_ratio = dropout_ratio
        self.hidden = self.init_gru_hidden(num_layers, batch_size, hidden_dim, use_gpu)

    @staticmethod
    def init_gru_hidden(num_layers, batch_size, hidden_dim, use_gpu: bool):
        if use_gpu:
            h0 = Variable(torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32).cuda(0))
        else:
            h0 = Variable(torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32))
        return h0

    def forward(self, input_data):
        self.hidden = self.hidden.detach()
        out, self.hidden = self.gru(input_data, self.hidden)
        out = F.dropout(F.relu(self.fc(out)), self.dropout_ratio, training=True)
        return out


class _JcrClassificationBranch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(_JcrClassificationBranch, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_classes + 1)  # M+1 neurons for M classes, considering no actions
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input_data):
        raw_out = self.fc1(input_data)
        c_out = self.softmax(raw_out)
        return raw_out, c_out


class JcrLSTM(nn.Module):
    def __init__(self, joint_dimensions, num_classes, batch_size=1,
                 subnet1_num_layers=1, subnet1_hidden_dim=100, subnet1_dropout=0.4,
                 subnet2_num_layers=1, subnet2_hidden_dim=110, subnet2_dropout=0.4,
                 subnet3_num_layers=1, subnet3_hidden_dim=100, subnet3_dropout=0.4,
                 regression_features=10, use_gpu: bool=True):
        super(JcrLSTM, self).__init__()
        self.num_classes = num_classes
        self.regression_features = regression_features
        self.subnet1 = _JcrLSTMSubnet(joint_dimensions, batch_size,
                                      subnet1_num_layers, subnet1_hidden_dim, subnet1_dropout,
                                      use_gpu)
        self.subnet2 = _JcrLSTMSubnet(subnet1_hidden_dim, batch_size,
                                      subnet2_num_layers, subnet2_hidden_dim, subnet2_dropout,
                                      use_gpu)
        self.subnet3 = _JcrLSTMSubnet(subnet2_hidden_dim, batch_size,
                                      subnet3_num_layers, subnet3_hidden_dim, subnet3_dropout,
                                      use_gpu)
        self.main_net = nn.Sequential(self.subnet1, self.subnet2, self.subnet3)
        self.classification_net = _JcrClassificationBranch(subnet3_hidden_dim, num_classes)
        self.fc2 = nn.Linear(subnet3_hidden_dim, regression_features * (num_classes + 1))
        self.fc3 = nn.Linear(regression_features, 2)

    def forward(self, input_data):
        out = self.main_net(input_data)
        c_out, s_out = self.classification_net(out)    # in dimensions (batch_size, sequence_length, num_classes + 1)
        s_out = s_out.permute(1, 2, 0)          # turn sequence length into dummy batch size at dim=0
        r_out = F.relu(self.fc2(out)[0])
        r_out = r_out.reshape(r_out.shape[0], self.num_classes + 1, self.regression_features)
        r_out = r_out * s_out   # soft-selector
        r_out = F.relu(self.fc3(r_out))
        # return the c_out FC output as cross entropy function already includes softmax
        return c_out[0], s_out[:, :, -1], r_out


class JcrGRU(nn.Module):
    def __init__(self, joint_dimensions, num_classes, batch_size=1,
                 subnet1_num_layers=1, subnet1_hidden_dim=100, subnet1_dropout=0.5,
                 subnet2_num_layers=1, subnet2_hidden_dim=110, subnet2_dropout=0.5,
                 subnet3_num_layers=1, subnet3_hidden_dim=100, subnet3_dropout=0.5,
                 regression_features=10, use_gpu: bool = True):
        super(JcrGRU, self).__init__()
        self.num_classes = num_classes
        self.regression_features = regression_features
        self.subnet1 = _JcrGRUSubnet(joint_dimensions, batch_size,
                                     subnet1_num_layers, subnet1_hidden_dim, subnet1_dropout,
                                     use_gpu)
        self.subnet2 = _JcrGRUSubnet(subnet1_hidden_dim, batch_size,
                                     subnet2_num_layers, subnet2_hidden_dim, subnet2_dropout,
                                     use_gpu)
        self.subnet3 = _JcrGRUSubnet(subnet2_hidden_dim, batch_size,
                                     subnet3_num_layers, subnet3_hidden_dim, subnet3_dropout,
                                     use_gpu)
        self.main_net = nn.Sequential(self.subnet1, self.subnet2, self.subnet3)
        self.classification_net = _JcrClassificationBranch(subnet3_hidden_dim, num_classes)
        self.fc2 = nn.Linear(subnet3_hidden_dim, regression_features * (num_classes + 1))
        self.fc3 = nn.Linear(regression_features, 2)

    def forward(self, input_data):
        out = self.main_net(input_data)
        c_out, s_out = self.classification_net(out)    # in dimensions (batch_size, sequence_length, num_classes + 1)
        s_out = s_out.permute(1, 2, 0)          # turn sequence length into dummy batch size at dim=0
        r_out = F.relu(self.fc2(out)[0])
        r_out = r_out.reshape(r_out.shape[0], self.num_classes + 1, self.regression_features)
        r_out = r_out * s_out   # soft-selector
        r_out = F.relu(self.fc3(r_out))
        # return the raw FC output as cross entropy function already includes softmax
        return c_out[0], s_out[:, :, -1], r_out
