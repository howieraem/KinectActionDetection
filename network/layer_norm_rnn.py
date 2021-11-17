"""
Implementation of layer-normalized LSTM/GRU.

Some code is forked from https://gist.github.com/andrewliao11/32beb021cb2813dab7e8a3a08c78f21d.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNCellBase


__all__ = ['LayerNormLSTM', 'LayerNormGRU']


def _layer_norm_lstm(x, hidden, w_ih, w_hh, ln, b_ih=None, b_hh=None):
    """Mathematical operations in a layer-normalized LSTM cell."""
    hx, cx = hidden
    gates = F.linear(x, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

    # use layer norm here
    input_gate = torch.sigmoid(ln['input_gate'](input_gate))
    forget_gate = torch.sigmoid(ln['forget_gate'](forget_gate))
    cell_gate = torch.tanh(ln['cell_gate'](cell_gate))
    output_gate = torch.sigmoid(ln['output_gate'](output_gate))

    # output math
    cy = (forget_gate * cx) + (input_gate * cell_gate)
    hy = output_gate * torch.tanh(ln['cy'](cy))
    return hy, cy


def _layer_norm_gru(x, hx, w_ih, w_hh, ln, b_ih=None, b_hh=None):
    """Mathematical operations in a layer-normalized GRU cell."""
    ih = F.linear(x, w_ih, b_ih).chunk(3, 1)
    hh = F.linear(hx, w_hh, b_hh).chunk(3, 1)
    update_gate = ih[0] + hh[0]
    reset_gate = ih[1] + hh[1]
    ni, nh = ih[2], hh[2]

    # use layer norm here
    update_gate = torch.sigmoid(ln['update_gate'](update_gate))     # z(t)
    reset_gate = torch.sigmoid(ln['reset_gate'](reset_gate))        # r(t)
    ni = ln['x_gate'](ni)
    nh = ln['h_gate'](nh)

    # output math
    hy = torch.tanh(ni + reset_gate * nh) * (1 - update_gate) + hx * update_gate
    return hy


# initialize as backend
torch.nn.backends.thnn._get_thnn_function_backend().register_function('LayerNormLSTMCell', _layer_norm_lstm)
torch.nn.backends.thnn._get_thnn_function_backend().register_function('LayerNormGRUCell', _layer_norm_gru)


class LayerNormLSTMCell(RNNCellBase):
    """Wrapper of the mathematical function _layer_norm_lstm(), with parameters stored."""
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormLSTMCell, self).__init__(input_size, hidden_size, bias, 4)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

        self.ln_input_gate = nn.LayerNorm(hidden_size)
        self.ln_forget_gate = nn.LayerNorm(hidden_size)
        self.ln_cell_gate = nn.LayerNorm(hidden_size)
        self.ln_output_gate = nn.LayerNorm(hidden_size)
        self.ln_cy = nn.LayerNorm(hidden_size)
        self.ln = {
            'input_gate': self.ln_input_gate,
            'forget_gate': self.ln_forget_gate,
            'cell_gate': self.ln_cell_gate,
            'output_gate': self.ln_output_gate,
            'cy': self.ln_cy
        }

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hx):
        self.check_forward_input(x)
        self.check_forward_hidden(x, hx[0], '[0]')
        self.check_forward_hidden(x, hx[1], '[1]')
        return self._backend.LayerNormLSTMCell(
            x, hx,
            self.weight_ih, self.weight_hh, self.ln,
            self.bias_ih, self.bias_hh,
        )


class LayerNormLSTM(nn.Module):
    """A layer of layer-normalized LSTM cells which captures hidden states of cells at every time step."""
    # TODO: multi-layer with PyTorch's ModuleList
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(LayerNormLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = LayerNormLSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

    def forward(self, x, hidden=None):
        seq_len, batch_size, _ = x.shape
        if hidden is None:
            hx = x.new_zeros(1, batch_size, self.hidden_size, requires_grad=False)
            cx = x.new_zeros(1, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        ht = [None, ] * seq_len
        ct = [None, ] * seq_len
        h, c = hx.view((1, self.hidden_size)), cx.view((1, self.hidden_size))
        for t in range(seq_len):
            ht[t], ct[t] = self.rnn_cell(x[t], (h, c))
            h, c = ht[t], ct[t]
        y = torch.stack([h[-1] for h in ht])
        hy = ht[-1]
        cy = ct[-1]
        return y.unsqueeze(1), (hy, cy)


class LayerNormGRUCell(RNNCellBase):
    """Wrapper of the mathematical function _layer_norm_gru(), with parameters stored."""
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias, 3)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

        self.ln_update_gate = nn.LayerNorm(hidden_size)
        self.ln_reset_gate = nn.LayerNorm(hidden_size)
        self.ln_x_gate = nn.LayerNorm(hidden_size)
        self.ln_h_gate = nn.LayerNorm(hidden_size)
        self.ln = {
            'update_gate': self.ln_update_gate,
            'reset_gate': self.ln_reset_gate,
            'x_gate': self.ln_x_gate,
            'h_gate': self.ln_h_gate
        }

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hx):
        self.check_forward_input(x)
        self.check_forward_hidden(x, hx, '')
        return self._backend.LayerNormGRUCell(
            x, hx,
            self.weight_ih, self.weight_hh, self.ln,
            self.bias_ih, self.bias_hh,
        )


class LayerNormGRU(nn.Module):
    """A layer of layer-normalized GRU cells which captures hidden states of cells at every time step."""
    # TODO: multi-layer with PyTorch's ModuleList
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(LayerNormGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = LayerNormGRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

    def forward(self, x, hx=None):
        seq_len, batch_size, _ = x.shape
        if hx is None:
            hx = x.new_zeros(1, batch_size, self.hidden_size, requires_grad=False)
        ht = [None, ] * seq_len
        h = hx.view((1, self.hidden_size))
        for t in range(seq_len):
            ht[t] = self.rnn_cell(x[t], h)
            h = ht[t]
        y = torch.stack([h[-1] for h in ht])
        hy = ht[-1]
        return y.unsqueeze(1), hy
