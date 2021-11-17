"""
This module is forked from https://github.com/StefOe/indrnn-pytorch that implements:
    S. Li, W. Li, C. Cook, C. Zhu, and Y. Gao, "Independently recurrent neural network (indrnn): Building A longer and
    deeper RNN," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 5457-5466.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, ParameterList
from torch.autograd import Variable


__all__ = ['IndRNN', 'LayerNormIndRNN']


def check_bounds(weight, min_abs, max_abs):
    if min_abs:
        abs_kernel = torch.abs(weight).clamp_(min=min_abs)
        weight = torch.mul(torch.sign(weight), abs_kernel)
    if max_abs:
        weight = weight.clamp(max=max_abs, min=-max_abs)
    return weight


class IndRNN(nn.Module):
    r"""Applies a multi-layer IndRNN with `tanh` or `ReLU` non-linearity to an
    x sequence.


    For each element in the x sequence, each layer computes the following
    function:

    .. math::

        h_t = \tanh(w_{ih} x_t + b_{ih}  +  w_{hh} (*) h_{(t-1)})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is
    the hidden state of the previous layer at time `t` or :math:`x_t`
    for the first layer. (*) is element-wise multiplication.
    If :attr:`nonlinearity`='relu', then `ReLU` is used instead of `tanh`.

    Args:
        x_size: The number of expected features in the x `x`
        hidden_size: The number of features in the hidden state `h`
        n_layer: Number of recurrent layers.
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
        hidden_inits: The init value generator for the hidden unit.
        recurrent_inits: The init value generator for the recurrent unit.
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_norm: If ``True``, then batch normalization is applied after each time step
        batch_first: If ``True``, then the x and output tensors are provided
            as `(batch, seq, feature)`
        hidden_min_abs: Minimal absolute inital value for hidden weights. Default: 0
        hidden_max_abs: Maximal absolute inital value for hidden weights. Default: None

    Inputs: x, h_0
        - **x** of shape `(seq_len, batch, x_size)`: tensor containing the features
          of the x sequence. The x can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          or :func:`torch.nn.utils.rnn.pack_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, hidden_size * num_directions)`: tensor
          containing the output features (`h_k`) from the last layer of the RNN,
          for each `k`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the x, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for `k = seq_len`.

    Attributes:
        cells[k]: individual IndRNNCells containing the weights

    Examples::

        >>> rnn = IndRNN(10, 20, 2)
        >>> x = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output = rnn(x, h0)
    """

    def __init__(self, x_size, hidden_size, n_layer=1, batch_norm=False,
                 batch_first=False, bidirectional=False, bias=True,
                 hidden_inits=None, recurrent_inits=None,
                 nonlinearity='relu', hidden_min_abs=0, hidden_max_abs=None,
                 gradient_clip=None):
        super(IndRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.n_layer = n_layer
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.hidden_min_abs = hidden_min_abs
        self.hidden_max_abs = hidden_max_abs

        self.gradient_clip = gradient_clip
        if gradient_clip:
            if isinstance(gradient_clip, tuple):
                min_g, max_g = gradient_clip
            else:
                max_g = gradient_clip
                min_g = -max_g

        if self.nonlinearity == 'tanh':
            self.activation = torch.tanh
        elif self.nonlinearity == 'relu':
            self.activation = F.relu6
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        self.num_directions = num_directions = 2 if self.bidirectional else 1

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        self.cells_recurrent = ParameterList(
            [Parameter(torch.Tensor(num_directions * hidden_size)) for _ in range(n_layer)]
        )
        if gradient_clip:
            for param in self.cells_recurrent:
                param.register_hook(
                    lambda x: x.clamp(min=min_g, max=max_g)
                )

        cells_hidden = []
        for i in range(n_layer):
            in_size = x_size * num_directions if i == 0 else hidden_size * num_directions**2
            hidden = nn.Conv1d(
                in_size, hidden_size * num_directions, 1, groups=num_directions
            )
            if hidden_inits is not None:
                hidden_inits[i](hidden.weight)
            else:
                torch.nn.init.normal_(hidden.weight, 0, 0.01)
            if bias:
                torch.nn.init.constant_(hidden.bias, 0)
                if gradient_clip:
                    hidden.bias.register_hook(
                        lambda x: x.clamp(min=min_g, max=max_g)
                    )

            if recurrent_inits is not None:
                recurrent_inits[i](self.cells_recurrent[i])
            else:
                # torch.nn.init.constant_(self.cells_recurrent[i], 1)
                torch.nn.init.uniform_(self.cells_recurrent[i])

            hidden.weight.data = check_bounds(
                hidden.weight.data, self.hidden_min_abs, self.hidden_max_abs
            )
            if gradient_clip:
                hidden.weight.register_hook(
                    lambda x: x.clamp(min=min_g, max=max_g)
                )
            cells_hidden.append(hidden)

        self.cells_hidden = nn.ModuleList(cells_hidden)

        if batch_norm:
            bns = []
            for i in range(n_layer):
                bns.append(nn.BatchNorm1d(hidden_size * num_directions))
            self.bns = nn.ModuleList(bns)

        h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        self.register_buffer('h0', h0)

    def forward(self, x, hidden=None):
        frame_size = x.size(self.time_index)
        batch_size = x.size(self.batch_index)
        x = x.permute(self.batch_index, -1, self.time_index)

        hiddens = []
        i = 0
        for cell_hidden in self.cells_hidden:
            cell_hidden.weight.data = check_bounds(
                cell_hidden.weight.data,
                self.hidden_min_abs, self.hidden_max_abs
            )
            if hidden is None:
                hx = self.h0.unsqueeze(0).expand(
                    batch_size,
                    self.hidden_size * self.num_directions).contiguous()
            else:
                hx = hidden[i]

            outputs = []
            if self.bidirectional:
                x_t = torch.cat([x, x.flip(-1)], 1)
            else:
                x_t = x

            lin = cell_hidden(x_t)
            lin = torch.unbind(lin, 2)
            recurrent_h = self.cells_recurrent[i]
            for t in range(frame_size):
                hx = self.activation(lin[t] +
                                     torch.mul(recurrent_h, hx))
                outputs.append(hx)
            x = torch.stack(outputs, 2)
            hiddens.append(hx)

            if self.batch_norm:
                x = self.bns[i](x)
            i += 1
        hiddens = torch.cat(hiddens, -1)
        if self.batch_first:
            x = x.permute((0, 2, 1))
        else:
            x = x.permute((2, 0, 1))
        hiddens = Variable(hiddens.view((self.n_layer, batch_size, self.hidden_size)).clone(),
                           requires_grad=True)
        return x.squeeze(2), hiddens


class LayerNormIndRNN(nn.Module):
    r"""Applies a multi-layer layer-normalized IndRNN with `tanh` or `ReLU` non-linearity to an
    x sequence.


    For each element in the x sequence, each layer computes the following
    function:

    .. math::

        h_t = \tanh(w_{ih} x_t + b_{ih}  +  w_{hh} (*) h_{(t-1)})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is
    the hidden state of the previous layer at time `t` or :math:`x_t`
    for the first layer. (*) is element-wise multiplication.
    If :attr:`nonlinearity`='relu', then `ReLU` is used instead of `tanh`.

    Args:
        x_size: The number of expected features in the x `x`
        hidden_size: The number of features in the hidden state `h`
        n_layer: Number of recurrent layers.
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
        hidden_inits: The init value generator for the hidden unit.
        recurrent_inits: The init value generator for the recurrent unit.
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_norm: If ``True``, then batch normalization is applied after each time step
        batch_first: If ``True``, then the x and output tensors are provided
            as `(batch, seq, feature)`
        hidden_min_abs: Minimal absolute inital value for hidden weights. Default: 0
        hidden_max_abs: Maximal absolute inital value for hidden weights. Default: None

    Inputs: x, h_0
        - **x** of shape `(seq_len, batch, x_size)`: tensor containing the features
          of the x sequence. The x can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          or :func:`torch.nn.utils.rnn.pack_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, hidden_size * num_directions)`: tensor
          containing the output features (`h_k`) from the last layer of the RNN,
          for each `k`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the x, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for `k = seq_len`.

    Attributes:
        cells[k]: individual IndRNNCells containing the weights

    Examples::

        >>> rnn = LayerNormIndRNN(10, 20, 2)
        >>> x = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output = rnn(x, h0)
    """

    def __init__(self, x_size, hidden_size, n_layer=1, batch_norm=False,
                 batch_first=False, bidirectional=False, bias=True,
                 hidden_inits=None, recurrent_inits=None,
                 nonlinearity='relu', hidden_min_abs=0, hidden_max_abs=None,
                 gradient_clip=None):
        super(LayerNormIndRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.n_layer = n_layer
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.hidden_min_abs = hidden_min_abs
        self.hidden_max_abs = hidden_max_abs

        self.gradient_clip = gradient_clip
        if gradient_clip:
            if isinstance(gradient_clip, tuple):
                min_g, max_g = gradient_clip
            else:
                max_g = gradient_clip
                min_g = -max_g

        if self.nonlinearity == 'tanh':
            self.activation = torch.tanh
        elif self.nonlinearity == 'relu':
            self.activation = F.relu6
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        self.num_directions = num_directions = 2 if self.bidirectional else 1

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        self.cells_recurrent = ParameterList(
            [Parameter(torch.Tensor(num_directions * hidden_size)) for _ in range(n_layer)]
        )
        if gradient_clip:
            for param in self.cells_recurrent:
                param.register_hook(
                    lambda x: x.clamp(min=min_g, max=max_g)
                )

        cells_hidden = []
        lns = []
        for i in range(n_layer):
            in_size = x_size * num_directions if i == 0 else hidden_size * num_directions**2
            hidden = nn.Conv1d(
                in_size, hidden_size * num_directions, 1, groups=num_directions
            )
            if hidden_inits is not None:
                hidden_inits[i](hidden.weight)
            else:
                torch.nn.init.normal_(hidden.weight, 0, 0.01)
            if bias:
                torch.nn.init.constant_(hidden.bias, 0)
                if gradient_clip:
                    hidden.bias.register_hook(
                        lambda x: x.clamp(min=min_g, max=max_g)
                    )

            if recurrent_inits is not None:
                recurrent_inits[i](self.cells_recurrent[i])
            else:
                # torch.nn.init.constant_(self.cells_recurrent[i], 1)
                torch.nn.init.uniform_(self.cells_recurrent[i])

            hidden.weight.data = check_bounds(
                hidden.weight.data, self.hidden_min_abs, self.hidden_max_abs
            )
            if gradient_clip:
                hidden.weight.register_hook(
                    lambda x: x.clamp(min=min_g, max=max_g)
                )
            cells_hidden.append(hidden)
            lns.append(nn.LayerNorm(hidden_size * num_directions, eps=1e-6))

        self.cells_hidden = nn.ModuleList(cells_hidden)
        self.lns = nn.ModuleList(lns)

        if batch_norm:
            bns = []
            for i in range(n_layer):
                bns.append(nn.BatchNorm1d(hidden_size * num_directions))
            self.bns = nn.ModuleList(bns)

        h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        self.register_buffer('h0', h0)

    def forward(self, x, hidden=None):
        frame_size = x.size(self.time_index)
        batch_size = x.size(self.batch_index)
        x = x.permute((self.batch_index, -1, self.time_index))

        hiddens = []
        i = 0
        for cell_hidden in self.cells_hidden:
            cell_hidden.weight.data = check_bounds(
                cell_hidden.weight.data,
                self.hidden_min_abs, self.hidden_max_abs
            )
            if hidden is None:
                hx = self.h0.unsqueeze(0).expand(
                    batch_size,
                    self.hidden_size * self.num_directions).contiguous()
            else:
                hx = hidden[i]

            outputs = []
            if self.bidirectional:
                x_t = torch.cat([x, x.flip(-1)], 1)
            else:
                x_t = x

            lin = cell_hidden(x_t)
            lin = torch.unbind(lin, 2)
            recurrent_h = self.cells_recurrent[i]
            for t in range(frame_size):
                hx = lin[t] + torch.mul(recurrent_h, hx)
                hx = self.activation(self.lns[i](hx))
                outputs.append(hx)
            x = torch.stack(outputs, 2)
            hiddens.append(hx)

            if self.batch_norm:
                x = self.bns[i](x)
            i += 1
        hiddens = torch.cat(hiddens, -1)
        if self.batch_first:
            x = x.permute((0, 2, 1))
        else:
            x = x.permute((2, 0, 1))
        hiddens = Variable(hiddens.view((self.n_layer, batch_size, self.hidden_size)).clone(),
                           requires_grad=True)
        return x.squeeze(2), hiddens
