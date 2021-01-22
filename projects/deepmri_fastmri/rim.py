from itertools import chain

import torch
from torch.nn import Module, Parameter, Sequential, ModuleList

from projects.deepmri_fastmri.utils import act_fns, loss_fns, recurrencies, convs


def multi_gll(eta, y, mask, sense, sigma=1.):
    re_sense, im_sense = sense.chunk(2, -1)
    re_eta, im_eta = map(lambda x: torch.unsqueeze(x, 1), eta.chunk(2, -1))
    re_se = re_eta * re_sense - im_eta * im_sense
    im_se = re_eta * im_sense + im_eta * re_sense
    sensed_e = torch.fft(torch.cat((re_se, im_se), -1), 2) - y
    sensed_e = mask.unsqueeze(1).unsqueeze(-1) * sensed_e
    sensed_e = torch.ifft(sensed_e, 2)
    re_sensed, im_sensed = sensed_e.chunk(2, -1)
    re_out = torch.sum(re_sensed * re_sense + im_sensed * im_sense, 1).permute(0, 3, 1, 2) / sigma ** 2.
    im_out = torch.sum(im_sensed * re_sense - re_sensed * im_sense, 1).permute(0, 3, 1, 2) / sigma ** 2.
    eta = eta.permute(0, 3, 1, 2)
    return torch.cat((eta, re_out, im_out), 1)


class ConvRNNStack(Module):
    def __init__(self, convs, rnn, act_fn):
        super(ConvRNNStack, self).__init__()
        self.act_fn = act_fn
        self.convs = convs
        self.rnn = rnn

    def forward(self, x, hidden):
        for conv in self.convs:
            x = self.act_fn(conv(x))
        return self.rnn(x, hidden)


class Rim(Module):
    def __init__(self, t_max=8, act_fn='relu', n_feature=[64], bias=True, recurrent='GRU', loss_fn='mse',
                 train_hidden=False, n_bins=30, reg_param=.000057143,
                 conv_params={'f': [0, 0, 64, 2], 'k': [3, 1], 'd': [1, 1]},
                 rnn_params={'f': [64, 64, 0, 0], 'k': [(3, 1), (3, 1)], 'd': [(1, 1), (1, 1)]}
                 ):
        super(Rim, self).__init__()
        n_feature = n_feature[0]
        self.n_feature = n_feature
        self.t_max = t_max
        self.act_fn = act_fns[act_fn]
        self.criterion = loss_fns[loss_fn](n_bins, reg_param)

        recur = recurrencies[recurrent]
        self.recur_type = recurrent

        assert len(conv_params['f']) == len(rnn_params['f'])
        assert (conv_params['f'][-1] == 2 and rnn_params['f'][-1] == 0) or rnn_params['f'] == 2
        self.conv_params = conv_params
        self.rnn_params = rnn_params
        channels = 4
        conv_kernels, rnn_kernels = iter(conv_params['k']), iter(rnn_params['k'])
        conv_dilations, rnn_dilations = iter(conv_params['d']), iter(rnn_params['d'])
        self.convRNNs = ModuleList()

        convstack = ModuleList()
        for i, (cf, rf) in enumerate(zip(conv_params['f'], rnn_params['f'])):
            if cf != 0:
                ck, cd = next(conv_kernels), next(conv_dilations)
                padding = int((ck + (ck - 1) * (cd - 1)) / 2)
                convstack.append(convs['2d'](
                    channels, cf, ck, padding=padding, dilation=cd, bias=bias))
                channels = cf
            if rf != 0:
                rk, rd = next(rnn_kernels), next(rnn_dilations)
                if recurrent == 'IndRNN':
                    self.convRNNs.append(ConvRNNStack(convstack, recur(
                        channels, rf, rk, dilation=rd, bias=bias), self.act_fn))
                else:
                    self.convRNNs.append(ConvRNNStack(convstack, recur(
                        channels, rf, rk, dilation=rd, bias=bias), self.act_fn))
                channels = rf
                convstack = ModuleList()

        if convstack:
            del convstack[-1]._parameters['bias']
            convstack[-1].register_parameter('bias', None)
            to_seq = list(chain(*[[c, act_fns[act_fn]] for c in convstack]))
            to_seq = to_seq[:-1]
            self.final_layer = Sequential(*to_seq)
        else:
            self.final_layer = lambda x: x
        self.restack = lambda x: torch.stack((x[:, 0, ...], x[:, 1, ...]), -1)

        self.fsc_weights = []

        self.train_hidden = train_hidden
        if self.train_hidden:
            self.hidden1 = Parameter(torch.zeros(self.n_feature))
            self.hidden2 = Parameter(torch.zeros(self.n_feature))

        print('Constructed RIM of {:,d} parameters.'.format(self.get_num_params()))

    def forward(self, eta, y, mask, sense, t_max=None, hiddens=None):
        if t_max is None:
            t_max = self.t_max

        gll = lambda e: multi_gll(e, y, mask, sense)

        if hiddens is None:
            hiddens = [torch.zeros(eta.size(0), f, *eta.size()[1:-1]).to(eta) for f in self.rnn_params['f'] if f != 0]

        for _ in range(t_max):
            d_eta = gll(eta).contiguous()
            for h, convrnn in enumerate(self.convRNNs):
                hiddens[h] = convrnn(d_eta, hiddens[h])
                d_eta = hiddens[h]
            eta = eta + self.restack(self.final_layer(d_eta))

        return eta, None

    def get_num_params(self):
        n = 0
        for na, p in self.named_parameters():
            n += p.numel()
        return n

    def inference(self, batch, t_max, n_batches=1):
        param = next(self.parameters())
        sbatches = zip(
            *[batch[k].chunk(min(n_batches, batch['eta'].size(0)), 0) for k in ['eta', 'y', 'mask', 'target', 'sense']])
        for args in sbatches:
            eta, y, mask, target, sense = args

            sense = sense.to(param)
            eta = eta.to(param)
            y = y.to(param)
            mask = mask.to(param)
            hiddens = None

            for t in range(max(1, int(self.t_max / t_max)) if t_max != 0 else 1):
                eta, hiddens = self(eta, y, mask, sense=sense, t_max=t_max, hiddens=hiddens)
                yield eta
