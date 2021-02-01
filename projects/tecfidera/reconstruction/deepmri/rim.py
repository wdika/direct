from itertools import chain

import torch
from torch.nn import Module, Sequential, ModuleList

from projects.deepmri.tecfidera.utils import act_fns, recurrencies, convs


def multi_gll(eta, y, mask, sense, sigma=1.):
    eta = eta.unsqueeze(0)

    re_se = eta.real * sense.real - eta.imag * sense.imag
    im_se = eta.real * sense.imag + eta.imag * sense.real

    pred = torch.fft.ifft2(mask * (torch.fft.fft2(torch.complex(re_se, im_se), dim=(-2, -1)) - y), dim=(-2, -1))

    re_out = torch.sum(pred.real * sense.real + pred.imag * sense.imag, 1) / sigma ** 2.
    im_out = torch.sum(pred.imag * sense.real - pred.real * sense.imag, 1) / sigma ** 2.

    eta = eta.squeeze(0)

    return torch.cat((eta.real, eta.imag, re_out, im_out), 0).unsqueeze(0).type(torch.float32)


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
    def __init__(self, t_max=8, act_fn='relu', bias=True, recurrent='GRU', n_feature='64',
                 conv_params={'f': [0, 0, 64, 2], 'k': [3, 1], 'd': [1, 1]},
                 rnn_params={'f': [64, 64, 0, 0], 'k': [(3, 1), (3, 1)], 'd': [(1, 1), (1, 1)]}
                 ):
        super(Rim, self).__init__()
        self.t_max = t_max
        self.act_fn = act_fns[act_fn]
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
                convstack.append(convs['2d'](channels, cf, ck, padding=padding, dilation=cd, bias=bias))
                channels = cf

            if rf != 0:
                rk, rd = next(rnn_kernels), next(rnn_dilations)
                self.convRNNs.append(
                    ConvRNNStack(convstack, recurrencies[recurrent](channels, rf, rk, dilation=rd, bias=bias),
                                 self.act_fn))
                channels = rf
                convstack = ModuleList()

        del convstack[-1]._parameters['bias']
        convstack[-1].register_parameter('bias', None)
        self.final_layer = Sequential(*list(chain(*[[c, act_fns[act_fn]] for c in convstack]))[:-1])
        self.restack = lambda x: x[:, 0, ...] + 1j * x[:, 1, ...]

        print('Constructed RIM of {:,d} parameters.'.format(self.get_num_params()))

    def forward(self, eta, y, mask, sense, t_max=None, hiddens=None):
        if t_max is None:
            t_max = self.t_max

        gll = lambda e: multi_gll(e, y, mask, sense)

        if hiddens is None:
            hiddens = [torch.zeros(eta.size(0), f, *eta.size()[1:]).to(eta.device) for f in self.rnn_params['f'] if
                       f != 0]

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

    def inference(self, eta, y, sense, mask, t_max):
        hiddens = None
        for t in range(max(1, int(self.t_max / t_max)) if t_max != 0 else 1):
            eta, hiddens = self(eta, y, mask, sense=sense, t_max=t_max, hiddens=hiddens)
            yield eta
