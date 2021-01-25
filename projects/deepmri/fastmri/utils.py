import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import GRUCell
from torch.nn import Module, Parameter, ReLU, Conv2d, Conv3d
from torch.nn.init import xavier_uniform_, calculate_gain, normal_, constant_


def complex_layer_norm(input, gamma, bias=None, eps=1e-5):
    re_in, im_in = input.chunk(2, 1)
    gamma = gamma.expand_as(re_in)
    re_mean, im_mean = (re_in.mean(1).expand_as(re_in),
                        im_in.mean(1).expand_as(im_in))
    re_diff, im_diff = re_in.sub(re_mean), im_in.sub(im_mean)
    std = re_diff.pow(2.).add(im_diff.pow(2.)).mean(1).sqrt().expand_as(re_input)
    re_out = re_diff.mul(gamma).div(std.add(eps))
    im_out = im_diff.mul(gamma).div(std.add(eps))
    out = torch.stack((re_out, im_out), 1)
    if bias is not None:
        out = out + bias.expand_as(out)
    return out


def layer_norm(input, gamma, bias=None, eps=1e-5):
    gamma = gamma.expand_as(input)
    mean = input.mean(1).expand_as(input)
    diff = input.sub(mean)
    std = diff.pow(2.).mean(1).add(eps).sqrt().expand_as(input)
    out = diff.mul(gamma).div(std)
    if bias is not None:
        out = out + bias.expand_as(out)
    return out


class LayerNorm(Module):
    def __init__(self, num_features, bias=True, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.gamma = Parameter(torch.ones(
            num_features).unsqueeze_(0))
        if bias:
            self.bias = Parameter(torch.zeros(
                num_features).unsqueeze_(0))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return layer_norm(
            input, self.gamma, bias=self.bias, eps=self.eps)


class LNGRUCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LNGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i_norm = LayerNorm(2 * hidden_size, bias=False)
        self.h_norm = LayerNorm(2 * hidden_size, bias=bias)
        self.u_norm = LayerNorm(hidden_size, bias=bias)
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight_ih = Parameter(torch.Tensor(
            3 * hidden_size, input_size).uniform_(-stdv, stdv))
        self.weight_hh = Parameter(torch.Tensor(
            3 * hidden_size, hidden_size).uniform_(-stdv, stdv))

    def forward(self, input, hidden):
        gi = F.linear(input, self.weight_ih)
        gh = F.linear(hidden, self.weight_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        i_r, i_z = self.i_norm(torch.cat((i_r, i_z), 1)).chunk(2, 1)
        h_r, h_z = self.h_norm(torch.cat((h_r, h_z), 1)).chunk(2, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_z + h_z)
        newgate = torch.tanh(self.u_norm(i_n + resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)

        return hy


class ComplexMSE(Function):
    def __init__(self, re_target, im_target):
        super(ComplexMSE, self).__init__()
        self.re_target = re_target
        self.im_target = im_target
        self.norm_const = re_target.numel()

    def forward(self, re_z, im_z):
        self.save_for_backward(re_z, im_z)
        loss = torch.pow(re_z - self.re_target, 2) \
               + torch.pow(im_z - self.im_target, 2)
        loss = torch.sum(loss) / self.norm_const
        return torch.Tensor([loss]).cuda()

    def backward(self, grad):
        re_z, im_z = self.saved_tensors
        re_grad = grad[0] * (re_z - self.re_target) / self.norm_const
        im_grad = grad[0] * (im_z - self.im_target) / self.norm_const
        return re_grad.cuda(), im_grad.cuda()


def mse(z, target):
    loss = torch.pow(z - target, 2.)
    sum_ = torch.sum(loss)
    return sum_ * (2 / loss.numel())


def complex_rmse(z, target):
    loss = torch.mean(torch.sqrt(torch.sum(torch.pow(
        z - target, 2.), -1) + 1e-6))
    return loss


def rmse(z, target):
    loss = torch.sqrt(torch.mean(torch.pow(z - target, 2.)) + 1e-8)
    return loss


def scaler(re_z, im_z, const):
    return const.mul(re_z), const.mul(im_z)


def mul(re_z, im_z, re_const, im_const, conj):
    re_out = re_z.mul(re_const)
    im_out = im_z.mul(re_const)
    if conj:
        re_out = re_out.add(im_z.mul(im_const))
        im_out = im_out.sub(re_z.mul(im_const))
    else:
        re_out = re_out.sub(im_z.mul(im_const))
        im_out = im_out.add(re_z.mul(im_const))
    return re_out, im_out


def absolute_loss(re_z, im_z, re_target, im_target, eps=1e-7):
    abs_z = torch.sqrt(re_z.pow(2.).add(im_z.pow(2.)).add(eps))
    abs_t = torch.sqrt(re_target.pow(2.).add(im_target.pow(2.)).add(eps))
    numel = abs_t.numel()
    return torch.sum(abs_z.sub(abs_t).pow(2.)) / numel


class FourierShellCorrelation(Module):
    def __init__(self, shape_or_img, max_r=None, n_bins=None,
                 device='cuda', shell_weights=False):
        super(FourierShellCorrelation, self).__init__()
        if isinstance(shape_or_img, (tuple, list)):
            shape = shape_or_img
        else:
            if isinstance(shape_or_img, np.ndarray):
                shape = shape_or_img.shape
            else:
                shape = shape_or_img.size()
            if len(shape) == 4:
                shape = shape[1:-1]
        frame = np.amax(shape)
        if max_r is None: max_r = (1. + np.amin(shape) / np.amax(shape)) / 2.
        if n_bins is None: n_bins = np.round(np.mean(shape)) - 1
        afocal, bfocal = shape[0] / 2, shape[1] / 2
        xx, yy = np.mgrid[:shape[0], :shape[1]]
        ellipse = np.power((xx - afocal) * 2 / frame,
                           2) + np.power((yy - bfocal) * 2 / frame, 2)
        ellipses = [torch.tensor(
            (ellipse < r_2d).astype(np.bool), dtype=torch.bool, device=device
        ) for r_2d in np.power(np.linspace(0, max_r, int(n_bins) + 1)[1:], 2.)]
        self.shells = [ellipses[0]] + [e2 ^ e1 for e1, e2 in zip(ellipses[:-1], ellipses[1:])]
        if shell_weights:
            self.weights = torch.linspace(.7, 1., len(self.shells), device=device)
        else:
            self.weights = None

    def forward(self, z_, t_):
        if isinstance(t_, np.ndarray):
            target = torch.stack((torch.tensor(t_.real, dtype=torch.float32, device=self.shells[0].device),
                                  torch.tensor(t_.imag, dtype=torch.float32, device=self.shells[0].device)), -1)
            if target.dim() == 3:
                target = target.unsqueeze(0)
        else:
            target = t_
        if isinstance(z_, np.ndarray):
            z = torch.stack((torch.tensor(z_.real, device=self.shells[0].device),
                             torch.tensor(z_.imag, device=self.shells[0].device)), -1)
        else:
            z = z_.to(target)
        norm_coeff = torch.sum(torch.pow(torch.fft(target, 2), 2), -1)
        errors = torch.sum(torch.pow(torch.fft(z - target, 2), 2), -1)
        # normed_radial_errors = [[
        #    torch.sum(torch.masked_select(e, m)) / torch.clamp(torch.sum(
        #    torch.masked_select(nc, m)), min=1e-12) for e, nc in zip(
        #    errors, norm_coeff)] for m in self.shells]
        normed_radial_errors = [[
            torch.sum(torch.masked_select(e, m)) / torch.clamp(torch.sum(
                torch.masked_select(nc, m)), min=1e-8) for m in self.shells
        ] for e, nc in zip(errors, norm_coeff)]
        if isinstance(t_, np.ndarray) or isinstance(z_, np.ndarray):
            return np.stack([np.array([t.cpu().item() for t in nre]) for nre in normed_radial_errors], 0)
        else:
            per_shell_loss = [sum(b) for b in normed_radial_errors]
            if self.weights is not None:
                loss = sum([l * w for l, w in zip(per_shell_loss, self.weights)]) / len(per_shell_loss)
            else:
                loss = sum(per_shell_loss) / len(per_shell_loss)
            return loss

        # per_shell_loss = [sum(b) for b in normed_radial_errors]
        # if weights is None:
        #    loss = sum(per_shell_loss)
        # else:
        #    loss = sum([l * w for l, w in zip(per_shell_loss, weights)])
        # return loss / (len(per_shell_loss) * len(normed_radial_errors[0]))


class FourierShellCorrelationLoss(Module):
    def __init__(self, n_bins):
        self.shells = []
        self.n_bins = n_bins
        super(FourierShellCorrelationLoss, self).__init__()

    def make_shells(self, shape, device='cuda'):
        frame = np.amax(shape)
        max_r = (1. + np.amin(shape) / np.amax(shape)) / 2.
        # n_bins = 30#np.round(np.mean(shape)) - 1
        afocal, bfocal = shape[0] / 2, shape[1] / 2
        xx, yy = np.mgrid[:shape[0], :shape[1]]
        ellipse = np.power((xx - afocal) * 2 / frame,
                           2) + np.power((yy - bfocal) * 2 / frame, 2)
        ellipses = [torch.tensor(
            (ellipse < r_2d).astype(np.bool), dtype=torch.bool, device=device
        ) for r_2d in np.power(np.linspace(0, max_r, self.n_bins + 1)[1:], 2.)]
        self.shells = [ellipses[0]] + [e2 ^ e1 for e1, e2 in zip(ellipses[:-1], ellipses[1:])]

    def forward(self, z, target, weights=None):
        if not self.shells or self.shells[0].size()[1:-1] != z.size()[1:-1]:
            self.make_shells(z.size()[1:-1], device=z.device)
        norm_coeff = torch.sum(torch.pow(torch.fft(target, 2), 2), -1)
        errors = torch.sum(torch.pow(torch.fft(z - target, 2), 2), -1)
        normed_radial_errors = [[
            torch.sum(torch.masked_select(e, m)) / torch.clamp(torch.sum(
                torch.masked_select(nc, m)), min=1e-12) for e, nc in zip(
                errors, norm_coeff)] for m in self.shells]
        per_shell_loss = [sum(b) for b in normed_radial_errors]
        if weights is None:
            loss = sum(per_shell_loss)
        else:
            loss = sum([l * w for l, w in zip(per_shell_loss, weights)])
        return loss / (len(per_shell_loss) * len(normed_radial_errors[0]))


class IndRNN(Module):
    def __init__(self, input_size, hidden_size, kernel=1,
                 dilation=1, bias=True):
        super(IndRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if isinstance(kernel, (list, tuple)):
            kernel = kernel[0]
            dilation = dilation[0]
        padding = int((kernel + (kernel - 1) * (dilation - 1)) / 2)
        self.weight_hh = Parameter(
            normal_(torch.empty(1, hidden_size, 1, 1),
                    std=1. / (hidden_size * (1 + kernel ** 2))))
        self.conv_ih = Conv2d(
            input_size, hidden_size, kernel,
            padding=padding, dilation=dilation, bias=bias)
        normal_(self.conv_ih.weight, std=1. / (hidden_size * (1 + kernel ** 2)))
        if bias:
            constant_(self.conv_ih.bias, 0)

    def forward(self, input, hidden):
        ih = self.conv_ih(input)
        out = ReLU()(ih + self.weight_hh * hidden)
        return out


class MIRNN(Module):
    def __init__(self, input_size, hidden_size, kernel=1,
                 dilation=1, bias=True):
        super(MIRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
            dilation = (dilation, dilation)
        padding = [int((k + (k - 1) * (d - 1)) / 2) for k, d in zip(kernel, dilation)]
        self.conv_ih = Conv2d(
            input_size, hidden_size, kernel[0],
            padding=padding[0], dilation=dilation[0], bias=False)
        self.conv_hh = Conv2d(
            hidden_size, hidden_size, kernel[1],
            padding=padding[1], dilation=dilation[1], bias=False)
        self.beta_ih = Parameter(
            xavier_uniform_(torch.empty(1, hidden_size, 1, 1), gain=calculate_gain('relu')))
        self.beta_hh = Parameter(
            xavier_uniform_(torch.empty(1, hidden_size, 1, 1), gain=calculate_gain('relu')))
        self.alpha = Parameter(
            xavier_uniform_(torch.empty(1, hidden_size, 1, 1), gain=calculate_gain('relu')))
        self.bias = Parameter(
            xavier_uniform_(torch.empty(1, hidden_size, 1, 1), gain=calculate_gain('relu')))

    def forward(self, input, hidden):
        ih = self.conv_ih(input)
        hh = self.conv_hh(hidden)
        return ReLU()(
            self.alpha * ih * hh + self.beta_ih * ih + self.beta_hh * hh + self.bias)


class IndMIRNN(Module):
    def __init__(self, input_size, hidden_size, kernel=1,
                 dilation=1, bias=True):
        super(IndMIRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
            dilation = (dilation, dilation)
        padding = [int((k + (k - 1) * (d - 1)) / 2) for k, d in zip(kernel, dilation)]
        self.conv_ih = Conv2d(
            input_size, hidden_size, kernel[0],
            padding=padding[0], dilation=dilation[0], bias=False)
        self.weight_hh = Parameter(
            xavier_uniform_(torch.empty(1, hidden_size, 1, 1), gain=calculate_gain('relu')))
        self.beta_ih = Parameter(
            xavier_uniform_(torch.empty(1, hidden_size, 1, 1), gain=calculate_gain('relu')))
        self.beta_hh = Parameter(
            xavier_uniform_(torch.empty(1, hidden_size, 1, 1), gain=calculate_gain('relu')))
        self.alpha = Parameter(
            xavier_uniform_(torch.empty(1, hidden_size, 1, 1), gain=calculate_gain('relu')))
        self.bias = Parameter(
            xavier_uniform_(torch.empty(1, hidden_size, 1, 1), gain=calculate_gain('relu')))

    def forward(self, input, hidden):
        ih = self.conv_ih(input)
        hh = hidden * self.weight_hh
        return ReLU()(
            self.alpha * ih * hh + self.beta_ih * ih + self.beta_hh * hh + self.bias)


class ConvGRU(Module):
    def __init__(self, input_size, hidden_size,
                 kernel=1, dilation=1, bias=True, spectral_norm=False):
        super(ConvGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.spectral_norm = spectral_norm

        if isinstance(kernel, int):
            kernel = (kernel, kernel)
            dilation = (dilation, dilation)
        padding = [int((k + (k - 1) * (d - 1)) / 2) for k, d in zip(kernel, dilation)]
        self.conv_ih = Conv2d(
            input_size, 3 * hidden_size, kernel[0],
            padding=padding[0], dilation=dilation[0], bias=False)
        self.conv_hh = Conv2d(
            hidden_size, 3 * hidden_size, kernel[1],
            padding=padding[1], dilation=dilation[1], bias=bias)
        if self.spectral_norm:
            self.conv_ih = SpectralNorm(self.conv_ih)
            self.conv_hh = SpectralNorm(self.conv_hh)

    def forward(self, input, hidden):
        i_r, i_z, i_c = self.conv_ih(input).chunk(3, 1)
        h_r, h_z, h_c = self.conv_hh(hidden).chunk(3, 1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        candidate = torch.tanh(i_c + reset * h_c)
        return candidate + update * (hidden - candidate)


class ConvMGU(Module):
    def __init__(self, input_size, hidden_size,
                 kernel=1, dilation=1, bias=True, spectral_norm=False):
        super(ConvMGU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.spectral_norm = spectral_norm
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
            dilation = (dilation, dilation)
        padding = [int((k + (k - 1) * (d - 1)) / 2) for k, d in zip(kernel, dilation)]
        self.conv_ih = Conv2d(input_size, 2 * hidden_size, kernel[0], padding=padding[0], dilation=dilation[0],
                              bias=bias)
        self.conv_hh = Conv2d(hidden_size, 2 * hidden_size, kernel[1], padding=padding[1], dilation=dilation[1],
                              bias=False)
        xavier_uniform_(self.conv_ih.weight, calculate_gain('relu'))
        xavier_uniform_(self.conv_hh.weight)
        if self.spectral_norm:
            self.conv_ih = SpectralNorm(self.conv_ih)
            self.conv_hh = SpectralNorm(self.conv_hh)

    def forward(self, input, hidden):
        f_i, c_i = self.conv_ih(input).chunk(2, 1)
        f_h, c_h = self.conv_hh(hidden).chunk(2, 1)
        forget = torch.sigmoid(f_i + f_h)
        candidate = torch.tanh(forget * c_h + c_i)
        hy = candidate + forget * (hidden - candidate)
        return hy


""" Minimal Gated Unit"""


class convMGUCell(Module):
    def __init__(self, input_size, hidden_size, kernel_size=1, bias=True):
        super(convMGUCell, self).__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.forget_gate = Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding, bias=self.bias)
        self.candidate_activation_gate = Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding,
                                                bias=self.bias)
        xavier_uniform_(self.forget_gate.weight, calculate_gain('relu'))
        xavier_uniform_(self.candidate_activation_gate.weight)
        constant_(self.candidate_activation_gate.bias, self.bias)

    def forward(self, input_, hidden):
        forget = torch.sigmoid(self.forget_gate(torch.cat([input_, hidden], dim=1)))
        candidate_activation = torch.tanh(
            self.candidate_activation_gate(torch.cat([input_, torch.mul(hidden, forget)], dim=1)))
        activation_state = torch.mul((1 - forget), hidden) + torch.mul(forget, candidate_activation)
        return activation_state


class MSEFSC(Module):
    def __init__(self, n_bins, reg_param):
        super(MSEFSC, self).__init__()
        self.n_bins = n_bins
        self.fsc = FourierShellCorrelationLoss(n_bins)
        self.reg_param = reg_param

    def forward(self, e, t, w):
        return mse(e, t) + self.reg_param * self.fsc(e, t, w)


def l1(z, target):
    return torch.mean(torch.abs(z - target))


# WL
from torch.autograd import grad


def inner(u, v):
    return torch.sum(torch.sum((u * v), 1), 1)


def divide(u, v):
    return torch.where(torch.lt(torch.abs(v), 1e-7), torch.zeros_like(u), u / v)


class KMatrixFFT2(object):
    def __init__(self, dist_matrix):
        self.dist_matrix = torch.stack((dist_matrix, torch.zeros_like(dist_matrix)), -1)
        self.dist_matrix_fft = self.fft(self.dist_matrix)

    def fft(self, x):
        return torch.fft(x, 2, normalized=True)

    def ifft(self, x):
        return (torch.ifft(x, 2, normalized=True))[..., 0]

    def __call__(self, x):
        fftx = self.fft(x)
        mult_dist = self.dist_matrix_fft[..., 0] * fftx[..., 0] - self.dist_matrix_fft[..., 1] * fftx[..., 1]
        mult_dist = torch.stack((mult_dist, torch.zeros_like(mult_dist)), -1)
        return self.ifft(mult_dist)

    def adjoint(self, y):
        with torch.enable_grad():
            # y.clone().detach().requires_grad_(True)
            _y = torch.stack((y, torch.zeros_like(y)), -1)
            _y = self(_y)
            grad_ = grad(_y, y, y)[0]
            return grad_


def wasserstein_distance_helper(matrix_param, epsilon, mu0, mu1, niter):
    matrix_param = torch.cuda.FloatTensor(matrix_param)

    K_op = KMatrixFFT2(torch.exp(-matrix_param / epsilon))
    K_op_2 = KMatrixFFT2(matrix_param * torch.exp(-matrix_param / epsilon))

    v = torch.stack((torch.ones_like(mu1) / mu1.numel(), torch.zeros_like(mu1)), -1)
    # v = v.type(torch.FloatTensor).clone().detach().requires_grad_(True)

    for j in range(niter):
        u = divide(mu0, K_op(v))  # .clone().detach().requires_grad_(True)
        v = divide(mu1, K_op.adjoint(u))
        v = torch.stack((v, torch.zeros_like(v)), -1)

        if (u != u).sum() > 0 or (v != v).sum() > 0 or u.max() > 1e9 or v.max() > 1e9:  # u!=u is a test for NaN...
            raise Exception(
                str(('Warning: numerical errrors', j + 1, "u", (u != u).sum(), u.max(), "v", (v != v).sum(), v.max())))

    return inner(u, K_op_2(v)) / torch.tensor(mu1.numel())


def wasserstein_distance(z, target, epsilon=1e-3, niter=10, cutoff=0.3, p=4):
    z = z.permute(0, 3, 1, 2).contiguous().view(-1, *z.size()[1:3])
    target = target.permute(0, 3, 1, 2).contiguous().view(-1, *target.size()[1:3])

    # z = z.type(torch.FloatTensor)
    # target = target.type(torch.FloatTensor)

    xp, yp = np.meshgrid(np.linspace(-1, 1, z.shape[1]), np.linspace(-1, 1, z.shape[2]))
    matrix_param = xp ** (p) + yp ** (p)
    matrix_param = cutoff ** p * (1 - np.exp(-matrix_param / cutoff ** p))
    # import matplotlib.pyplot as plt
    # plt.imshow(matrix_param, cmap=cc.m_gray)
    # plt.show()
    matrix_param = (np.fft.ifftshift(matrix_param))[None, ...]
    result = wasserstein_distance_helper(matrix_param, epsilon=epsilon, mu0=z, mu1=target, niter=niter)
    dist = result / cutoff ** (p)
    dist = torch.mean(dist)

    return dist


class SinkhornDistance(Module):
    def __init__(self, eps, max_iter, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        n_points = x.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, n_points, dtype=torch.float, requires_grad=False).fill_(1.0 / n_points).squeeze()
        nu = torch.empty(batch_size, n_points, dtype=torch.float, requires_grad=False).fill_(1.0 / n_points).squeeze()
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - self.lse(self.M(C, u, v))) + u
            v = self.eps * (torch.log(nu + 1e-8) - self.lse(self.M(C, u, v).transpose(-2, -1))) + v
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1
            if err.item() < thresh:
                break
        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def lse(A):
        "log-sum-exp"
        # add 10^-6 to prevent NaN
        result = torch.log(torch.exp(A).sum(-1) + 1e-6)
        return result

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists
from datetime import datetime as dt


def write_wl_imgs(C, P):
    _range = C.size(0)
    C = C.detach().numpy()
    P = P.detach().numpy()

    incr = 0
    cwdir = 'imgs/wasserstein_distance_imgs/cost_matrix_imgs/' + str(incr) + '/'
    if not exists(cwdir):
        pwdir = 'imgs/wasserstein_distance_imgs/transport_plan_imgs/' + str(incr) + '/'
        makedirs(cwdir)
        makedirs(pwdir)
    elif exists(cwdir):
        while exists('imgs/wasserstein_distance_imgs/cost_matrix_imgs/' + str(incr) + '/'):
            incr += 1
        cwdir = 'imgs/wasserstein_distance_imgs/cost_matrix_imgs/' + str(incr) + '/'
        pwdir = 'imgs/wasserstein_distance_imgs/transport_plan_imgs/' + str(incr) + '/'
        makedirs(cwdir)
        makedirs(pwdir)

    for i in range(_range):
        plt.imshow(C[i, :, :])
        fig = plt.gcf()
        fig.savefig(join(cwdir, 'cost_matrix_' + str(i) + '.png'))
        plt.imshow(P[i, :, :])
        fig = plt.gcf()
        fig.savefig(join(pwdir, 'transport_plan_' + str(i) + '.png'))


def _wasserstein_distance(z, target, p=4):
    z = z.permute(0, 3, 1, 2).contiguous().view(-1, *z.size()[1:3])
    target = target.permute(0, 3, 1, 2).contiguous().view(-1, *target.size()[1:3])
    z = z.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)

    # x_col = z.unsqueeze(-2)
    # y_lin = target.unsqueeze(-3)
    # cost_matrix = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)

    # dist, P = wasserstein_distance_sinkhorn(z, cost_matrix, epsilon=1e-3, max_iter=100)
    # dist = torch.mean(dist)

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
    dist, P, C = sinkhorn(z, target)

    repeat_mins = [0, 20, 40]
    if (dt.now().minute in repeat_mins) and (dt.now().second < 11):
        write_wl_imgs(C, P)

    return dist


act_fns = {'relu': ReLU(), 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}
loss_fns = {'mse': lambda x, y: lambda e, t, w: mse(e, t),
            'complex_rmse': lambda x, y: lambda e, t, w: complex_rmse(e, t),
            'rmse': lambda x, y: lambda e, t, w: rmse(e, t),
            'fsc': lambda x, y: FourierShellCorrelationLoss(x),
            'msefsc': lambda x, y: MSEFSC(x, y),
            'l1': lambda x, y: lambda e, t, w: l1(e, t),
            # 'wl': lambda x, y: lambda e, t, w: wasserstein_distance(e, t)}
            'wl': lambda x, y: lambda e, t, w: _wasserstein_distance(e, t)}
scale = {'scale': scaler}
mult = {'mul': mul}
recurrencies = {
    'GRU': GRUCell, 'LNGRU': LNGRUCell, 'convGRU': ConvGRU,
    'convMGU': ConvMGU, 'convMGUCell': convMGUCell, 'IndRNN': IndRNN, 'MIRNN': MIRNN,
    'IndMIRNN': IndMIRNN}
convs = {'2d': Conv2d, '3d': Conv3d}
