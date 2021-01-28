import collections
import itertools
import os
import random
import re
import sys
from os import listdir
from os.path import join, isdir, isfile

import numpy as np
import numpy.fft as fft
import pickle5 as pickle
import torch
from torch._six import string_classes, int_classes
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')

import bart


def center_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
        and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class MRIData(DataLoader):
    """
    Top-level class for data handling. Creates a dataset and collate function to
    process the retrieved data and return it in mini-batches.
    Input:
    batch_size:    int, size of mini-batches
    data_path:     directory containing train and test directories, containing
                   pickle dumps with data
    train:         bool, to collect data from train or test folder
    k_space:       bool, is data stored in K-space?
    accsampler:    callable, returning a random acceleration level per mini-batch
    acceleration:  float, acceleration level, if the same should be used
                   for all mini-batches
    premade_mask:  Path to pickle dumped mask, masks are randomly generated if
                   set to None
    fwhms:         Full-width-half-maximum of Gaussian used to pick samples from
    ellipse_scale: The percent of image axes to use for ellipse half-axes when
                   fully sampling low frequencies near origin
    sampler:       list of ints if the data to use is known
    superres:      Set True for doing super resolution task
    superresscale: Scale of ellipse to use for super resolution
    superresprob:  Probability of no further random downsampling during super
                   resolution training.
    """

    def __init__(
        self, batch_size=1, data_path=None, mask_path=None, train=False,
        num_workers=4, sampling_dist='gaussian', accsampler=None, inputscale=1.,
        acceleration=None, fwhms=(.7, .7), ellipse_scale=.08,
        drop_last=False, crop=True, shift=True, permute=True, superres=False,
        minmaxcrop=(190, 190), superresscale=None, superresprob=1 / 3,
        complex_dim=-1, prospective=False,
        multiple_datasets_dir=None, n_coil=1,
        sampler_type=None, sampler_weights=1):
        # sampler should be list of indices to pick images, otherwise it's
        # random uniform when training, sequential when not.
        self.n_coil = n_coil
        self.data_path = data_path
        self.batch_size = batch_size
        self.prospective = prospective
        self.scale = inputscale
        sampler = None
        if train:
            assert not prospective
            _masker = Masker(
                sampling_dist, accsampler, acceleration, fwhms,
                ellipse_scale, superres, superresscale, superresprob)
            masker = lambda s: _masker(s)
            collate = default_collate
            train = True
            transforms = []
            if crop: transforms.append(RandomCrop(minmaxcrop[1]))
            if shift: transforms.append(RandomPhaseShift())
            if permute: transforms.append(RandomPermute())
            if transforms:
                transforms.append(np.ascontiguousarray)
                transform = Compose(transforms)
            else:
                transform = lambda x: x
            # dataset = Brains(self.data_path, train, masker, transform, complex_dim, inputscale, prospective)
        elif prospective:
            dataset = Brains(
                self.data_path, train, None, lambda x: x,
                complex_dim, self.scale, prospective, step=batch_size,
                n_coil=self.n_coil)
            self.mask = dataset.mask
            masker = lambda *s: self.mask
            sampler = SubsetSequentialSampler(np.arange(len(dataset)))
            self.batch_size = 1

            def collate(batch):
                batch = {key: torch.tensor(batch[0][key]) for key in batch[0]}
                batch['mask'].unsqueeze_(0)
                return batch
        else:
            if mask_path is None:
                self.mask = []
            else:
                with open(mask_path, 'rb') as f:
                    self.mask = pickle.load(f)
            masker = lambda *s: self.mask

            def collate(batch):
                batch = default_collate(batch)
                batch['mask'] = torch.tensor(masker(), dtype=batch['y'].dtype
                                             ).unsqueeze(0)
                return batch

            transform = lambda x: x

        if multiple_datasets_dir is not None:
            _m_datasets_dir = multiple_datasets_dir
            _m_datasets_dir.insert(0, self.data_path)
            _multiple_datasets = []
            length_datasets = []

            for i in range(0, len(_m_datasets_dir)):
                _dataset = Brains(_m_datasets_dir[i], train, masker, transform, complex_dim, inputscale, prospective, 0,
                                  self.n_coil)
                _multiple_datasets.append(_dataset)

            length_datasets = [len(e) for e in _multiple_datasets]
            dataset = torch.utils.data.ConcatDataset(_multiple_datasets)

            if sampler_type == 'weighted':
                weights = []
                if len(sampler_weights) == len(length_datasets):
                    _sampler_weights = sampler_weights
                    for ld, sw in zip(length_datasets, sampler_weights):
                        weights += [sw] * ld
                else:
                    _multiple_datasets.sort(key=len)
                    length_datasets = [len(e) for e in _multiple_datasets]
                    dataset = torch.utils.data.ConcatDataset(_multiple_datasets)
                    _sampler_weights = []
                    for m in range(0, len(length_datasets)):
                        weight = (1. / len(dataset)) * length_datasets[m]
                        _sampler_weights.append(weight)

                    _sampler_weights.sort(reverse=True)
                    for _ld, _sw in zip(length_datasets, _sampler_weights):
                        weights += [_sw] * _ld

                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size, True)
                train = False
        else:
            dataset = Brains(self.data_path, train, masker, transform, complex_dim, inputscale, prospective, 0,
                             self.n_coil)

        if sampler is not None:
            _datapaths = [p.data_path.split('/')[-2] for p in dataset.datasets]
            if sampler_weights == _sampler_weights:
                _weights = sampler_weights
            else:
                _ind = [len(p) - 1 for p in dataset.datasets]
                _weights = [p.item() for p in sampler.weights[_ind]]
            print("Sampler weights per dataset:",
                  ", ".join("{} = {}".format(x, y) for x, y in zip(_datapaths, _weights)))

        super(MRIData, self).__init__(dataset, batch_size=self.batch_size, shuffle=train, sampler=sampler,
                                      collate_fn=collate, drop_last=drop_last, num_workers=num_workers)

    def set_mask(self, mask_path):
        with open(mask_path, 'rb') as f:
            self.mask = pickle.load(f)

        super(MRIData, self).__init__(
            dataset, batch_size=self.batch_size, shuffle=train, sampler=sampler,
            collate_fn=collate, drop_last=drop_last, num_workers=num_workers)

    def set_mask(self, mask_path):
        with open(mask_path, 'rb') as f:
            self.mask = pickle.load(f)

    def get_loaded_target(self):
        assert isfile(self.data_path)

        with open(self.data_path, 'rb') as f:
            target = pickle.load(f, encoding='latin1')

        if target.ndim == 4 or target.ndim == 5:
            target = np.sum(target[0] * target[1].conj(), -1)

        return self.dataset.scale * target


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices,
    without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter([index for index in self.indices])

    def __len__(self):
        return len(self.indices)


class RandomPhaseShift(object):
    def __call__(self, target):
        shiftangle = np.random.uniform(-np.pi, np.pi)
        return np.absolute(target).real * (
            np.cos(shiftangle) + 1j * np.sin(shiftangle))


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, target):
        h, w = target.shape[-2:]
        new_h, new_w = self.output_size
        if h == new_h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)
        if w == new_w:
            left = 0
        else:
            left = np.random.randint(0, w - new_w)
        return target[..., top: top + new_h, left: left + new_w]


class RandomPermute(object):
    def __init__(self):
        self.permutations = [
            lambda t: t, lambda t: t.T, lambda t: np.fliplr(t),
            lambda t: np.flipud(t), lambda t: np.flipud(np.fliplr(t)),
            lambda t: np.flipud(t).T, lambda t: np.fliplr(t).T,
            lambda t: np.flipud(np.flipud(t).T)
        ]

    def __call__(self, target):
        perm = random.choice(self.permutations)
        if target.ndim == 4:
            s = target.shape
            target = target.reshape((-1, *s[-2:]))
            out = np.stack([perm(img) for img in target], 0)
            return out.reshape(s)
        elif target.ndim == 2:
            return perm(target)
        else:
            raise ValueError(
                'Stored ndarray should be either 2- or 4-dim, '
                'not {}-dim.'.format(target.ndim))


class Brains(Dataset):
    def __init__(self, data_path, train, masker, transform, complex_dim, scale, prospective, step, n_coil):
        super(Brains, self).__init__()
        self.n_coil = n_coil
        self.scale = scale
        self.train = train
        self.masker = masker
        self.transform = transform
        self.complex_dim = complex_dim
        self.prospective = prospective
        self.data_path = data_path

        if prospective:
            with open(self.data_path, 'rb') as f:
                y, sense = pickle.load(f)
            y = y * self.scale
            self.mask = np.where(np.sum(np.sum(np.absolute(y).real, 0), -1) > 0., 1, 0)
            self.masker = lambda *x: self.mask
            eta = np.sum(np.fft.ifft2(y, axes=(1, 2)) * sense.conj(), -1)
            y = np.transpose(y, (0, 3, 1, 2))
            sense = np.transpose(sense, (0, 3, 1, 2))
            self.scan = [{
                'mask': self.mask,
                'eta': np.stack(
                    (eta[i:i + step].real, eta[i:i + step].imag), self.complex_dim),
                'target': np.stack(
                    (eta[i:i + step].real, eta[i:i + step].imag), self.complex_dim),
                'sense': np.stack(
                    (sense[i:i + step].real, sense[i:i + step].imag), -1),
                'y': np.stack((y[i:i + step].real, y[i:i + step].imag), -1)
            } for i in range(0, len(y), step)]
            self.bins = [len(self.scan)]
        else:
            if isfile(data_path):
                self.bins = [1]
            else:
                if self.train:
                    self.data_path = join(data_path, 'train')
                else:
                    self.data_path = join(data_path, 'test')

                self.subdirs = listdir(self.data_path)
                if isdir(join(self.data_path, self.subdirs[0])):
                    self.fnames = [listdir(join(
                        self.data_path, sub)) for sub in self.subdirs]
                    self.bins = np.cumsum([len(f) for f in self.fnames])
                else:
                    self.fnames = [self.subdirs]
                    self.subdirs = ['']
                    self.bins = [len(self.fnames[0])]

    def __len__(self):
        return self.bins[-1]

    def __getitem__(self, index):
        if self.prospective:
            return self.scan[index]
        else:
            if isfile(self.data_path):
                fname = self.data_path
            else:
                idx = np.digitize(index, self.bins)
                if idx > 0:
                    self.bins = torch.LongTensor(self.bins)
                    index -= self.bins[idx - 1]
                fname = join(
                    self.data_path, self.subdirs[idx], self.fnames[idx][index])
            with open(fname, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            data = self.transform(data)
            imspace, sense = data[0], data[1]
            imspace = np.transpose(imspace, (2, 0, 1))
            sense = np.transpose(sense, (2, 0, 1))

            mask = self.masker(imspace.shape[-2:])
            if mask.ndim > 3:
                mask = mask[..., 0]

            pics = np.fft.fftshift(bart.bart(1, 'pics -d0 -S -R W:7:0:0.005 -i 60', np.expand_dims(np.transpose(
                np.fft.ifftshift(np.fft.fft2(imspace) * mask, axes=(-2, -1)), (1, 2, 0)), 0),
                np.expand_dims(np.transpose(np.fft.fftshift(sense, axes=(-2, -1)), (1, 2, 0)), 0))[0], axes=(-2, -1))

            target = np.sum(imspace * sense.conj(), 0)
            y = np.fft.fft2(imspace) * mask
            eta = np.sum(np.fft.ifft2(y) * sense.conj(), 0)

            data_dict = {'eta': np.stack((eta.real, eta.imag), self.complex_dim),
                         'y': np.stack((y.real, y.imag), self.complex_dim),
                         'target': np.stack((target.real, target.imag), self.complex_dim),
                         'sense': np.stack((sense.real, sense.imag), self.complex_dim),
                         'pics': pics}

            if self.train:
                data_dict['mask'] = mask

            return data_dict


class Masker(object):
    def __init__(self, sampling_dist, accsampler, acceleration,
                 fwhms, scale, superres, superresscale, superresprob):
        self.superres = superres
        self.superresscale = superresscale
        self.superresprob = superresprob
        self.accsampler = accsampler
        self.sampling_dist = sampling_dist
        self.scale = scale
        self.acceleration = acceleration

        # Define functions for getting indices to make masks from,
        # corresponding to various sub-sampling schemes
        def gaussian_kspace(shape, scale):
            a, b = scale * shape[0], scale * shape[1]
            afocal, bfocal = shape[0] / 2, shape[1] / 2
            xx, yy = np.mgrid[:shape[0], :shape[1]]
            ellipse = np.power(
                (xx - afocal) / a, 2) + np.power((yy - bfocal) / b, 2)
            return (ellipse < 1).astype(float)

        def gaussian_coordinates(shape, acceleration):
            n_sample = int(shape[0] * shape[1] / acceleration)
            cartesian_prod = [e for e in np.ndindex(shape)]
            kernel = _make_gaussian_kernel(shape, fwhms)
            idxs = np.random.choice(
                range(len(cartesian_prod)), size=n_sample,
                replace=False, p=kernel.flatten())
            return list(zip(*list(map(cartesian_prod.__getitem__, idxs))))

        def gaussian1d_kspace(shape, scale):
            scaled = int(shape[0] * scale)
            center = np.ones((scaled, shape[1]))
            topscaled = (shape[0] - scaled) // 2
            btmscaled = (shape[0] - scaled - topscaled)
            top = np.zeros((topscaled, shape[1]))
            btm = np.zeros((btmscaled, shape[1]))
            return np.concatenate((top, center, btm))

        def gaussian1d_coordinates(shape, acceleration):
            n_sample = int(shape[0] / acceleration)
            kernel = _make_gaussian_kernel(shape, fwhms, dim=1)
            idxs = np.random.choice(
                range(shape[0]), size=n_sample,
                replace=False, p=kernel)
            xsamples = np.concatenate([np.tile(i, shape[1]) for i in idxs])
            ysamples = np.concatenate([range(shape[1]) for i in idxs])
            return xsamples, ysamples

        if sampling_dist == 'radial':
            raise NotImplementedError()

        def periodic_kspace(shape, scale):
            mask = np.zeros(shape)
            scaled = int(scale * shape[0]), int(scale * shape[1])
            start = (shape[0] - scaled[0]) // 2, (shape[1] - scaled[1]) // 2
            end = (shape[0] + scaled[0]) // 2, (shape[1] + scaled[1]) // 2
            mask[start[0]:end[0], start[1]:end[1]] = 1.
            return mask

        def periodic_coordinates(shape, acceleration):
            return list(zip(*list(itertools.product(
                np.arange(0, shape[0], int(acceleration)),
                np.arange(0, shape[1], int(acceleration))))))

        def periodic1d_kspace(shape, scale):
            mask = np.zeros(shape)
            scaled = int(scale * shape[0])
            start = (shape[0] - scaled) // 2
            end = (shape[0] + scaled) // 2
            mask[start:end, :] = 1.
            return mask

        def periodic1d_coordinates(shape, acceleration):
            return list(zip(*list(itertools.product(
                np.arange(0, shape[0], int(acceleration)),
                np.arange(shape[1])))))

        def poisson_coordinates(shape, acceleration):
            mask = np.squeeze(bart(1,
                                   'poisson -Y{0} -Z{1} -y{2} -z{2} -v -e -s{3}'.format(
                                       *shape, acceleration, np.random.randint(10000000000000))))
            return np.nonzero(mask)

        def poisson1d_coordinates(shape, acceleration):
            mask = np.squeeze(bart(1,
                                   'poisson -Y{0} -Z{1} -y{2} -z1 -v -s{3}'.format(
                                       *shape, acceleration, np.random.randint(10000000000000))))
            return np.nonzero(mask)

        self.get_central_kspace = {
            'gaussian': gaussian_kspace,
            'gaussian1d': gaussian1d_kspace,
            'periodic': periodic_kspace,
            'periodic1d': periodic1d_kspace,
            'poisson': gaussian_kspace,
            'poisson1d': gaussian1d_kspace
        }
        self.get_sample_coordinates = {
            'gaussian': gaussian_coordinates,
            'gaussian1d': gaussian1d_coordinates,
            'periodic': periodic_coordinates,
            'periodic1d': periodic1d_coordinates,
            'poisson': poisson_coordinates,
            'poisson1d': poisson1d_coordinates
        }
        self.sampling_dist = sampling_dist

    def __call__(self, shape):
        """
        Generates masks used for data corruption purposes in such a way that
        samples are picked from k-space in accordance with a centered gaussian
        kernel pdf stored in self.gaussian_kernel. Sample_size says how many
        samples should be picked, n_mask says how many masks to generate for
        each image.
        """
        if self.sampling_dist == 'random':
            sampling_dist = [
                'gaussian', 'gaussian1d', 'periodic',
                'periodic1d', 'poisson', 'poisson1d'
            ][np.random.randint(len(dists))]
        else:
            sampling_dist = self.sampling_dist
        if self.superres:
            if self.superresscale is None:
                scale = np.random.uniform(.2, .5)
            else:
                scale = self.superresscale
            base = self.get_central_kspace[sampling_dist](shape, scale)
            if self.acceleration is None:
                if np.random.uniform() > self.superresprob:
                    acceleration = self.accsampler()
                else:
                    acceleration = 1
            else:
                acceleration = self.acceleration
        else:
            base = None
            if self.acceleration is None:
                acceleration = self.accsampler()
            else:
                acceleration = self.acceleration
        mask = self.get_central_kspace[sampling_dist](shape, self.scale)
        mask[tuple(self.get_sample_coordinates[
                       sampling_dist](shape, acceleration))] = 1.
        if base is not None:
            mask *= base
        mask = np.fft.ifftshift(mask)
        return mask


def _make_gaussian_kernel(shape, fwhms, dim=2):
    """
    Creates the gaussian kernel used in generate_mask. mask_shape should be the
    same as the shape of the MR-image, fwhms are the Full-width-half-maximum
    """
    kernels = []
    for fwhm, kern_len in zip(fwhms, shape):
        sigma = fwhm / np.sqrt(8 * np.log(2))
        x = np.linspace(-1., 1., kern_len)
        g = np.exp(-(x ** 2 / (2 * sigma ** 2)))
        if dim == 1:
            kernel = g
            break
        else:
            kernels.append(g)
    if dim == 2:
        kernel = np.sqrt(np.outer(kernels[0], kernels[1]))
    kernel = kernel / kernel.sum()
    return kernel
