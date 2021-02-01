import argparse
import glob
import re
import sys
from os import listdir
from os.path import join, exists, split, isdir
from pathlib import Path

import dill
import h5py
import numpy as np
import pickle5 as pickle
import torch

from projects.tecfidera.reconstruction.deepmri.plot import ModelLog
from projects.tecfidera.reconstruction.deepmri.rim import Rim


def MRIData(data_path=None, mask_path=None):
    data = []

    volumes = glob.glob(data_path + "/*") if isdir(data_path) else [data_path]
    for i in range(len(volumes)):
        with open(volumes[i], 'rb') as f:
            vol = pickle.load(f, encoding='latin1')
        imspace, sense = vol[0], vol[1]
        imspace = torch.from_numpy(imspace).permute(2, 0, 1)
        sense = torch.from_numpy(sense).permute(2, 0, 1)

        with open(mask_path, 'rb') as f:
            mask = pickle.load(f)

        mask = torch.from_numpy(mask)
        if mask.dim() > 3:
            mask = mask[..., 0]

        y = torch.fft.fft2(imspace, dim=(-2, -1)) * mask
        eta = torch.sum(torch.fft.ifft2(y, dim=(-2, -1)) * torch.conj(sense), 0).unsqueeze(0)

        mask = mask.unsqueeze(0)

        data.append(['_'.join(str(volumes[i]).split('/')[-1].split('_')[:-1]), eta, y, sense, mask])

    return data


def get_network(model, loaded=False, device='cuda'):
    with open(join(model.runfolder, 'args'), 'rb') as f:
        ARGS = dill.load(f)

    conv_params = {'f': ARGS.conv_features, 'k': ARGS.conv_kernels, 'd': ARGS.conv_dilations}
    rnn_params = {'f': ARGS.rnn_features, 'k': [[k1, k2] for k1, k2 in zip(
        ARGS.rnn_kernels[::2], ARGS.rnn_kernels[1::2])], 'd': [[d1, d2] for d1, d2 in zip(
        ARGS.rnn_dilations[::2], ARGS.rnn_kernels[1::2])]}

    network = Rim(t_max=ARGS.t_max, act_fn=ARGS.act_fn,
                  n_feature=ARGS.n_feature if type(ARGS.n_feature) != int else [ARGS.n_feature],
                  recurrent=ARGS.recurrent,
                  conv_params=conv_params, rnn_params=rnn_params).to(dtype=getattr(torch, ARGS.dtype))

    if loaded:
        savedmodel = join(model.runfolder, 'models', model.to_reconstruct)
        while savedmodel != None and savedmodel != 'None':
            try:
                load = torch.load(savedmodel, map_location=lambda storage, loc: storage.cpu())
                break
            except FileNotFoundError:
                savedmodel = join(ARGS.saved_model.split('/models/')[0], 'models', model.to_reconstruct)
                args = join(ARGS.saved_model.split('models')[0], 'args')

                if not exists(args):
                    savedmodel = join(model.train_path, savedmodel.split('train_dir/')[1])
                    args = join(model.train_path, args.split('train_dir/')[1])

                with open(args, 'rb') as f:
                    ARGS = dill.load(f)

        network.load_state_dict(load)

    return network.to(device)


def get_model_log(modeldir, name='', gridloc='0', acc=None):
    modelsplit = modeldir.split('models/')
    traindir, folder = split(modelsplit[0])

    if not folder:
        traindir, folder = split(traindir)

    if acc is None:
        with open(args.mask_dir, 'rb') as f:
            mask = pickle.load(f)
        acc = np.round(mask.size / np.sum(mask), 2)

    testaccs = [float(re.search('\d{1,2}', a).group(0)) for a in
                filter(lambda x: 'testloss_acc' in x, listdir(modeldir))]
    model = ModelLog(folder, traindir, name=name, acceleration=[int(np.round(acc))])
    model.runfolder = modeldir

    if len(modelsplit) == 2:
        fromto = [int(m) for m in modelsplit[1][re.search('\d', modelsplit[1]).start():].split('-')]
        if len(fromto) == 1:
            model.set_reconstruction_model(modelsplit[1])
        else:
            m = 'checkpoint' + str(model.get_best_model(min(testaccs, key=lambda x: abs(x - acc)), trainlim=fromto))
            model.set_reconstruction_model(m)
            print('Selecting best model between checkpoints {} and {}, found to be {}.'.format(*fromto, m))
    else:
        m = 'checkpoint' + str(model.get_best_model(acc=min(testaccs, key=lambda x: abs(x - acc))))
        model.set_reconstruction_model(m)
        print('No model checkpoint specified for {}, using {}, the best performing model on validation set at '
              'acceleration {}.'.format(folder, m, acc))

    model.gridloc = (int(n) for n in gridloc.split())
    return model


def main(args):
    if args.wdir is not None:
        Path(args.wdir).mkdir(parents=True, exist_ok=True)

    data = MRIData(data_path=args.mri_dir, mask_path=args.mask_dir)

    device = args.device
    t_max = args.t_max

    network = get_network(get_model_log(args.modeldir), True, device=device)

    recons = []
    prev_name = ''
    for i in range(len(data)):
        name, eta, y, sense, mask = data[i]
        recon = next(network.inference(eta.to(device), y.to(device), sense.to(device), mask.to(device), t_max))
        recon = recon.squeeze().detach().cpu().numpy()
        recon = np.abs(recon / np.max(recon))
        recons.append(recon)

        if (name != prev_name and prev_name != '') or i == len(data) - 1:
            with h5py.File(args.wdir + '/' + name + ".h5", "w") as f:
                f['reconstruction'] = recons
            recons = []
        else:
            prev_name = name

        del name, eta, y, sense, mask, recon


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mri_dir', help='Path to MR-image data, stored as a pickle dump')
    parser.add_argument('mask_dir', help='Path to mask data.')
    parser.add_argument('modeldir', help='Path to saved models.')
    parser.add_argument('--wdir', help='Enter path to save reconstructions.')
    parser.add_argument('--inputscale', type=float, default=1.0, help='Scale of the input with, defaults to 1.0.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Enable GPU.')
    parser.add_argument('--t_max', type=int, default=8, help='What time-step to show.')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
