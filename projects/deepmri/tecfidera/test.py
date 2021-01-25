import re
from os import listdir
from os.path import join, exists, split, basename

import numpy as np
import pandas as pd
import pickle5 as pickle
import torch
from skimage.metrics import structural_similarity, mean_squared_error, normalized_root_mse, peak_signal_noise_ratio

from projects.deepmri.tecfidera.data import MRIData
from projects.deepmri.tecfidera.plot import ModelLog


class Test(object):
    def __init__(self, logdir, testloader, testmaskdir='masks/'):
        self.testloader = testloader
        self.logdir = logdir
        self.testmaskdir = testmaskdir

    def __call__(self, network, t_max, trainstep, masksfolder=None, sigma=1., writer=None):
        accloss = {}
        if masksfolder is None:
            masksfolder = join(self.testloader.data_path, self.testmaskdir)
        accs = sorted(listdir(masksfolder))
        for acc in accs:
            a = float(acc.split('acc')[1])
            if a.is_integer():
                csvname = join(self.logdir, 'testloss_acc' + str(int(a)) + '.csv')
            else:
                csvname = join(self.logdir, 'testloss_acc' + str(a) + '.csv')
            masks = listdir(join(masksfolder, acc))
            to_log = np.zeros(t_max)
            for mask in masks:
                self.testloader.set_mask(join(masksfolder, acc, mask))
                for batch in self.testloader:
                    loss = next(network.inference(
                        batch, t_max, get_loss=True,
                        get_estimate=False, final=False, sigma=sigma))
                    loss = np.array([l.item() for l in loss]).flatten()
                    writer.add_scalar('testloss_{}x_final'.format(a), loss[-1], int(trainstep))
                    to_log += loss
            to_log /= (len(masks) * len(self.testloader))
            accloss[a] = np.mean(to_log)
            writer.add_scalar('testloss_{}x_total'.format(a), accloss[a], int(trainstep))
            df = pd.DataFrame([int(trainstep)] + to_log.tolist()).T
            df.columns.name = 'time-step'
            df.index.name = 'train-step'
            df.to_csv(csvname, header=False, index=False, mode='a')
        for n, p in network.named_parameters():
            writer.add_histogram(n, p.cpu().numpy(), int(trainstep))
        return accloss


def evaluate_model(
    network, model_log, datadir, mask_folder='masks', prefix='',
    eval_folder='evaluate',
    device='cuda', trainlim=(None, None), timelim=(None, None),
    batch_size=1, scale=1.0,
    n_coil=1, multiple_datasets=False):
    model_name = model_log.name if prefix == '' else prefix

    dataset_prefix = str(datadir).split('/')
    dataset_prefix = dataset_prefix[-2]

    num_param = network.get_num_params()
    maskdir = join(mask_folder)
    braindir = join(datadir)
    brain_fnames = listdir(braindir)
    brain_batch_fnames = [brain_fnames[i:i + batch_size] for i in range(0, len(brain_fnames), batch_size)]
    df = pd.DataFrame()

    with open(join(braindir, brain_fnames[0]), 'rb') as f:
        tmpimg = pickle.load(f)

    if tmpimg.ndim == 4:
        shape = tmpimg.shape[2:]
    else:
        shape = tmpimg.shape

    wdir = join(model_log.runfolder, 'evaluation_stats_' + braindir.split('/')[-3] + '.csv')
    # assert not exists(wdir)

    for acc in listdir(maskdir):
        try:
            model = model_log.get_best_model(float(re.search('\d{1,2}', acc).group(0)), trainlim=trainlim,
                                             timelim=timelim)
        except KeyError:
            print('{} has no log for acceleration {}, skipping it...'.format(model_log.name, acc))
            continue

        print(acc, 'checkpoint: ', model)
        # Get linked list of saved models starting from model_log's input going back
        # print(model_log.models)
        for folder in model_log.models:
            if multiple_datasets:
                folder = '/'.join(folder.split('/')[:-1])
            try:
                load = torch.load(join(folder, 'models', 'checkpoint' + str(model)),
                                  map_location=lambda storage, loc: storage.cpu())
            except FileNotFoundError:
                print('Best model (checkpoint{}) not in {} ...'.format(model, folder))
                continue

            network.load_state_dict(load)
            network.to(device)
            break

        for mask in listdir(join(maskdir, acc)):
            for brains in brain_batch_fnames:
                batch = [next(iter(
                    MRIData(data_path=join(braindir, brain), mask_path=join(maskdir, acc, mask), inputscale=scale))) for
                    brain in brains]

                batch = {key: torch.cat([d[key] for d in batch], 0) for key in batch[0]}
                targets = batch['target'].cpu().numpy()
                targets = np.clip(np.absolute(targets[..., 0] + 1j * targets[..., 1]).real, 0, 1)
                df_dict = {}

                for t, e in enumerate(
                    next(network.inference(batch, network.t_max, get_loss=False, get_estimate=True, final=False))):
                    e_ = e.cpu().numpy()
                    if e_.ndim == 4:
                        e_ = np.clip(np.absolute(e_[..., 0] + 1j * e_[..., 1]).real, 0, 1)
                    e_ = e_.astype(targets.dtype)

                    for brain, e_t, target in zip(brains, e_, targets):
                        df_dict['time-step'] = [t + 1]
                        df_dict['SSIM'] = [structural_similarity(target, e_t, data_range=1)]
                        df_dict['MSE'] = [mean_squared_error(target, e_t)]
                        df_dict['NMSE'] = [normalized_root_mse(target, e_t)]
                        try:
                            psnr = peak_signal_noise_ratio(target, e_t, data_range=1)
                        except ValueError:
                            psnr = np.nan
                        df_dict['PSNR'] = [psnr]
                        df_dict['acc'] = [int(re.search('\d{1,2}', acc).group(0))]
                        df_dict['mask'] = [mask]
                        df_dict['brain'] = [brain]
                        df_dict['n_params'] = [num_param]
                        df_dict['n_features'] = [network.n_feature]
                        df_dict['train-step'] = [int(model)]
                        df_dict['model'] = [model_name]
                        df_dict['scale'] = scale
                        head, data = split(datadir)
                        if not data:
                            data = basename(head)
                        df_dict['data'] = [data]

                        if not exists(wdir):
                            pd.DataFrame(columns=df_dict.keys()).to_csv(wdir, index=False, mode='w')
                        pd.DataFrame(df_dict).to_csv(wdir, index=False, header=None, mode='a')


class SigmaSearcher(object):
    def __init__(self, model, datadir):
        self.model = ModelLog(model)
        self.datadir = datadir

    def __call__(self, rim, sigmas, t_max=None, maskfolder=None):
        t_max = rim.t_max if t_max is None else t_max
        num_param = rim.get_num_params()
        if maskfolder is None:
            masksfolder = join(self.datadir, 'masks')
        braindir = join(self.datadir, 'test')
        accs = sorted(listdir(masksfolder))
        df_ = pd.DataFrame()
        for acc in accs:
            model = self.model.get_best(float(re.search('\d{1,2}', acc).group(0)))
            while True:
                print('acc: ', acc, 'model: ', model)
                try:
                    load = torch.load(join(self.model.runfolder, 'models', 'checkpoint' + model))
                    break
                except:
                    print('Best model (' + model + ') not in ' + modeldir + '...')
                    modeldir = ARGS.saved_model.split('models')[0]
                    print('... trying ' + modeldir)
            rim.load_state_dict(load)
            rim.cuda()
            for sigma in sigmas:
                masks = listdir(join(masksfolder, acc))
                to_log = np.zeros(t_max)
                for mask in masks:
                    for brain in listdir(braindir):
                        batch = next(iter(MRIData(
                            data_path=join(braindir, brain),
                            mask_path=join(masksfolder, acc, mask))))
                        target = np.absolute(batch['re_target'].squeeze().cpu().numpy() + 1j * batch[
                            'im_target'].squeeze().cpu().numpy()).astype(float)
                        loss, ests = rim.inference(
                            batch, t_max, get_loss=True, get_estimate=True, final=False, sigma=sigma)
                        for t, (l, e) in enumerate(zip(loss, ests)):
                            df = {}
                            df['time-step'] = [t + 1]
                            df['ssim'] = [structural_similarity(target, np.absolute(
                                e[0].data.squeeze().cpu().numpy() + 1j * e[1].data.squeeze().cpu().numpy()).astype(
                                float))]
                            df['sigma'] = [sigma]
                            df['acceleration'] = [float(acc.split('acc')[1])]
                            df['mask'] = [mask]
                            df['brain'] = [brain]
                            df['mse'] = [l]
                            df['train-step'] = [model]
                            df['model'] = [model_log.name]
                            df_ = pd.concat((df_, pd.DataFrame(df)))
        return df_
