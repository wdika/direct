from os.path import basename, join, exists

import colorcet as cc
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from skimage.measure import compare_mse, compare_nrmse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from projects.deepmri_tecfidera.data import MRIData

_mode = {
    'absolute': lambda x: np.clip(np.absolute(x).astype(float), 0, 1),
    'phase': lambda x: np.angle(x).astype(float),
    'real': lambda x: np.clip(x.real.astype(float), -1, 1),
    'imag': lambda x: np.clip(x.imag.astype(float), -1, 1)
}

_metric = {
    'ssim': lambda t, e, d: ' | SSIM: {}'.format(np.around(compare_ssim(t, e, data_range=1), 3)),
    'mse': lambda t, e, d: ' | MSE: {}'.format(np.around(compare_mse(t, e), 6)),
    'nrmse': lambda t, e, d: ' | NRMSE: {}'.format(np.around(compare_nrmse(t, e), 3)),
    'psnr': lambda t, e, d: ' | PSNR: {}'.format(
        np.around(compare_psnr(t.astype(float), e.astype(float), data_range=1), 3)),
    None: lambda *e: ''
}


def mrimshow(ax, img, title, fourier_shell, cmap, min_clim, max_clim, set_max_clim, crop_box, zoom_factor, label,
             interpolation, boxcolor, inset, locs, edgecolor, fontoutlinecolor, textloc, fontsize, fontname, rotate):
    img = np.rot90(img, rotate)
    if crop_box is not None:
        h, w = img.shape
        l, r, b, t = [elem if elem >= 0 else 0 for elem in crop_box]
        img = img[t:h - b, l:w - r]
        l, r, b, t = [abs(elem) if elem < 0 else 0 for elem in crop_box]
        h, w = img.shape
        w = w + l + r
        img = np.concatenate((np.zeros((t, w)), np.concatenate((np.zeros((h, l)), img, np.zeros((h, r))), 1),
                              np.zeros((b, w))), 0)

    h, w = img.shape

    if set_max_clim:
        climfig, histax = plt.subplots()
        histax.hist(img.ravel(), bins=256, range=(.0, 1.0), fc='k', ec='k')
        histax.set_title(title)
        climfig.show()
        maxclim = float(input('Set maximum clim value: ') or 1.)
    else:
        maxclim = max_clim
    if 'Error' in title:
        maxclim = .1 * max_clim

    # h, w = 1, 1
    ax = plt.subplot(ax)
    ax.imshow(img, cmap=cmap, extent=[0, w, 0, h], clim=(min_clim, maxclim), aspect=h / w, interpolation=interpolation)
    ax.set_xticks([])
    ax.set_yticks([])

    if label == 'text':
        text = ax.text(textloc[0], h - textloc[1], title, fontsize=fontsize, name=fontname, color=boxcolor)
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground=fontoutlinecolor), path_effects.Normal()])

    if label == 'title':
        ax.set_title(title, fontsize=fontsize)

    if inset is not None:
        axins = zoomed_inset_axes(ax, zoom_factor, loc=locs[0])
        for spine in axins.spines.values():
            spine.set_color(boxcolor)
        axins.imshow(img, cmap=cmap, extent=[0, w, 0, h], aspect=h / w, interpolation=interpolation,
                     clim=(min_clim, maxclim))
        axins.set_xlim(*inset[:2])
        axins.set_ylim(*inset[2:])
        plt.xticks([])
        plt.yticks([])
        pp, p1, p2 = mark_inset(ax, axins, loc1=locs[1], loc2=locs[2])
        p1.set_edgecolor(edgecolor)
        p2.set_edgecolor(edgecolor)
        pp.set_edgecolor(boxcolor)

    if fourier_shell[0] is not None:
        fourier_shell[0].plot(fourier_shell[1], '-', color=fourier_shell[2], label=title.split('|')[0])
        fourier_shell[0].legend(loc='upper left')


def plot_reconstructions(loader, models, data_kwargs, mode='absolute', t_max=None, figsize=(30, 20), crop_box=None,
                         cmap='gray', inset=None, zoom_factor=3, locs=[1, 2, 4], boxcolor='white', label='text',
                         fontsize=8,
                         interpolation='nearest', notitles=False, edgecolor='none', grid=(1, 3), gridhwspace=[0., 0.],
                         textloc=[1, 1], targetsuffix='', gridlrbt=[1. / 8, 7. / 8, 1., 1.], error=None,
                         fontname='helvetica',
                         min_clim=0, max_clim=1., set_max_clim=False, fontoutlinecolor='black', rotate=False, metric='',
                         fsc=None,
                         generate_new=False, inputscale=1.0):
    l, r, b, t = gridlrbt
    hs, ws = gridhwspace
    if error == 'row':
        grid = 2 * grid[0], grid[1]
        error = True
    if error == 'col':
        error = True
        grid = grid[0], 2 * grid[1]
    gs = gridspec.GridSpec(*grid, wspace=ws, hspace=hs, top=t, bottom=b, left=l, right=r)
    gs.update(wspace=ws, hspace=hs)

    isgif = False
    plotdicts = [{}]

    modes = mode.split()
    mode = {}
    cmaps = {}
    max_clims = {}
    min_clims = {}
    if len(modes) > 1:
        for m, integers in zip(modes[::2], modes[1::2]):
            for i in range(*map(int, integers.split('-'))):
                mode.update({i: _mode[m]})
                if m == 'phase':
                    max_clims.update({i: np.pi})
                    min_clims.update({i: -np.pi})
                    if 'cyclic' not in cmap:
                        cmaps.update({i: cc.cm['cyclic_grey_15_85_c0']})
                    else:
                        cmaps.update({i: cmap})
                else:
                    max_clims.update({i: max_clim})
                    min_clims.update({i: min_clim})
                    cmaps.update({i: cmap})
    else:
        for i in range(grid[0] * grid[1]):
            mode.update({i: _mode[modes[0]]})
            if modes[0] == 'phase':
                max_clims.update({i: np.pi})
                min_clims.update({i: -np.pi})
                if 'cyclic' not in cmap:
                    cmaps.update({i: cc.cm['cyclic_grey_15_85_c0']})
                else:
                    cmaps.update({i: cc.cm[cmap]})
            else:
                max_clims.update({i: max_clim})
                min_clims.update({i: min_clim})
                cmaps.update({i: cc.cm[cmap]})

    metric = _metric[metric]

    target_np = MRIData(data_path=data_kwargs['data_path'][0], mask_path=data_kwargs['mask_path'][0],
                        inputscale=inputscale).get_loaded_target()

    if fsc is None:
        fsc_fig, fsc_ax = None, None
        fourier_shells = lambda x: None
        colorpalette = (None for _ in range(len(models) * len(data_kwargs['mask_path']) + 1))

    tlocs = map(int, data_kwargs['data_path'][1].split())

    for tloc in tlocs:
        if tloc < 0:
            target = mode[0](target_np)
        else:
            target = mode[tloc](target_np)

        target = np.abs(target).astype(np.float32)
        target = np.clip(target / np.max(target), 0, 1)

        if tloc >= 0:
            for pdict in plotdicts:
                pdict.update(
                    {gs[tloc // grid[1], tloc % grid[1]]: (target, '' if notitles else 'Target ' + targetsuffix,
                                                           (None, None, None), cmaps[tloc], min_clims[tloc],
                                                           max_clims[tloc])})

    imshow = lambda ax, img, title, fsc_ax, cmap, min_clim, max_clim: mrimshow(ax, img, title, fsc_ax, cmap, min_clim,
                                                                               max_clim, set_max_clim, crop_box,
                                                                               zoom_factor, label, interpolation,
                                                                               boxcolor, inset, locs, edgecolor,
                                                                               fontoutlinecolor, textloc, fontsize,
                                                                               fontname, rotate)

    def frameplotter(pltdict):
        for k, v in pltdict.items():
            imshow(k, *v)

    for mask, etaloc in zip(data_kwargs['mask_path'][::2], map(int, data_kwargs['mask_path'][1::2])):
        batch = next(iter(MRIData(data_path=data_kwargs['data_path'][0], mask_path=mask, inputscale=inputscale)))
        acc = np.around(batch['mask'].numel() / torch.sum(batch['mask']))

        if etaloc != -1:
            eta_mode = batch['eta'].squeeze().numpy()
            eta_mode = np.abs(eta_mode[..., 0] + 1j * eta_mode[..., 1])
            eta_mode = np.clip(eta_mode / np.max(eta_mode), 0, 1)

            plotdicts[0].update(
                {gs[etaloc // grid[1], etaloc % grid[1]]: (eta_mode, '' if notitles else '{}x'.format(int(acc)),
                                                           (fsc_ax, '', next(colorpalette)), cmaps[etaloc],
                                                           min_clims[etaloc], max_clims[etaloc])})

        pics_mode = np.abs(batch['pics'].squeeze().numpy())
        pics_mode = np.clip(pics_mode / np.max(pics_mode), 0, 1)

        plotdicts[0].update({gs[(etaloc + 1) // grid[1], (etaloc + 1) % grid[1]]: (pics_mode, 'CS', (
            fsc_ax, '', next(colorpalette)), cmaps[etaloc], min_clims[etaloc], max_clims[etaloc])})

        for model in models:
            gl = next(model.gridloc)

            if t_max is None:
                try:
                    T = model.t_max
                except AttributeError:
                    T = 1
            else:
                T = t_max

            if model.to_reconstruct == 'CS':
                fname = model.name
            else:
                fname = join(model.runfolder, 'imgs', '{}x_{}t_{}_{}_{}scale'.format(str(acc.item()), str(T), basename(
                    data_kwargs['data_path'][0]), model.to_reconstruct, inputscale))

            if not exists(fname) or generate_new or isgif:
                network = loader(model)
                imgs = next(network.inference(batch, T))

            if not isinstance(imgs, list):
                i = 0
                imgs = [imgs]
            else:
                i = 1

            for img, pdict in zip(imgs, plotdicts[i:]):
                img_np = img.squeeze().cpu().numpy()
                img_np = img_np[..., 0] + 1j * img_np[..., 1]

                im_mode = np.abs(mode[gl](img_np)).astype(np.float32)
                im_mode = np.clip(im_mode / np.max(im_mode), 0, 1)

                pdict.update({gs[gl // grid[1], gl % grid[1]]: (im_mode, '' if notitles else '{}'.format(model.name),
                                                                (fsc_ax, '', next(colorpalette)), cmaps[gl],
                                                                min_clims[gl], max_clims[gl])})

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('black')
    frameplotter(plotdicts[0])
    fig.subplots_adjust(hspace=0, wspace=0)
    gs.update(hspace=0, wspace=0)
    plt.show()
