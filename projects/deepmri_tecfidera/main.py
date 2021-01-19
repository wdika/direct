import argparse
import pickle
import re
from os import makedirs, listdir
from os.path import join, exists, split

import dill
import numpy as np
import torch
import torch.utils.data

from plot import ModelLog
from rim import Rim
from visualize import plot_reconstructions

models = {'rim': Rim}


def get_network(model, loaded=False, device='cuda'):
    with open(join(model.runfolder, 'args'), 'rb') as f:
        ARGS = dill.load(f)

    conv_params = {'f': ARGS.conv_features, 'k': ARGS.conv_kernels, 'd': ARGS.conv_dilations}
    rnn_params = {'f': ARGS.rnn_features, 'k': [[k1, k2] for k1, k2 in zip(
        ARGS.rnn_kernels[::2], ARGS.rnn_kernels[1::2])], 'd': [[d1, d2] for d1, d2 in zip(
        ARGS.rnn_dilations[::2], ARGS.rnn_kernels[1::2])]}

    network = models[ARGS.model](t_max=ARGS.t_max, act_fn=ARGS.act_fn,
                                 n_feature=ARGS.n_feature if type(ARGS.n_feature) != int else [ARGS.n_feature],
                                 recurrent=ARGS.recurrent,
                                 conv_params=conv_params, rnn_params=rnn_params).to(dtype=getattr(torch, ARGS.dtype))

    if loaded:
        savedmodel = join(model.runfolder, 'models', model.to_reconstruct)
        while savedmodel is not None and savedmodel is not 'None':
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
        with open(FLAGS.mask_dir[0], 'rb') as f:
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


def run():
    models = []
    for network, name, gridloc in zip(FLAGS.modeldir[::3], FLAGS.modeldir[1::3], FLAGS.modeldir[2::3]):
        models.append(get_model_log(network, name, gridloc))

    with torch.no_grad():
        plot_reconstructions(lambda x: get_network(x, True, device=device), models, {'data_path': FLAGS.mri_dir,
                                                                                     'mask_path': FLAGS.mask_dir},
                             t_max=FLAGS.t_max, notitles=FLAGS.notitles, mode=FLAGS.mode,
                             crop_box=FLAGS.crop_box, cmap=FLAGS.cmap, inset=FLAGS.inset, zoom_factor=FLAGS.zoom_factor,
                             locs=FLAGS.locs,
                             boxcolor=FLAGS.boxcolor, edgecolor=FLAGS.edgecolor,
                             interpolation=FLAGS.interpolation, grid=FLAGS.grid, label=FLAGS.label,
                             figsize=FLAGS.figsize,
                             gridhwspace=FLAGS.gridhwspace, gridlrbt=FLAGS.gridlrbt, min_clim=FLAGS.min_clim,
                             max_clim=FLAGS.max_clim,
                             textloc=FLAGS.textloc, set_max_clim=FLAGS.set_max_clim, metric=FLAGS.metric,
                             fontname=FLAGS.fontname, fontsize=FLAGS.fontsize, fsc=None, rotate=FLAGS.rotate90,
                             targetsuffix=FLAGS.targetsuffix, generate_new=FLAGS.generate_new, error=FLAGS.error,
                             inputscale=FLAGS.inputscale, fontoutlinecolor=FLAGS.fontoutlinecolor)


def mask():
    from data import Masker
    import matplotlib.pyplot as plt
    import pickle
    for acc in FLAGS.acceleration:
        masker = Masker(
            FLAGS.sampling_dist, None, acc,
            FLAGS.fwhms, FLAGS.center_scale, FLAGS.superres,
            FLAGS.superresscale, FLAGS.superresprob)
        if FLAGS.likethis is not None:
            with open(FLAGS.likethis, 'rb') as f:
                tmp = pickle.load(f)
            if tmp.ndim == 4:
                shape = tmp.shape[2:]
            else:
                shape = tmp.shape
        else:
            shape = FLAGS.shape
        for i in map(str, range(FLAGS.nmasks)):
            mask = masker(tuple(shape))
            if FLAGS.plot:
                plt.imshow(np.fft.fftshift(mask))
                plt.show()
            if FLAGS.wdir is not None:
                wdir = join(FLAGS.wdir, 'acc' + str(np.round(acc, 0)))
                if not exists(wdir):
                    makedirs(wdir)
                with open(join(wdir, FLAGS.name + i), 'wb') as f:
                    pickle.dump(mask, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='mode')

    runner = subparsers.add_parser('run', help='Reconstruct an image slice')
    runner.set_defaults(func=run)

    masker = subparsers.add_parser('mask', help='Show sub-sampling patterns generated for training, store if needed '
                                                'using wdir argument')
    masker.set_defaults(func=mask)

    masker.add_argument(
        'acceleration', type=float, nargs='+',
        help='The acceleration factor. The number of samples '
             'drawn will be floor(shape[0]*shape[1]/acceleration).')
    masker.add_argument(
        '--likethis',
        help='Path to mask of same size as the one you want to make, '
             'valid alternative to --shape argument.')
    masker.add_argument(
        '--shape', nargs=2, type=int,
        help='The shape of the 2D image to mask (tuple of ints)')
    masker.add_argument(
        '--sampling_dist', default='gaussian',
        choices=['gaussian', 'gaussian1d', 'periodic', 'periodic1d', 'poisson', 'poisson1d'],
        help='The distribution to use when sampling K-space. Adding "1d" to the end '
             'simulates data acquisitioned as horizontal lines on a 2D-plane, meaning '
             'samples are drawn as individual lines from a vertical 1D-line. Without '
             'the "1d"-suffix, data is acquisitioned as points on the 2D-plane, simulating '
             'lines sampled from a 3D volume going inwards.\nDefaults to "gaussian".'
    )
    masker.add_argument(
        '--fwhms', nargs=2, type=float, default=(.7, .7),
        help='The full-width at half maxima of the distribution along the two axes')
    masker.add_argument(
        '--center_scale', type=float, default=.02,
        help='The proportion of K-space values to sample near the origin (regardless of'
             ' the acceleration level). Will be elliptical for 2D-distributions, and rectangular'
             ' for 1D-distributions.')
    masker.add_argument(
        '--plot', action='store_true', help='Toggle to plot the masks created')
    masker.add_argument(
        '--wdir', help='Path to store mask, including file name')
    masker.add_argument(
        '--nmasks', type=int, default=1, help='How many masks to create')
    masker.add_argument(
        '--superres', action='store_true', help='Toggle to make mask for super-resolution tasks')
    masker.add_argument(
        '--superresscale',
        type=float,
        help='The ellipse scale to use for downsampling in K-space during super resolution training/testing'
    )
    masker.add_argument(
        '--superresprob',
        type=float,
        default=.3333,
        help='Probability of subsampling interior ellipse during super resolution training/testing'
    )
    masker.add_argument(
        '--name',
        default='mask',
        help='Name of mask files')

    runner.add_argument(
        '--mri_dir',
        nargs='+',
        help='Path to MR-image data, stored as a pickle dump')
    runner.add_argument(
        '--mask_dir',
        nargs='+',
        help='Path to mask data, stored as a pickle dump, center of K-space '
             'in the corners, following the numpy.fft convention')
    runner.add_argument(
        '--modeldir',
        nargs='+',
        help='Path to stored parameters, should be contained in a "models"-folder within a '
             'folder containing everything relevant to a specific train run, including an '
             '"args"-file of arguments stored with dill.')
    runner.add_argument(
        '--fname',
        help='File name to save image as. Image is saved into imgs-folder of'
             'all model run folders called via --modeldir parameter.')
    runner.add_argument(
        '--wdir',
        help='Enter path and file name of image to store. Stores image at this location only.')
    runner.add_argument(
        '--plot',
        action='store_true',
        help='Toggle to plot image')
    runner.add_argument(
        '--all_times',
        action='store_true',
        help='Toggle to visualize all time-steps up until t_max.'
    )
    runner.add_argument(
        '--inputscale',
        type=float,
        default=1.0,
        help='A float to scale the input with, defaults to 1.0.')
    runner.add_argument(
        '--mode',
        default='absolute',
        help='Which complex representation to plot'
    )
    runner.add_argument(
        '--notitles',
        action='store_true',
        help='Toggle to turn off subplot titles')
    runner.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[40, 40],
        help='Figure size of image to make'
    )
    runner.add_argument(
        '--crop_box',
        nargs=4,
        type=int,
        help='left right bottom top coordinates of box to crop, '
             'defaults to the entire image')
    runner.add_argument(
        '--generate_new',
        action='store_true',
        help='Toggle to generate new images instead of using '
             'already generated images in "imgs" folder of the selected models.')
    runner.add_argument(
        '--metric',
        choices=['', 'ssim', 'mse', 'nrmse', 'psnr'],
        help='Type of metric to use for quality assesment used for image text')
    runner.add_argument(
        '--inset',
        nargs=4,
        type=int,
        help='left right bottom top coordinates of zoomed inset, '
             'defaults to no zoomed inset')
    runner.add_argument(
        '--cmap',
        default='gray',
        help='Name of colorcet cmap to use for plotting')
    runner.add_argument(
        '--zoom_factor',
        type=float,
        default=3.,
        help='The zoom factor for the zoomed inset, defaults to 3x.')
    runner.add_argument(
        '--locs',
        nargs=3,
        type=int,
        default=[1, 2, 4],
        help='List of three locations (matplotlib loc-keyword), first for the '
             'zoomed inset, the remaining two elements for the corners to attach lines '
             'between inset and zoom'
    )
    runner.add_argument(
        '--fsc',
        action='store_true',
        help='Toggle to plot Fourier Shell Correlation')
    runner.add_argument(
        '--edgecolor',
        default='none',
        help='Color to use for lines connecting inset and zoom corners')
    runner.add_argument(
        '--boxcolor',
        default='white',
        help='Color to use for inset and zoom bounding boxes')
    runner.add_argument(
        '--fontoutlinecolor',
        default='black',
        help='Color of text outline')
    runner.add_argument(
        '--interpolation',
        default='nearest',
        help='Interpolation keyword passed to imshow')
    runner.add_argument(
        '--grid',
        nargs=2,
        type=int,
        default=[1, 3],
        help='The number of rows and columns for the subplot grid')
    runner.add_argument(
        '--error',
        choices=['row', 'col'],
        help="'row' or 'col' for plotting error along every other row or column, respectively.")
    runner.add_argument(
        '--gridhwspace',
        nargs=2,
        type=float,
        default=[0., 0.],
        help='Space between images in vertical and horizontal direction, respectively.')
    runner.add_argument(
        '--gridlrbt',
        nargs=4,
        type=float,
        default=[1. / 8, 7. / 8, 0., 1.],
        help='Spacing around image, left right bottom top')
    runner.add_argument(
        '--gifinterval',
        type=int,
        default=300,
        help='Number of milliseconds to show each time frame in the gif')
    runner.add_argument(
        '--gifrepeatdelay',
        type=int,
        default=500,
        help='Number of milliseconds to pause before repeating gif')
    runner.add_argument(
        '--rotate90',
        type=int,
        default=0,
        help='Number of times to rotate image 90 degrees before plotting')
    runner.add_argument(
        '--label',
        default='text',
        choices=['text', 'title'],
        help='To add model name and ssim as titles or as text on top of image')
    runner.add_argument(
        '--textloc',
        nargs=2,
        type=int,
        default=[1, 7],
        help='The location to place image label text at in x- y-coordinates')
    runner.add_argument(
        '--nocuda',
        action='store_true',
        help='Toggle to use cpu')
    runner.add_argument(
        '--fontsize',
        type=float,
        default=8,
        help='Size of font of image text')
    runner.add_argument(
        '--fontname',
        default='helvetica',
        help='Name of font to use')
    runner.add_argument(
        '--t_max',
        type=int,
        help='What time-step to show.')
    runner.add_argument(
        '--min_clim',
        type=float,
        default=0.,
        help='The minimum saturation level')
    runner.add_argument(
        '--max_clim',
        type=float,
        default=1.,
        help='The maximum saturation level'
    )
    runner.add_argument(
        '--targetsuffix',
        default='',
        help='Add text suffix to "Target "-title')
    runner.add_argument(
        '--set_max_clim',
        action='store_true',
        help='Toggle to interactively set the saturation level using histogram plots.'
    )
    runner.add_argument(
        '--dpi',
        type=float,
        default=300,
        help='Set the DPI of the saved image'
    )
    runner.add_argument(
        '--multiple_datasets_dir',
        nargs='+',
        help='Datasets names of the model trained on, if trained on multiple datasets.'
    )

    FLAGS, _ = parser.parse_known_args()
    try:
        device = torch.device('cpu' if FLAGS.nocuda else 'cuda')
    except AttributeError:
        device = torch.device('cuda')
    FLAGS.func()
