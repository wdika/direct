import pickle
import re
from math import factorial
from os import listdir, makedirs
from os.path import join, exists, split

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    3     The Savitzky-Golay filter removes high frequency noise from data.
    4     It has the advantage of preserving the original shape and
    5     features of the signal better than other types of filtering
    6     approaches, such as moving averages techniques.
    7     Parameters
    8     ----------
    9     y : array_like, shape (N,)
    10         the values of the time history of the signal.
    11     window_size : int
    12         the length of the window. Must be an odd integer number.
    13     order : int
    14         the order of the polynomial used in the filtering.
    15         Must be less then `window_size` - 1.
    16     deriv: int
    17         the order of the derivative to compute (default = 0 means only smoothing)
    18     Returns
    19     -------
    20     ys : ndarray, shape (N)
    21         the smoothed signal (or it's n-th derivative).
    22     Notes
    23     -----
    24     The Savitzky-Golay is a type of low-pass filter, particularly
    25     suited for smoothing noisy data. The main idea behind this
    26     approach is to make for each point a least-square fit with a
    27     polynomial of high order over a odd-sized window centered at
    28     the point.
    29     Examples
    30     --------
    31     t = np.linspace(-4, 4, 500)
    32     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    33     ysg = savitzky_golay(y, window_size=31, order=4)
    34     import matplotlib.pyplot as plt
    35     plt.plot(t, y, label='Noisy signal')
    36     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    37     plt.plot(t, ysg, 'r', label='Filtered signal')
    38     plt.legend()
    39     plt.show()
    40     References
    41     ----------
    42     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
    43        Data by Simplified Least Squares Procedures. Analytical
    44        Chemistry, 1964, 36 (8), pp 1627-1639.
    45     .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
    46        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
    47        Cambridge University Press ISBN-13: 9780521880688
    48     """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:  # , msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


class LogPlotter(object):
    def __init__(
        self, log, name, train):
        # self.log = log.assign(name=pd.Series(name, copy=True, index=log.index).values)
        self.log = log  # pandas df of loss values per time-step with training step as index
        self.name = name
        self.train = train
        self.x_steps = []

    def __repr__(self):
        return 'Log of {}:\nMin. value: {} at iteration {}, time-step {}.'.format(
            self.name, self.best_loss, self.best_model, self.best_time + 1)

    def set_best(self, train_step_lim=(None, None), time_step_lim=(None, None)):
        # Limit the train- or time-steps from which the best model is determined
        trainlims = slice(*train_step_lim, None)
        timelims = slice(*time_step_lim, None)
        matlog = self.log.loc[trainlims].values.astype(float)[:, timelims]
        argmin = np.nanargmin(matlog, 0)
        minivals = np.nanmin(matlog, 0)
        self.best_loss = np.nanmin(minivals)
        if time_step_lim[0] is None:
            self.best_time = np.nanargmin(minivals)
        else:
            self.best_time = np.nanargmin(minivals) + time_step_lim[0]
        if train_step_lim[0] is None:
            self.best_model = int(
                self.log.index.values[argmin[self.best_time]])
        else:
            self.best_model = int(
                self.log.index.values[argmin[self.best_time]]) + train_step_lim[0]

    def extend(self, df, checkpoint_freq):
        if np.all(df.iloc[0, :] == range(
            df.shape[1])) or df.iloc[0, -1] == 'Avg.':
            df = df.iloc[1:, :]
        start = checkpoint_freq
        if len(self.x_steps) != 0:
            start += self.x_steps[-1]
        self.x_steps = np.concatenate((self.x_steps, np.linspace(
            start, start + (len(df) - 1) * checkpoint_freq, num=len(df))))
        self.log = pd.concat((self.log, df))

    def plot(self, name=None, step='best', color=['m'], ax=None,
             linestyle='-', savgol_params=[501, 3]):
        x_steps = [self.log.index]
        if name is None:
            name = self.name
        if step == 'best':
            df = self.log.iloc[:, self.best_time]
            step = [df]
            if df.isnull().values.any():
                nullinds = df[df.isnull()].index
                i = self.best_time - 1
                while self.log.loc[nullinds].iloc[:, i].isnull().values.any():
                    i -= 1
                    if i < 0:
                        break
                if i >= 0:
                    step.append(self.log.loc[nullinds].iloc[:, i])
                    x_steps.append(nullinds)

        if step == 'all':
            step = [self.log.iloc[:, i] for i in range(len(self.log.columns))]
            x_steps = x_steps * len(self.log.columns)
        if len(color) == 1:
            color = color * len(step)
        for x, s, c in zip(x_steps, step, color):
            if self.train:
                self.log.columns.name = 'time-step'
                train = np.squeeze(s.values)
                if savgol_params is not None:
                    train_smooth = savitzky_golay(train, *savgol_params)
                else:
                    train_smooth = train
                train_step_smooth = np.arange(len(train_smooth))
                train_step = np.arange(len(train))
                if ax is None:
                    plt.plot(train_step, train, alpha=.3, color=c, linestyle=linestyle)
                    plt.plot(train_step_smooth, train_smooth, color=c,
                             linestyle=linestyle, label=name)
                else:
                    ax.plot(train_step, train, alpha=.3, color=c, linestyle=linestyle)
                    ax.plot(train_step_smooth, train_smooth, color=c,
                            linestyle=linestyle, label=name)
            else:
                if ax is None:
                    plt.plot(x, s, color=c, linestyle=linestyle, label=name)
                else:
                    ax.plot(x, s, color=c, linestyle=linestyle, label=name)

    def heatmap(self, vmin=None, vmax=None, figsize=(20, 20), save=None):
        if self.train:
            self.log.columns.name = 'time-step'
        grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
        f, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=figsize)
        sns.heatmap(self.log, ax=ax, cmap=cc.m_inferno, cbar_ax=cbar_ax,
                    cbar_kws={"orientation": "vertical"}, vmin=vmin, vmax=vmax)
        ax.set_title(self.name)

        if save is not None:
            plt.savefig(join(save, self.name + '.png'), bbox_inches='tight')


class SavedModels(object):
    def __init__(self, saved_model):
        if saved_model is None or saved_model == 'None':
            self.model = None
            self.checkpoint = None
        else:
            (self.model, cp
             ) = saved_model.split('/models/')
            cp = cp.split('checkpoint')[1]
            if cp:
                self.checkpoint = int(cp)
            else:
                self.checkpoint = None
            with open(join(self.model, 'args_txt'), 'r') as f:
                args = f.read()
            previous = args.split(
                'saved_model : ')[1].split('\n')[0]
            self.checkpoint_freq = int(args.split(
                'checkpoint_freq : ')[1].split('\n')[0])
            self.previous = SavedModels(previous)

    def __iter__(self):
        model = self
        while model.model != 'None' and model.model is not None:
            yield model.model, model.checkpoint, model.checkpoint_freq
            model = model.previous


class ModelLog(object):
    def __init__(self, runfolder, train_path='./train_dir/', name=None, header=None, acceleration='all',
                 recursive=False,
                 train=False, train_step_lim=(None, None), time_step_lim=(None, None), evalstats=None,
                 multiple_datasets=False):
        self.multiple_datasets = multiple_datasets

        if len(acceleration) == 1:
            if acceleration[0] < 4:
                acceleration = [4]
            elif acceleration[0] > 4 and acceleration[0] < 6:
                acceleration = [6]
            elif acceleration[0] > 6 and acceleration[0] < 8:
                acceleration = [8]
            elif acceleration[0] > 8 and acceleration[0] < 10:
                acceleration = [10]

        if self.multiple_datasets is True:
            self.multi_runfolder = runfolder

        self.runfolder = runfolder

        self.train_path = train_path
        self.set_history(join(self.runfolder, 'models', 'checkpoint'))

        if name is None:
            self.name = runfolder
        else:
            self.name = name

        self.evalstats = evalstats

        if recursive and train:
            self.train = LogPlotter(pd.read_csv(join(self.runfolder, 'loss.csv'), header=header, index_col=0),
                                    self.name + ' train', True)

        self.test = {}

        if acceleration == 'all':
            strings = ['testloss']
        else:
            strings = ['testloss_acc' + str(a) for a in acceleration]

        testruns = filter(lambda x: any(st in x for st in strings), listdir(self.train_path + '/' + self.runfolder))
        # testruns = filter(lambda x: any(st in x for st in strings), self.runfolder)

        for testrun in testruns:
            acc = float(re.search('\d{1,2}', testrun).group(0))
            # acc = float(testrun.split('testloss_acc')[1].split('.csv')[0])
            self.test[acc] = LogPlotter(pd.DataFrame(), str(acc) + 'x ' + self.name, False)
            prev_cp = 0

            for model, checkpoint, freq in zip(reversed(self.models), reversed(self.checkpoints), reversed(self.freqs)):
                try:
                    df = pd.read_csv(join(model, testrun), header=header, index_col=0)
                except FileNotFoundError:
                    continue

                if checkpoint == '':
                    sl = slice(None, None, None)
                else:
                    cp = (int(checkpoint) - prev_cp) / freq
                    sl = slice(None, int(cp), None)
                    prev_cp = int(checkpoint)

                self.test[acc].extend(df.iloc[sl, :], freq)
            self.test[acc].set_best(train_step_lim=train_step_lim, time_step_lim=time_step_lim)

    def list_image_cache(self):
        path = join(self.runfolder, 'imgs')
        if exists(path):
            return listdir(path)
        else:
            return []

    def get_cached_image(self, imgname):
        with open(join(self.runfolder, 'imgs', imgname), 'rb') as f:
            img = pickle.load(f).squeeze().cpu().numpy()
        return img[..., 0] + 1j * img[..., 1]

    def set_history(self, saved_model):
        self.models = []
        self.checkpoints = []
        self.freqs = []
        while saved_model is not None and saved_model != 'None':
            model, checkpoint = saved_model.split('/models/checkpoint')
            self.checkpoints.append(checkpoint)
            try:
                if self.multiple_datasets:
                    model = join('/', model, self.multi_runfolder)
                f = open(join(model, 'args_txt'), 'r')
            except FileNotFoundError:
                model = join(self.train_path, split(model)[1])
                f = open(join(model, 'args_txt'), 'r')
            args = f.read()
            f.close()
            self.models.append(model)
            saved_model = args.split(
                'saved_model : ')[1].split('\n')[0]
            self.freqs.append(int(args.split(
                'checkpoint_freq : ')[1].split('\n')[0]))

    def heatmaps(self, plot=True, save=False, vmin=None, vmax=None, figsize=(20, 20)):
        for logg in (self.train, *self.test.values()):
            if save:
                path = join(self.runfolder, 'plots')
                if not exists(path):
                    makedirs(path)
            else:
                path = None
            logg.heatmap(vmin=vmin, vmax=vmax, figsize=figsize, save=path)
            if plot:
                plt.show()

    def plot(self, includetrain=False, savgol_params=[501, 3], acceleration='all', step='best', color='m', ax=None):
        if includetrain:
            self.train.plot(
                step=step, color=color, savgol_params=savgol_params)
        if acceleration == 'all':
            acceleration = sorted(self.test.keys())
        else:
            accs = sum([str(float(a)) for a in acceleration])
            acceleration = sorted(filter(lambda a: a in accs, self.test.keys()))
        for acc, line in zip(acceleration, ('-', ':', '-.', '--', '-')):
            self.test[acc].plot(step=step, color=[color], linestyle=line, ax=ax)

    def get_best_model(self, acc, trainlim=(None, None), timelim=(None, None)):
        if trainlim[0] is not None or timelim[0] is not None:
            self.test[acc].set_best(train_step_lim=trainlim, time_step_lim=timelim)
        return self.test[acc].best_model

    def set_reconstruction_model(self, model):
        self.to_reconstruct = model


class ModelPlotter(object):
    def __init__(self):
        self.models = {}
        self.colors = ['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:pink',
                       'xkcd:red', 'xkcd:light blue', 'xkcd:teal', 'xkcd:orange',
                       'xkcd:magenta', 'xkcd:yellow', 'xkcd:grey', 'xkcd:lime green']

    def add_model_run(self, *modellogs):
        for model in modellogs:
            if model.name in self.models.keys():
                ValueError('ModelLogs with name "{}" passed twice to ModelPlotter.add_model_run.')
            self.models[model.name] = model

    def print_best(self):
        for model in self.models.values():
            print(model.name)
            for a in sorted(model.test.keys()):
                print('Acceleration ' + str(a))
                print(repr(model.test[a]))

    def heatmaps(self, plot=True, save=True, vmin=None, vmax=None):
        for model in self.models.values():
            print(model.name)
            model.heatmaps(plot=plot, save=save, vmin=vmin, vmax=vmax)

    def plot_time_steps(self, model1, model2, acc, step='all', train=False, plot=True,
                        save=True, figsize=(10, 15), xlim=[0, 14000], ylim=[0.01, .03],
                        savgol_params=[501, 3]):
        plt.figure(figsize=figsize)
        if train:
            self.models[model1].train.plot(step=step, color=self.colors)
            self.models[model2].train.plot(step=step, color=self.colors)
        else:
            self.models[model1].test[float(acc)].plot(
                step=step, color=self.colors, linestyle=':')
            self.models[model2].test[float(acc)].plot(
                step=step, color=self.colors, linestyle='-.')
        leg = plt.legend(loc=1, frameon=True, fontsize=7)
        leg.get_frame().set_edgecolor('b')
        axes = plt.gca()
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if plot:
            plt.show()
        if save:
            plt.savefig()

    def on_motion(self, event):
        if mouse_event.inaxes != self.ax: return
        self.ax.add_patch()

    def plot_models(self, models=None, includetrain=False,
                    savgol_params=[501, 3], acceleration='all', step='best',
                    figsize=(10, 15), xlim=[0, 14000], ylim=[0.01, .03],
                    fontsize=10, loc=1, frameon=True):
        plt.figure(figsize=figsize)
        if models is None:
            models = self.models.keys()
        for model, c in zip(models, self.colors):
            self.models[model].plot(
                includetrain=includetrain, savgol_params=savgol_params,
                acceleration=acceleration, step=step, color=c)
        leg = plt.legend(loc=loc, frameon=frameon, fontsize=fontsize)
        leg.get_frame().set_edgecolor('b')
        axes = plt.gca()
        axes.set_ylim(ylim)
        axes.set_xlim(xlim)
        self.ax = axes
        self.cidmotion = self.ax.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        plt.show()


def factor_plot(models, x='time-step', hue='model', hue_order=None, size=5, aspect=1, row=None,
                data=None, palette=None, style='whitegrid', time_step=None, kind='point',
                capsize=.2, order=None, metric='ssim', window=None, dpi=600, fontsize=1,
                col=None, col_order=None, rename=None, ylim=None, wdir=None, font='Times New Roman',
                dodge=True, legend_out=False, legend=True, plot=True, linewidth=1, joined=True,
                fill_gaps=False, estimator='mean'
                ):
    estimator = {'mean': np.mean, 'median': np.median}[estimator]
    df = pd.DataFrame()

    def retrieve_csv(model):
        df = pd.DataFrame()
        for evalstats in model.evalstats:
            readcsv = pd.read_csv(join(model.runfolder, evalstats), header=0, index_col=False)
            if 'nmse' in readcsv.columns:
                readcsv.rename(columns={'nmse': 'nrmse'}, inplace=True)
            df = pd.concat((df, readcsv))
        df['model'] = pd.Series(model.name, index=df.index, copy=True)
        if time_step is None or time_step[0] == 'last':
            df = df[df['time-step'] == np.amax(df['time-step'])]
        else:
            df = df[df['time-step'].isin(time_step)]
        return df

    if isinstance(models, dict):
        for k, v in models.items():
            df_ = pd.DataFrame()
            for model in v:
                df_ = pd.concat((df_, retrieve_csv(model)))
            df_[k[0]] = pd.Series(k[1], index=df_.index, copy=True)
            df = pd.concat((df, df_))
    else:
        for model in models:
            if hasattr(model, 'evalstats') and isinstance(model.evalstats, pd.DataFrame):
                model.evalstats['model'] = pd.Series(model.name, index=model.evalstats.index, copy=True)
                df = pd.concat((df, model.evalstats))
            else:
                df = pd.concat((df, retrieve_csv(model)))

    df.loc[df['data'] == '7T_unet', 'data'] = 'Brain 0.7mm T2*'
    df.loc[df['data'] == '3T_unet', 'data'] = 'Brain 1.0mm T1'
    df.loc[df['data'] == '3T', 'data'] = 'Brain 1.0mm T1'
    df.loc[df['data'] == '7T_cfl', 'data'] = 'Brain 0.7mm T2*'
    df.loc[df['data'] == 'knee', 'data'] = 'Knee 0.5-0.6mm T2'
    df.loc[df['data'] == '32Cbrain7T', 'data'] = 'Brain 0.7mm T2*'
    df.loc[df['data'] == '31Cbrain3T', 'data'] = 'Brain 1.0mm T1'
    df.loc[df['data'] == '8Cknee', 'data'] = 'Knee 0.5-0.6mm T2'
    if data is not None:
        df = df[df['data'] == data]
    df[['ssim']] = df[['ssim']].apply(pd.to_numeric)
    df[['acceleration']] = df[['acceleration']].applymap(lambda s: str(s) + 'x')
    sns.set(style=style)
    sns.set_context('paper', rc={
        'lines.linewidth': linewidth, 'lines.markersize': 1, 'font.size': fontsize,
        'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize})
    if rename is not None:
        df.rename(columns=rename, inplace=True)
    kwargs = {'x': x, 'y': metric, 'palette': palette, 'hue': hue,
              'row': row, 'legend_out': legend_out, 'legend': legend,
              'data': df, 'kind': kind, 'ci': 'sd', 'col': col, 'estimator': estimator,
              'size': size, 'aspect': aspect, 'dodge': dodge, 'join': joined}
    if kind != 'box':
        kwargs['capsize'] = capsize
    if hue_order is not None:
        kwargs['hue_order'] = hue_order
    if col_order is not None:
        kwargs['col_order'] = col_order
    if order is not None:
        kwargs['order'] = order
    facetgrid = sns.factorplot(**kwargs)
    if not joined and fill_gaps:
        assert hue_order is not None
        for ax in facetgrid.axes.flat:
            for points, hue_name in zip(ax.collections, hue_order):
                # Retrieve the x axis positions for the points
                coords = list(zip(*points.get_offsets()))
                # Manually calculate the mean y-values to use with the line
                ax.plot(coords[0], coords[1], lw=2 * linewidth, color=palette[hue_name])
    if ylim is not None:
        facetgrid.set(ylim=ylim)
    if wdir is not None:
        facetgrid.savefig(wdir, dpi=dpi)
    if plot:
        plt.show()
    else:
        return facetgrid


if __name__ == '__main__':
    factor_plot(*(ModelLog('RV8T64F1C_3tesla_1subject', name='3T 1 subject'),
                  ModelLog('RV8T64F1C_3tesla_3subjects', name='3T 3 subjects'),
                  ModelLog('RV8T64F1C_3tesla_10subjects', name='3T 10 subjects'),
                  ModelLog('RV8T64F1C3', name='7T 1.5 subjects')),
                tesla=(3, 7), metric='ssim', size=10, aspect=2, kind='box', capsize=.2,
                col='acceleration', time_step=8, x='tesla', hue='model')
#	factor_plot(*(ModelLog('RV4T64F1C.4gamma1', name='RIM 4T 64F'), ModelLog('RV6T64F1C1', name='RIM 6T 64F'),
#				  ModelLog('RV8T64F1C3', name='RIM 8T 64F'), ModelLog('RV10T64F1C1', name='RIM 10T 64F'),
#				  ModelLog('RV12T64F1C', name='RIM 12T 64F')),
#				tesla=3, metric='ssim', size=10, aspect=2, kind='point', capsize=.2, col='acceleration', x='time-step')
#	factor_plot(*(ModelLog('RV8T32F1C', name='32F'), ModelLog('RV8T44F1C1', name='44F'),
#				  ModelLog('RV8T64F1C3', name='64F'), ModelLog('RV8T128F1C5', name='128F')),
#				x='model', tesla=3, kind='point', metric='ssim', hue='acceleration', time_step=8, size=10, aspect=2)
#	factor_plot(*(ModelLog('RV8T64F1C1', name='RIM 8T 64F log-weight'),  ModelLog('RV8T64F1C3', name='RIM 8T 64F uni-weight')),
#				tesla=3, x='acceleration', metric='ssim', size=10, aspect=2, kind='point', time_step=8, hue='model')
#	factor_plot(*(ModelLog('RV8T64F1C_3tesla_1subject', name='1 subject'),
#				  ModelLog('RV8T64F1C_3tesla_3subjects', name='3 subjects'),
#				  ModelLog('RV8T64F1C_3tesla_10subjects', name='10 subjects')),
#				x='acceleration', metric='ssim', size=10, aspect=2, kind='point', hue='model', time_step=8)
