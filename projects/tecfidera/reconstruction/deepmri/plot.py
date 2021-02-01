import pickle
import re
from os import listdir, makedirs
from os.path import join, exists, split

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

