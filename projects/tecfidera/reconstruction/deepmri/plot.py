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
    def __init__(self, log, name, train):
        self.log = log  # pandas df of loss values per time-step with training step as index
        self.name = name
        self.train = train
        self.x_steps = []

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
            self.best_model = int(self.log.index.values[argmin[self.best_time]])
        else:
            self.best_model = int(self.log.index.values[argmin[self.best_time]]) + train_step_lim[0]

    def extend(self, df, checkpoint_freq):
        if np.all(df.iloc[0, :] == range(df.shape[1])) or df.iloc[0, -1] == 'Avg.':
            df = df.iloc[1:, :]
        start = checkpoint_freq
        if len(self.x_steps) != 0:
            start += self.x_steps[-1]
        self.x_steps = np.concatenate((self.x_steps, np.linspace(start, start + (len(df) - 1) * checkpoint_freq, num=len(df))))
        self.log = pd.concat((self.log, df))



class ModelLog(object):
    def __init__(self, runfolder, train_path='./train_dir/', name=None, header=None, acceleration='all',
                 train_step_lim=(None, None), time_step_lim=(None, None), evalstats=None, multiple_datasets=False):

        if len(acceleration) == 1:
            if acceleration[0] < 4:
                acceleration = [4]
            elif acceleration[0] > 4 and acceleration[0] < 6:
                acceleration = [6]
            elif acceleration[0] > 6 and acceleration[0] < 8:
                acceleration = [8]
            elif acceleration[0] > 8 and acceleration[0] < 10:
                acceleration = [10]

        self.runfolder = runfolder
        self.train_path = train_path
        self.set_history(join(self.runfolder, 'models', 'checkpoint'))

        if name is None:
            self.name = runfolder
        else:
            self.name = name

        self.evalstats = evalstats
        self.test = {}

        strings = ['testloss_acc' + str(a) for a in acceleration]
        testruns = filter(lambda x: any(st in x for st in strings), listdir(self.train_path + '/' + self.runfolder))

        for testrun in testruns:
            acc = float(re.search('\d{1,2}', testrun).group(0))
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

    def set_history(self, saved_model):
        self.models = []
        self.checkpoints = []
        self.freqs = []
        while saved_model is not None and saved_model != 'None':
            model, checkpoint = saved_model.split('/models/checkpoint')
            self.checkpoints.append(checkpoint)
            model = join(self.train_path, split(model)[1])
            f = open(join(model, 'args_txt'), 'r')
            args = f.read()
            f.close()
            self.models.append(model)
            saved_model = args.split('saved_model : ')[1].split('\n')[0]
            self.freqs.append(int(args.split('checkpoint_freq : ')[1].split('\n')[0]))

    def get_best_model(self, acc, trainlim=(None, None), timelim=(None, None)):
        if trainlim[0] is not None or timelim[0] is not None:
            self.test[acc].set_best(train_step_lim=trainlim, time_step_lim=timelim)
        return self.test[acc].best_model

    def set_reconstruction_model(self, model):
        self.to_reconstruct = model

