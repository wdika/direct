import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': 'Arial'})


def retrieve_csv(csv_path, time_step):
    df = pd.DataFrame()

    readcsv = pd.read_csv(csv_path, header=0, index_col=False)
    df = pd.concat((df, readcsv))

    if time_step is None or time_step[0] == 'last':
        df = df[df['time-step'] == np.amax(df['time-step'])]
    else:
        df = df[df['time-step'].isin(time_step)]

    return df


def cat_plot(csv_path='', x='time-step', hue='method', hue_order=None, height=5, aspect=1, row=None,
             palette=None, style='whitegrid', time_step=None, kind='point', capsize=.2, order=None, metric='SSIM',
             fontsize=1, col=None, col_order=None, rename=None, ylim=None, dodge=True, legend_out=False, legend=True,
             linewidth=1, joined=True, fill_gaps=False, estimator='mean', save_plot=False):
    estimator = {'mean': np.mean, 'median': np.median}[estimator]
    df = retrieve_csv(csv_path, ['last'])

    df.sort_values(by=[metric], ascending=[False], inplace=True)

    df[[metric]] = df[[metric]].apply(pd.to_numeric)
    df[['acc']] = df[['acc']].applymap(lambda s: str(s) + 'x')

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
              'height': height, 'aspect': aspect, 'dodge': dodge}

    if kind != 'box':
        kwargs['join'] = joined
        kwargs['capsize'] = capsize

    if hue_order is not None:
        kwargs['hue_order'] = hue_order

    if col_order is not None:
        kwargs['col_order'] = col_order

    if order is not None:
        kwargs['order'] = order

    facetgrid = sns.catplot(**kwargs)

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

    if save_plot:
        save_path = '/'.join(str(csv_path).split('/')[:-1]) + '/plots/' + metric
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        acc = 'all' if len(order) > 1 else order[0]
        plt.savefig(save_path + '/' + str(csv_path).split('/')[-1].split('.')[0].split('_')[0] + '_' + acc + '.png')

    plt.show()


def main():
    reds = sns.color_palette('Reds', n_colors=8)
    greens = sns.color_palette('Greens', n_colors=8)
    blues = sns.color_palette('Blues', n_colors=8)
    purples = sns.color_palette('Purples', n_colors=1)
    PuBuGn_d = sns.color_palette('PuBuGn_d', n_colors=1)

    """
    palette = {
        'CS': purples[0], 'zero_filled': PuBuGn_d[0],
        'GRIM_3T_T1_3D_Brains_l2': reds[0], 'GRIM_3T_T1_3D_Brains_l1': reds[4],
        'GRIM_7T_T2star_6echoes_3D_Brains_l2': reds[1], 'GRIM_7T_T2star_6echoes_3D_Brains_l1': reds[5],
        'GRIM_3T_PD_FSE_3D_Knees_l2': reds[2], 'GRIM_3T_PD_FSE_3D_Knees_l1': reds[6],
        'GRIM_Combined_l2': reds[3], 'GRIM_Combined_l1': reds[7],
        'MRIM_3T_T1_3D_Brains_l2': greens[0], 'MRIM_3T_T1_3D_Brains_l1': greens[4],
        'MRIM_7T_T2star_6echoes_3D_Brains_l2': greens[1], 'MRIM_7T_T2star_6echoes_3D_Brains_l1': greens[5],
        'MRIM_3T_PD_FSE_3D_Knees_l2': greens[2], 'MRIM_3T_PD_FSE_3D_Knees_l1': greens[6],
        'MRIM_Combined_l2': greens[3], 'MRIM_Combined_l1': greens[7],
        'IRIM_3T_T1_3D_Brains_l2': blues[0], 'IRIM_3T_T1_3D_Brains_l1': blues[4],
        'IRIM_7T_T2star_6echoes_3D_Brains_l2': blues[1], 'IRIM_7T_T2star_6echoes_3D_Brains_l1': blues[5],
        'IRIM_3T_PD_FSE_3D_Knees_l2': blues[2], 'IRIM_3T_PD_FSE_3D_Knees_l1': blues[6],
        'IRIM_Combined_l2': blues[3], 'IRIM_Combined_l1': blues[7],
    }
    """

    palette = {
        'CS': purples[0], 'zero_filled': PuBuGn_d[0],
        'GRIM_T1_Brains_l2': reds[0], 'GRIM_T1_Brains_l1': reds[4],
        'GRIM_T2star_Brains_l2': reds[1], 'GRIM_T2star_Brains_l1': reds[5],
        'GRIM_T2_Knees_l2': reds[2], 'GRIM_T2_Knees_l1': reds[6],
        'GRIM_Combined_l2': reds[3], 'GRIM_Combined_l1': reds[7],
        'MRIM_T1_Brains_l2': greens[0], 'MRIM_T1_Brains_l1': greens[4],
        'MRIM_T2star_Brains_l2': greens[1], 'MRIM_T2star_Brains_l1': greens[5],
        'MRIM_T2_Knees_l2': greens[2], 'MRIM_T2_Knees_l1': greens[6],
        'MRIM_Combined_l2': greens[3], 'MRIM_Combined_l1': greens[7],
        'IRIM_T1_Brains_l2': blues[0], 'IRIM_T1_Brains_l1': blues[4],
        'IRIM_T2star_Brains_l2': blues[1], 'IRIM_T2star_Brains_l1': blues[5],
        'IRIM_T2_Knees_l2': blues[2], 'IRIM_T2_Knees_l1': blues[6],
        'IRIM_Combined_l2': blues[3], 'IRIM_Combined_l1': blues[7],
    }

    order = ['4x', '6x', '8x', '10x'] if args.order == 'all' else args.order

    cat_plot(csv_path=args.csv_path, x=args.x_axis, hue=args.hue, height=8, palette=palette, style='whitegrid',
             time_step=args.t_max, kind='box', order=order, metric=args.metric, fontsize=14, legend_out=True,
             ylim=args.y_lim,
             rename={'acc': 'Acceleration', args.metric.lower(): args.metric, 'model': 'Model'},
             save_plot=args.save_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv_path', type=pathlib.Path, required=True, help='Path to the csv file.')
    parser.add_argument('--save_plot', action='store_true', help='Toggle to save the plot')
    parser.add_argument('--t_max', default=[8], type=int, help='Maximum number of time-steps for the RIM.')
    parser.add_argument('--hue', type=str, default='Model', help='Hue value. Default: "Model"')
    parser.add_argument('--x_axis', type=str, default='Acceleration', help='x-axis attribute. Default: "Acceleration"')
    parser.add_argument('--metric', type=str, choices=['MSE', 'NMSE', 'PSNR', 'SSIM'], default='SSIM',
                        help='Metric to evaluate')
    parser.add_argument('--order', default='all', type=str, nargs='+', help='Order x-axis by value.')
    parser.add_argument('--y_lim', default=None, type=float, help='Value to limit the y axis.')

    args = parser.parse_args()

    if len(args.t_max) == 1:
        args.t_max = 2 * args.t_max

    main()
