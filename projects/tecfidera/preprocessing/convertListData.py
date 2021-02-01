# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import glob
import logging
import time
from multiprocessing import Process

from projects.tecfidera.preprocessing.utils import *
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocessing(root, output, export_type, device):
    """
    Parses all subjects, acquisitions, and scans. Performs all the necessary preprocessing steps for the TECFIDERA data.

    Parameters
    ----------
    root :   root directory of containing cfl data
    output : output directory to save data
    """
    subjects = glob.glob(root + "/*/")
    for subject in tqdm(subjects):
        acquisitions = glob.glob(subject + "/*/")
        for acquisition in acquisitions:
            data = glob.glob(acquisition + "*.list")
            for filename in data:
                name = filename.split('.')[0].split('/')[-1].split('_')[1]

                if name != '501':  # exclude the sense ref scan
                    name = 'AXFLAIR' if name == '301' else 'AXT1_MPRAGE'

                    logger.info(
                        f"Processing subject: {subject.split('/')[-2]} | time-point: {acquisition.split('/')[-2]} "
                        f"| acquisition: {name}")

                    kspace_path = filename.split('.')[0]
                    raw_kspace = get_kspace_from_listdata(kspace_path)
                    if raw_kspace is not None:
                        sense_path = kspace_path.split('_')[0] + '_501'
                        sense = get_kspace_from_listdata(sense_path) if os.path.isfile(sense_path
                                                                           ) else get_kspace_from_listdata(kspace_path)

                        kspace, mask, imspace, sensitivity_map = preprocess_volume(kspace=raw_kspace, sense=sense,
                            slice_range=None, device=device)
                        del raw_kspace, kspace_path, sense_path, sense, filename
                        torch.cuda.empty_cache()

                        if export_type == 'png':
                            output_dir = output + 'png/' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                                -2] + '/' + name
                            create_dir(output_dir)

                            # Save target (SENSE reconstructed) png images
                            Process(target=save_png_outputs, args=(
                                complex_tensor_to_real_np(sense_reconstruction(imspace, sensitivity_map, dim=-1)),
                                output_dir + '/targets/')).start()

                            # Save mask
                            plt.imshow(mask, cmap='gray')
                            plt.savefig(output_dir + '/mask.png')
                            plt.close()

                            # Save sense coil combined png images
                            Process(target=save_png_outputs, args=(
                                complex_tensor_to_real_np(rss_reconstruction(sensitivity_map, dim=-1)),
                                output_dir + '/csms/')).start()

                        elif export_type == 'h5':
                            name = subject.split('/')[-2] + '_' + acquisition.split('/')[-2] + '_' + name

                            output_dir = output + '/kspaces/'
                            create_dir(output_dir)
                            Process(target=save_h5_outputs,
                                    args=(complex_tensor_to_complex_np(kspace), "kspace", output_dir + name)).start()

                            # Save mask
                            output_dir_mask = output + '/masks/'
                            create_dir(output_dir_mask)
                            np.save(output_dir_mask + name + ".npy", mask)

                            # Save csm
                            output_dir_csm = output + '/csms/'
                            create_dir(output_dir_csm)
                            Process(target=save_h5_outputs, args=(
                                complex_tensor_to_complex_np(sensitivity_map), "sensitivity_map",
                                output_dir_csm + name)).start()

                        elif export_type == 'pickle':
                            # Save data
                            output_dir = output + '/test/'
                            create_dir(output_dir)
                            Process(target=save_pickle_outputs, args=(
                                complex_tensor_to_complex_np(torch.stack((imspace, sensitivity_map), 1)),
                                output_dir + subject.split('/')[-2] + '_' + acquisition.split('/')[
                                    -2] + '_' + name)).start()

                            # Save mask
                            acceleration = np.round(mask.size / mask.sum())

                            if acceleration < 4:
                                acceleration = 4.0
                            elif 4 < acceleration < 6:
                                acceleration = 6.0
                            elif 6 < acceleration < 8:
                                acceleration = 8.0
                            elif 8 < acceleration < 10:
                                acceleration = 10.0

                            output_dir_mask = output + '/masks_pickle/' + subject.split('/')[-2] + '/' + \
                                              acquisition.split('/')[
                                                  -2] + '/' + name + '/acc' + str(acceleration)
                            create_dir(output_dir_mask)
                            save_pickle_mask(mask, output_dir_mask + '/mask0')

                        del kspace, mask, imspace, sensitivity_map, name

                    else:
                        del raw_kspace, kspace_path, name
                        torch.cuda.empty_cache()


def main(args):
    start_time = time.perf_counter()
    logger.info("Converting data. This might take some time, please wait...")
    preprocessing(args.root, args.output, args.export_type, args.device)
    time_taken = np.round(time.perf_counter() - start_time)
    logger.info(f"Done! Run Time = {'% d: % 02dmn' % (time_taken / 60, time_taken % 60)} min")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='Root dir containing folders with raw files.')
    parser.add_argument('output', type=str, help='Output dir to save data.')
    parser.add_argument('--export-type', choices=['h5', 'pickle', 'png'], default='h5', help='Choose output format.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Enable GPU.')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
