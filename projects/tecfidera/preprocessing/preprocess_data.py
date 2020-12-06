# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import os
import argparse
import glob
import logging
import sys
import time
from multiprocessing import Process

from torch.fft import fftn
from tqdm import tqdm

from projects.tecfidera.preprocessing.utils import *

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')
from bart import bart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocessing(root, output, export_type, device):
    """
    Parses all subjects, acquisitions, and scans. Performs all the necessary preprocessing steps for the TECFIDERA data.

    Parameters
    ----------
    root : root directory of containing cfl data
    output : output directory to save data
    export_type : h5 or png
    device : cuda or cpu

    """
    subjects = glob.glob(root + "/*/")

    for subject in tqdm(subjects):
        acquisitions = glob.glob(subject + "/*/")

        for acquisition in acquisitions:
            kspaces = glob.glob(acquisition + "*kspace.cfl")

            for kspace in kspaces:
                name = kspace.split('.')[0].split('/')[-1].split('_')[0]

                if name != '501':  # exclude the sense ref scan
                    name = 'AXFLAIR' if name == '301' else 'AXT1_MPRAGE'
                    csm = kspace.split('_')[0] + '_csm'

                    logger.info(
                        f"Processing subject: {subject.split('/')[-2]} | acquisition: {acquisition.split('/')[-2]}"
                        f" | scan: {name}")

                    input_kspace = torch.from_numpy(readcfl(kspace.split('.')[0])).to(device)
                    mask = complex_tensor_to_real_np(extract_mask(input_kspace))
                    input_imspace = preprocessing_ifft(input_kspace)

                    # fixed number of slices, selected after checking the pngs
                    start = 17 if name == 'AXFLAIR' else 22
                    end = 217 if name == 'AXFLAIR' else 222

                    imspace = slice_selection(input_imspace, start=start, end=end)
                    # csm = slice_selection(torch.from_numpy(readcfl(csm)).to(device), start=start, end=end)

                    del input_imspace

                    if export_type == 'png':
                        output_dir = output + '/png/' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                            -2] + '/' + name
                        create_dir(output_dir)

                        # Save target (SENSE reconstructed) png images
                        Process(target=save_png_outputs, args=(
                            complex_tensor_to_real_np(sense_reconstruction(imspace, csm, dim=-1)),
                            output_dir + '/targets/')).start()

                        # Save sense coil combined png images
                        Process(target=save_png_outputs, args=(
                            complex_tensor_to_real_np(csm_sense_coil_combination(csm, dim=-1)),
                            output_dir + '/csms/')).start()

                        # Save mask
                        plt.imshow(mask, cmap='gray')
                        plt.savefig(output_dir + '/mask.png')
                        plt.close()

                    elif export_type == 'h5':
                        output_dir = output + '/kspaces/'
                        create_dir(output_dir)

                        output_dir_csm = output + '/csms/'
                        create_dir(output_dir_csm)

                        output_dir_mask = output + '/masks/'
                        create_dir(output_dir_mask)

                        name = subject.split('/')[-2] + '_' + acquisition.split('/')[-2] + '_' + name

                        # Save kspace
                        k = fftn(imspace, dim=(1, 2), norm="ortho")
                        Process(target=save_h5_outputs, args=(
                            complex_tensor_to_complex_np(k), "kspace",
                            output_dir + name)).start()

                        csm = slice_selection(
                            torch.from_numpy(
                                bart(1, f"caldir 60",
                                     complex_tensor_to_complex_np(
                                         T.fftshift(k, dim=(0, 1, 2)).permute(1, 2, 0, 3)
                                     )
                                     )
                            ).permute(2, 0, 1, 3), start=start, end=end)

                        import matplotlib.pyplot as plt
                        sense = complex_tensor_to_complex_np(torch.sum(torch.conj(csm), -1))
                        sense2 = complex_tensor_to_complex_np(torch.sum(csm, dim=-1))
                        for i in range(sense.shape[0]):
                            plt.subplot(1, 3, 1)
                            plt.imshow(np.abs(complex_tensor_to_complex_np(torch.sum(imspace[i], dim=-1))), cmap='gray')
                            plt.subplot(1, 3, 2)
                            plt.imshow(np.abs(sense[i]), cmap='gray')
                            plt.subplot(1, 3, 3)
                            plt.imshow(np.abs(sense2[i]), cmap='gray')
                            plt.show()

                        # Save csm
                        Process(target=save_h5_outputs, args=(complex_tensor_to_complex_np(csm), "sensitivity_map",
                                                              output_dir_csm + name)).start()

                        # Save mask
                        np.save(output_dir_mask + name + ".npy", mask)


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")
    preprocessing(args.root, args.output, args.export_type, args.device)
    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help='Root dir containing folders with cfl files.')
    parser.add_argument('output', type=str, help='Output dir to save files.')
    parser.add_argument('--export-type', choices=['h5', 'png'], default='png', help='Choose output format.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Enable GPU.')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
