# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import glob
import logging
import sys
import time
from multiprocessing import Process

from torch.fft import fftn
from tqdm import tqdm

from projects.tecfidera.preprocessing.utils import *

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
                kspace = kspace.split('.')[0]
                name = kspace.split('/')[-1].split('_')[0]

                if name != '501':  # exclude the sense ref scan
                    name = 'AXFLAIR' if name == '301' else 'AXT1_MPRAGE'
                    csm = kspace.split('_')[0] + '_csm'

                    logger.info(
                        f"Processing subject: {subject.split('/')[-2]} | acquisition: {acquisition.split('/')[-2]}"
                        f" | scan: {name}")

                    input_imspace = preprocessing_ifft(torch.from_numpy(readcfl(kspace)).to(device))
                    input_kspace = fftn(input_imspace, dim=(1, 2), norm="ortho")
                    input_csm = torch.from_numpy(readcfl(csm)).to(device)

                    # fixed number of slices, selected after checking the pngs
                    kspace = slice_selection(input_kspace, start=17, end=217)
                    imspace = slice_selection(input_imspace, start=17, end=217)
                    csm = slice_selection(input_csm, start=17, end=217)

                    del input_kspace, input_imspace, input_csm

                    mask = complex_tensor_to_real_np(extract_mask(kspace))

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
                        output_dir = output + '/h5/' + subject.split('/')[-2] + '/' + acquisition.split('/')[-2] + '/'
                        create_dir(output_dir)

                        # Save kspace
                        Process(target=save_h5_outputs, args=(
                            complex_tensor_to_complex_np(kspace), "kspace",
                            output_dir + name)).start()
                        # Save csm
                        Process(target=save_h5_outputs, args=(
                            complex_tensor_to_complex_np(csm), "sensitivity_map",
                            output_dir + name + '_csm')).start()
                        # Save mask
                        Process(target=save_h5_outputs, args=(mask, "mask", output_dir + 'mask')).start()


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
