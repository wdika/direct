# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import glob
import logging
import os
import sys
import time
from multiprocessing import Process

from tqdm import tqdm

from projects.tecfidera.preprocessing.utils import *

os.environ['TOOLBOX_PATH'] = "/home/dkarkalousos/bart-0.6.00/"
sys.path.append('/home/dkarkalousos/bart-0.6.00/python/')
from bart import bart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_csms(root, output, export_type, device):
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
            sense_ref_scans = glob.glob(acquisition + "*kspace.cfl")

            for sense_ref_scan in sense_ref_scans:
                name = sense_ref_scan.split('.')[0].split('/')[-1].split('_')[0]

                if name == '501':  # estimate csms from the sense ref scan
                    logger.info(
                        f"Processing subject: {subject.split('/')[-2]} | acquisition: {acquisition.split('/')[-2]}"
                        f" | scan: {name}")

                    input_sense_ref_scan = torch.from_numpy(readcfl(sense_ref_scan.split('.')[0])).to(device)
                    input_sense_ref_scan_kspace = T.fftshift(fftn(preprocessing_ifft(input_sense_ref_scan), dim=(1, 2),
                                                       norm="ortho"), dim=(1, 2))
                    input_sense_ref_scan_kspace = input_sense_ref_scan_kspace.permute(1, 2, 0,
                                                                                      3)  # readout dir, phase-encoding dir, slices, coils
                    input_sense_ref_scan_kspace = complex_tensor_to_complex_np(input_sense_ref_scan_kspace)

                    input_csm = bart(1, f"caldir 60", input_sense_ref_scan_kspace)
                    input_csm = np.transpose(input_csm, axes=(2, 0, 1, 3))

                    csm = np.where(input_csm == 0, np.array([0.0], dtype=input_csm.dtype),
                                   (input_csm / np.max(input_csm)))
                    csm = torch.from_numpy(csm)
                    csm = T.ifftshift(csm, dim=(1, 2))

                    # fixed number of slices, selected after checking the pngs
                    AXFLAIR_csm = slice_selection(csm, start=17, end=217)
                    AXT1_MPRAGE_csm = slice_selection(csm, start=22, end=222)

                    if export_type == 'png':
                        output_dir = output + '/png/' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                            -2] + '/SENSEREFSCAN/'
                        create_dir(output_dir)

                        # Save sense coil combined png images
                        Process(target=save_png_outputs, args=(
                            complex_tensor_to_real_np(csm_sense_coil_combination(AXFLAIR_csm, dim=-1)),
                            output_dir + 'AXFLAIR/')).start()

                        Process(target=save_png_outputs, args=(
                            complex_tensor_to_real_np(csm_sense_coil_combination(AXT1_MPRAGE_csm, dim=-1)),
                            output_dir + 'AXT1_MPRAGE/')).start()

                    elif export_type == 'h5':
                        output_dir = output + '/csms/'
                        create_dir(output_dir)

                        # Save csm
                        Process(target=save_h5_outputs,
                                args=(complex_tensor_to_complex_np(AXFLAIR_csm), "sensitivity_map",
                                      output_dir + subject.split('/')[-2] + '_' + acquisition.split('/')[
                                          -2] + '_AXFLAIR')).start()

                        Process(target=save_h5_outputs,
                                args=(complex_tensor_to_complex_np(AXT1_MPRAGE_csm), "sensitivity_map",
                                      subject.split('/')[-2] + '_' + acquisition.split('/')[
                                          -2] + '_AXT1_MPRAGE')).start()


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")
    estimate_csms(args.root, args.output, args.export_type, args.device)
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
