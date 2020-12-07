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


def estimate_csms(root, output, calibration_region_size, export_type, device):
    """
    Parses all subjects, acquisitions, and scans. Estimates csms using the sense ref scan [501] for the TECFIDERA data.

    Parameters
    ----------
    root : root directory of containing cfl data
    output : output directory to save data
    calibration_region_size : size of the calibration region
    export_type : h5 or png
    device : cuda or cpu

    """
    subjects = glob.glob(root + "/*/")

    for subject in tqdm(subjects):
        time_points = glob.glob(subject + "/*/")

        for time_point in time_points:
            logger.info(
                f"Processing subject: {subject.split('/')[-2]} | time-point: {time_point.split('/')[-2]}"
                f" | acquisition: Sense Ref Scan")

            sense_ref_scan_kspace = torch.from_numpy(readcfl(time_point + '/501_kspace')).to(device)

            # fixed number of slices, selected after checking the pngs
            AXFLAIR_csm = make_csm_from_sense_ref_scan(readcfl(time_point + '/301_kspace').shape, sense_ref_scan_kspace)
            AXFLAIR_csm = bart(1, f"caldir {calibration_region_size}", AXFLAIR_csm)
            AXFLAIR_csm = T.ifftshift(torch.from_numpy(normalize(AXFLAIR_csm)).permute(2, 0, 1, 3), dim=(1, 2))

            # fixed number of slices, selected after checking the pngs
            AXFLAIR_csm = slice_selection(AXFLAIR_csm, start=17, end=217)

            if export_type == 'png':
                output_dir = output + '/png/' + subject.split('/')[-2] + '/' + time_point.split('/')[
                    -2] + '/SENSEREFSCAN/'
                create_dir(output_dir)

                # Save sense coil combined png images
                Process(target=save_png_outputs, args=(
                    complex_tensor_to_real_np(csm_sense_coil_combination(AXFLAIR_csm, dim=-1)),
                    output_dir + 'AXFLAIR/')).start()

                # Process(target=save_png_outputs, args=(
                #     complex_tensor_to_real_np(csm_sense_coil_combination(AXT1_MPRAGE_csm, dim=-1)),
                #     output_dir + 'AXT1_MPRAGE/')).start()

            elif export_type == 'h5':
                output_dir = output + '/csms/'
                create_dir(output_dir)

                # Save csm
                Process(target=save_h5_outputs,
                        args=(complex_tensor_to_complex_np(AXFLAIR_csm), "sensitivity_map",
                              output_dir + subject.split('/')[-2] + '_' + time_point.split('/')[
                                  -2] + '_AXFLAIR')).start()

                # Process(target=save_h5_outputs,
                #         args=(complex_tensor_to_complex_np(AXT1_MPRAGE_csm), "sensitivity_map",
                #               output_dir + subject.split('/')[-2] + '_' + time_point.split('/')[
                #                   -2] + '_AXT1_MPRAGE')).start()


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")
    estimate_csms(args.root, args.output, args.calibration_region_size, args.export_type, args.device)
    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help='Root dir containing folders with cfl files.')
    parser.add_argument('output', type=str, help='Output dir to save files.')
    parser.add_argument('calibration_region_size', type=int, help='Size of the calibration region.'
                                                                  'For the caldir method.')
    parser.add_argument('--export-type', choices=['h5', 'png'], default='png', help='Choose output format.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Enable GPU.')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
