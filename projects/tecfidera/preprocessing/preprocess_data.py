# encoding: utf-8
__author__ = 'Dimitrios Karkalousos'

import argparse
import glob
import logging
import sys
import time
from multiprocessing import Process

from tqdm import tqdm

from projects.tecfidera.preprocessing.utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocessing(root, output, skip_csm, export_type, device):
    """
    Parses all subjects, acquisitions, and scans. Performs all the necessary preprocessing steps for the TECFIDERA data.

    Parameters
    ----------
    root : root directory of containing cfl data
    output : output directory to save data
    skip_csm : toggle this option to skip storing the csms
    export_type : h5 or png
    device : cuda or cpu

    """
    subjects = glob.glob(root + "/*/")

    for subject in tqdm(subjects):
        acquisitions = glob.glob(subject + "/*/")

        for acquisition in acquisitions:
            kspaces = glob.glob(acquisition + "*kspace.cfl")

            for filename_kspace in kspaces:
                name = filename_kspace.split('.')[0].split('/')[-1].split('_')[0]

                if name != '501':  # exclude the sense ref scan
                    name = 'AXFLAIR' if name == '301' else 'AXT1_MPRAGE'

                    # fixed number of slices, selected after checking the pngs
                    start = 17 if name == 'AXFLAIR' else 22
                    end = 217 if name == 'AXFLAIR' else 222

                    logger.info(
                        f"Processing subject: {subject.split('/')[-2]} | time-point: {acquisition.split('/')[-2]}"
                        f" | acquisition: {name}")

                    input_kspace = torch.from_numpy(readcfl(filename_kspace.split('.')[0])).to(device)
                    mask = complex_tensor_to_real_np(extract_mask(input_kspace))

                    input_kspace = slice_selection(input_kspace, start=start, end=end)
                    # imspace = preprocessing_ifft(input_kspace)
                    imspace = ifftn(input_kspace, dim=(1, 2), norm="ortho")
                    del input_kspace

                    imspace = imspace / torch.max(torch.abs(imspace))

                    print(np.min(complex_tensor_to_real_np(imspace)), np.max(complex_tensor_to_real_np(imspace)))

                    # Normalize data
                    # TODO (dk) : change np normalization to pytorch normalization, once complex tensors are supported.
                    #  It is still unclear why normalizing the data here doesn't work with the dataloaders.
                    # imspace = normalize(imspace)

                    if not skip_csm:
                        csm = slice_selection(readcfl(filename_kspace.split('_')[0] + '_csm'), start=start, end=end)

                        # Normalize data
                        # TODO (dk, kp) : make sure about the csm normalization. Here it seems the csm is normalized.

                        print(np.min(np.abs(csm)), np.max(np.abs(csm)))

                    if export_type == 'png':
                        output_dir = output + '/png/' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                            -2] + '/' + name
                        create_dir(output_dir)

                        # Save target (SENSE reconstructed) png images
                        Process(target=save_png_outputs, args=(complex_tensor_to_real_np(
                            sense_reconstruction(imspace, torch.from_numpy(csm).to(device), dim=-1)),
                                                               output_dir + '/targets/')).start()

                        # Save mask
                        plt.imshow(mask, cmap='gray')
                        plt.savefig(output_dir + '/mask.png')
                        plt.close()

                        if not skip_csm:
                            # Save sense coil combined png images
                            Process(target=save_png_outputs, args=(
                                complex_tensor_to_real_np(csm_sense_coil_combination(torch.from_numpy(csm), dim=-1)),
                                output_dir + '/csms/')).start()

                    elif export_type == 'h5':
                        name = subject.split('/')[-2] + '_' + acquisition.split('/')[-2] + '_' + name

                        # Save kspace
                        output_dir = output + '/kspaces/'
                        create_dir(output_dir)
                        Process(target=save_h5_outputs, args=(
                            complex_tensor_to_complex_np(fftn(imspace, dim=(1, 2), norm="ortho")),
                            "kspace", output_dir + name)).start()
                        del imspace

                        # Save mask
                        output_dir_mask = output + '/masks/'
                        create_dir(output_dir_mask)
                        np.save(output_dir_mask + name + ".npy", mask)
                        del mask

                        if not skip_csm:
                            # Save csm
                            output_dir_csm = output + '/csms/'
                            create_dir(output_dir_csm)
                            Process(target=save_h5_outputs,
                                    args=(csm, "sensitivity_map", output_dir_csm + name)).start()
                            del csm


def main(args):
    start_time = time.perf_counter()
    logger.info("Saving data. This might take some time, please wait...")
    preprocessing(args.root, args.output, args.skip_csm, args.export_type, args.device)
    time_taken = time.perf_counter() - start_time
    logger.info(f"Done! Run Time = {time_taken:}s")


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help='Root dir containing folders with cfl files.')
    parser.add_argument('output', type=str, help='Output dir to save files.')
    parser.add_argument('--skip-csm', action="store_true",
                        help='In case you have precomputed the sense maps in another way, '
                             'then toggle this option to skip saving sense maps.')
    parser.add_argument('--export-type', choices=['h5', 'png'], default='png', help='Choose output format.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Enable GPU.')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
