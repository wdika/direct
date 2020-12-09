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

                    # input_kspace = torch.from_numpy(readcfl(filename_kspace.split('.')[0])).to(device)
                    # mask = complex_tensor_to_real_np(extract_mask(input_kspace))
                    # input_imspace = slice_selection(preprocessing_ifft(input_kspace), start=start, end=end)
                    #
                    # # Normalize data
                    # imspace = torch.from_numpy(normalize(complex_tensor_to_complex_np(input_imspace)))
                    # del input_imspace

                    if not skip_csm:
                        input_csm = slice_selection(readcfl(filename_kspace.split('_')[0] + '_csm'), start=start, end=end)

                        # Normalize data
                        # TODO (dk, kp) : remove this normalization when saving to .cfl, then this line should go.
                        # input_csm = input_csm * np.expand_dims(np.sqrt(np.sum(input_csm.conj() * input_csm, -1)), -1)
                        # csm = torch.from_numpy(normalize_csm(input_csm))
                        csm = normalize(input_csm)
                        # del input_csm

                        import matplotlib.pyplot as plt
                        sense = np.sum(normalize_rss(normalize(input_csm)).conj(), -1)[100]
                        sense2 = np.sum(normalize_rss(normalize_csm(input_csm).conj()), -1)[100]
                        sense3 = np.sum(normalize_rss(normalize_rss(input_csm).conj()), -1)[100]

                        print('sense', np.max(np.abs(sense)), np.min(np.abs(sense)))
                        print('sense2', np.max(np.abs(sense2)), np.min(np.abs(sense2)))
                        print('sense3', np.max(np.abs(sense3)), np.min(np.abs(sense3)))

                        plt.subplot(3, 2, 1)
                        plt.imshow(np.abs(sense), cmap='gray')
                        plt.title('sense')
                        plt.colorbar()
                        plt.subplot(3, 2, 2)
                        plt.imshow(np.angle(sense), cmap='gray')
                        plt.title('sense phase')
                        plt.colorbar()
                        plt.subplot(3, 2, 3)
                        plt.imshow(np.abs(sense2), cmap='gray')
                        plt.title('sense2')
                        plt.colorbar()
                        plt.subplot(3, 2, 4)
                        plt.imshow(np.angle(sense2), cmap='gray')
                        plt.title('sense2 phase')
                        plt.colorbar()
                        plt.subplot(3, 2, 5)
                        plt.imshow(np.abs(sense3), cmap='gray')
                        plt.title('sense3')
                        plt.colorbar()
                        plt.subplot(3, 2, 6)
                        plt.imshow(np.angle(sense3), cmap='gray')
                        plt.title('sense3 phase')
                        plt.colorbar()
                        plt.show()

                    if export_type == 'png':
                        output_dir = output + '/png/' + subject.split('/')[-2] + '/' + acquisition.split('/')[
                            -2] + '/' + name
                        create_dir(output_dir)

                        # Save target (SENSE reconstructed) png images
                        Process(target=save_png_outputs, args=(
                            complex_tensor_to_real_np(sense_reconstruction(imspace, csm, dim=-1)),
                            output_dir + '/targets/')).start()

                        if not skip_csm:
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

                        output_dir_mask = output + '/masks/'
                        create_dir(output_dir_mask)

                        name = subject.split('/')[-2] + '_' + acquisition.split('/')[-2] + '_' + name

                        # Save kspace
                        # TODO (dk) : find the correct transformation in pytorch,
                        #  so the norm doesn't change the scale of the data.
                        #  For now I will be using numpy, but that's inefficient.
                        kspace = complex_tensor_to_complex_np(fftn(imspace, dim=(1, 2), norm="ortho"))
                        Process(target=save_h5_outputs, args=(kspace, "kspace", output_dir + name)).start()

                        if not skip_csm:
                            output_dir_csm = output + '/csms/'
                            create_dir(output_dir_csm)

                            # Save csm
                            Process(target=save_h5_outputs, args=(complex_tensor_to_complex_np(csm), "sensitivity_map",
                                                                  output_dir_csm + name)).start()

                        # Save mask
                        np.save(output_dir_mask + name + ".npy", mask)


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
