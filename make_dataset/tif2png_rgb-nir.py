import argparse
from pathlib import Path

import numpy as np
import gdal
from PIL import Image
from joblib import Parallel, delayed


def tif2png(filename, input_dir, output_dir):
    gdal.UseExceptions()

    filename = Path(filename)
    name = filename.name.replace(filename.suffix, '')

    path_in = Path(input_dir) / filename

    ds = gdal.Open(str(path_in))

    data_matrix = ds.ReadAsArray()

    rgb = np.transpose(data_matrix[:3, :, :], (1, 2, 0))
    nir = data_matrix[3, :, :]

    path_out_rgb = Path(output_dir) / 'RGB' / (name+'_rgb.png')
    path_out_nir = Path(output_dir) / 'NIR' / (name+'_nir.png')

    if not path_out_rgb.parent.exists():
        path_out_rgb.parent.mkdir(parents=True)
        print('{} was made'.format(path_out_rgb.parent))

    if not path_out_nir.parent.exists():
        path_out_nir.parent.mkdir(parents=True)
        print('{} was made'.format(path_out_nir.parent))

    Image.fromarray(rgb).save(str(path_out_rgb))
    Image.fromarray(nir).save(str(path_out_nir))

    print('done')


def multi_process(input_dir, output_dir):
    tiffiles = Path(input_dir).glob('*.tif')
    Parallel(n_jobs=-1)([delayed(tif2png)(filename, input_dir, output_dir) for filename in tiffiles])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to the directory where the input file is located')
    parser.add_argument('--output', '-o', required=True, help='Path to the directory to output the files')
    parser.add_argument('--filename', default=None, help='filename *.tif')
    args = parser.parse_args()

    if args.filename is not None:
        tif2png(args.filename, args.input, args.output)
    else:
        multi_process(args.input, args.output)
