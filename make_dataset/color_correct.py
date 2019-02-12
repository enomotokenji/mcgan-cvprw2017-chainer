import os
import argparse
from PIL import Image
from tqdm import tqdm

import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil


def color_correct(args):
    in_dir = args.in_dir
    out_dir = args.out_dir
    if not os.path.exists(in_dir):
        raise Exception('{} does not exists.'.format(in_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    files = os.listdir(in_dir)

    grey_world = cca.grey_world if args.grey_world else lambda x: x
    stretch = cca.stretch if args.stretch else lambda x: x
    max_white = cca.max_white if args.max_white else lambda x: x

    for file in tqdm(files):
        image = Image.open(os.path.join(in_dir, file))
        try:
            image = to_pil(stretch(max_white(grey_world(from_pil(image)))))
        except:
            print(file)
            pass
        image.save(os.path.join(out_dir, file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--grey_world', action='store_true')
    parser.add_argument('--stretch', action='store_true')
    parser.add_argument('--max_white', action='store_true')
    args = parser.parse_args()

    color_correct(args)