import os
import argparse


def main(args):
    dirs = os.listdir(args.dir)
    dirs = sorted(dirs)

    if os.path.isfile(os.path.join(args.dir, dirs[0])):
        with open(args.filename, 'w') as f:
            f.write('\n'.join(dirs))
        return None

    txt = []
    for dir in dirs:
        files = os.listdir(os.path.join(args.dir, dir))
        files = sorted(files)

        paths = [os.path.join(dir, file) for file in files]
        txt.append('\n'.join(paths))

    with open(args.filename, 'w') as f:
        f.write('\n'.join(txt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    args = parser.parse_args()

    main(args)
