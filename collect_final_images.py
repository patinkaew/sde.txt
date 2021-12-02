import os
import argparse
import util
from shutil import copy


def get_subdirectories(base_dir):
    return [dI for dI in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,dI))]


def collect_final(args):
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    util.mkdir_if_not_exists(dst_dir)
    for sub_dir in get_subdirectories(src_dir):
        img_dir = src_dir + '/' + sub_dir
        copy(img_dir + '/final.png', dst_dir + '/final_{:02d}.png'.format(int(sub_dir)))


def main():
    parser = argparse.ArgumentParser(description='Copy final images')
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--dst_dir', type=str)
    args = parser.parse_args()
    collect_final(args)


if __name__ == '__main__':
    main()
