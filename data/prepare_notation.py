"""This script is for preparing notation for specific train/test dir
with following options:
- dataset_dir: contains images with format label_id.{jpg,png}
- out: notation file for images in dataset_dir

Example:
Create 'train.txt' for 'name/train' directory:
> python prepare_notation.py --dataset_dir name/train --out train.txt
"""
import argparse
import os
import unicodedata


def main(args):
    with open(args.out, 'w') as f:
        for img_name in os.listdir(args.dataset_dir):
            old_img_path = os.path.join(args.dataset_dir, img_name)

            # Normalize data due to Unicode problem
            img_name = unicodedata.normalize('NFC', img_name)
            label = img_name.split('_')[0]
            img_path = os.path.join(args.dataset_dir, img_name)

            os.rename(old_img_path, img_path)
            f.write('{}\t{}\n'.format(img_path, label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    main(args)
