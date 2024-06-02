from metrics import metric_utils
from collections import namedtuple
import torch

import tarfile
import numpy as np
import json
import os
from PIL import Image
from io import BytesIO
import sys
import argparse
from tqdm import tqdm

DumbOptCls = namedtuple("DumbOpt", ["device", "rank", "num_gpus"])
opts = DumbOptCls(device=torch.ones((1,)).cuda().device, rank=0, num_gpus=1)

import importlib


def resize_image(image, target_size):
    return image.resize(target_size, Image.LANCZOS)

def extract_and_resize_images_from_tar(tar_path, target_size=(256, 256)):
    images = []
    with tarfile.open(tar_path, 'r') as tar:
        for member in tqdm(tar.getmembers()):
            if member.isfile() and member.name.endswith('.png'):
                file = tar.extractfile(member)
                image = Image.open(BytesIO(file.read()))
                resized_image = resize_image(image, target_size)
                images.append(np.array(resized_image))
    return np.array(images, dtype=np.uint8)

def load_images_from_tars(tar_paths, target_size=(256, 256)):
    all_images = []
    for tar_path in tar_paths:
        images = extract_and_resize_images_from_tar(tar_path, target_size)
        all_images.append(images)
    return np.concatenate(all_images, axis=0)


def compute_is(gt_ds, pred_ds):
    from metrics import inception_score
    importlib.reload(inception_score)
    from metrics import metric_utils
    importlib.reload(metric_utils)

    inception_score.metric_utils.compute_feature_stats_for_dataset = (
        lambda **kwargs: metric_utils.hijackable_compute_stats(dataset=gt_ds, **kwargs))
    inception_score.metric_utils.compute_feature_stats_for_generator = (
        lambda **kwargs: metric_utils.hijackable_compute_stats(dataset=pred_ds, **kwargs))
    return inception_score.compute_is(
        opts, num_gen=len(pred_ds), num_splits=1)
    
def compute_fid(gt_ds, pred_ds):
    from metrics import frechet_inception_distance
    importlib.reload(frechet_inception_distance)
    from metrics import metric_utils
    importlib.reload(metric_utils)

    frechet_inception_distance.metric_utils.compute_feature_stats_for_dataset = (
        lambda **kwargs: metric_utils.hijackable_compute_stats(dataset=gt_ds, **kwargs))
    frechet_inception_distance.metric_utils.compute_feature_stats_for_generator = (
        lambda **kwargs: metric_utils.hijackable_compute_stats(dataset=pred_ds, **kwargs))
    return frechet_inception_distance.compute_fid(
        opts, max_real=len(gt_ds), num_gen=len(pred_ds))
    
def compute_kid(gt_ds, pred_ds):
    from metrics import kernel_inception_distance
    importlib.reload(kernel_inception_distance)
    from metrics import metric_utils
    importlib.reload(metric_utils)

    kernel_inception_distance.metric_utils.compute_feature_stats_for_dataset = (
        lambda **kwargs: metric_utils.hijackable_compute_stats(dataset=gt_ds, **kwargs))
    kernel_inception_distance.metric_utils.compute_feature_stats_for_generator = (
        lambda **kwargs: metric_utils.hijackable_compute_stats(dataset=pred_ds, **kwargs))
    return kernel_inception_distance.compute_kid(
        opts, max_real=len(gt_ds), num_gen=len(pred_ds), num_subsets=100, max_subset_size=100)


def main():
    parser = argparse.ArgumentParser(description='Compute metrics from tarfiles of images.')
    parser.add_argument('--gt_tarfiles', nargs='+', required=True, help='List of tarfiles for the ground truth dataset')
    parser.add_argument('--pred_tarfiles', nargs='+', required=True, help='List of tarfiles for the predicted dataset')
    parser.add_argument('--output_json', required=True, help='Path to the output JSON file')
    args = parser.parse_args()

    gt_images = load_images_from_tars(args.gt_tarfiles)
    pred_images = load_images_from_tars(args.pred_tarfiles)

    # length adjustment
    assert len(gt_images) >= len(pred_images)
    gt_images = gt_images[:len(pred_images)]

    results = {
        "inception_score": compute_is(gt_images, pred_images),
        "frechet_inception_distance": compute_fid(gt_images, pred_images),
        "kernel_inception_distance": compute_kid(gt_images, pred_images),
    }

    with open(args.output_json, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()