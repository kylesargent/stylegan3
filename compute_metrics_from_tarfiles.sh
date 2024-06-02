#!/bin/bash

python3 compute_metrics_from_tarfiles.py \
    --gt_tarfiles \
        /svl/u/ksarge/cs348k_dit/ffhq/train_images1024x1024/00000.tar \
        /svl/u/ksarge/cs348k_dit/ffhq/train_images1024x1024/01000.tar \
    --pred_tarfiles \
        /svl/u/ksarge/cs348k_dit/ffhq/train_images1024x1024/02000.tar \
        /svl/u/ksarge/cs348k_dit/ffhq/train_images1024x1024/03000.tar \
    --output_json report.json

