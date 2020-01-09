#!/usr/bin/env bash
name="singleview_with_cv_only"
version="1depth_2allloss"
options="experiments/default/tensorflow.yml"
backbone="costvolume"
if [ -z "$1" ]
then
    cuda_visible_devices="0"
else
    cuda_visible_devices=$1
fi

CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python entrypoint_train.py --name ${name} --version ${version} \
    --options ${options} --backbone ${backbone}  --num-epochs 50 --only-depth-training

printf "Done training depth only\n\n\n\n"
CUDA_VISIBLE_DEVICES=${cuda_visible_devices}  python entrypoint_train.py --name ${name} --version ${version} \
    --options ${options} --backbone costvolume  --num-epochs 50 \
    --checkpoint "checkpoints/${name}/${version}/best_test_loss_depth.pt"
