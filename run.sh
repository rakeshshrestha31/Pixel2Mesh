#!/usr/bin/env bash
name="base_multiview_seq_MVS_VGG"
version="4gpu_2batch_0.1depth_1chamfer"
options="experiments/default/shapenet.yml"
backbone="costvolume"
if [ -z "$1" ]
then
    cuda_visible_devices="0"
else
    cuda_visible_devices=$1
fi

# CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python entrypoint_train.py --name ${name} --version "${version}_step1" \
#     --options ${options} --backbone ${backbone}  --num-epochs 5 --depth-loss-weight 1000 --only-depth-training
#
# printf "\n\n\n\nDone training depth only\n\n\n\n"

CUDA_VISIBLE_DEVICES=${cuda_visible_devices}  python entrypoint_train.py --name ${name} --version "${version}" \
    --options ${options} --backbone costvolume  --num-epochs 100  --depth-loss-weight 0.1 --batch-size 2
