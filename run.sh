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

CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python entrypoint_train.py --name ${name} --version "${version}_step1" \
    --options ${options} --backbone ${backbone}  --num-epochs 5 --depth-loss-weight 1000 --only-depth-training

printf "\n\n\n\nDone training depth only\n\n\n\n"

mkdir -p "checkpoints/${name}/${version}_step2"
cp "checkpoints/${name}/${version}_step1/best_test_loss_depth.pt" "checkpoints/${name}/${version}_step2/"

CUDA_VISIBLE_DEVICES=${cuda_visible_devices}  python entrypoint_train.py --name ${name} --version "${version}_step2" \
    --options ${options} --backbone costvolume  --num-epochs 100  --depth-loss-weight 10 \
    --checkpoint "checkpoints/${name}/${version}_step2/best_test_loss_depth.pt"
