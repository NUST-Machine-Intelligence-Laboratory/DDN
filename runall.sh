#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export DATA_FOR_DDN='CUB200-Google-100-DDN'
export DATA_FOR_BILINEAR='CUB200-Google-100-Bilinear'

mkdir -p log
mkdir -p model

ln -s ${DATA_FOR_DDN} data
echo "Start training DDN Model ... "
bash runDDN.sh all
sleep 60s

rm data
cp -r ${DATA_FOR_BILINEAR} bilinear_folder/${DATA_FOR_BILINEAR}_copy
cp noise-list.txt bilinear_folder/
cd bilinear_folder
ln -s ${DATA_FOR_BILINEAR}_copy data
echo "Removing Noise ... "
python remove_noise.py
echo "Start training Bilinear Model ... "
python main.py --step 1
sleep 300s
python main.py --step 2