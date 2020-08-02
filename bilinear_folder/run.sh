#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

python main.py --step 1

sleep 300s

python main.py --step 2