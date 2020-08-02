#!/usr/bin/env bash


# --- CUDA Device ---
#export CUDA_VISIBLE_DEVICES=0
# --- CUDA Device ---

# --- Config ---
export DATA_BASE='data'
export MEAN_BAG_LEN=16
export VAR_BAG_LEN=2
export NUM_BAG_TRAIN=10000
export NUM_BAG_TEST=10000

export EPOCHS=20
export BASE_LR=1e-5
export WEIGHT_DECAY=1e-5

export ALPHA=0.25
export GAMMA=2.0
export LAMBDA1=1.0
export LAMBDA2=3.0
# --- Config ---

# --- Timestamp ---
export TIMESTAMP=$(date '+%Y_%m_%d_%H_%M')
# --- Timestamp ---
# --- Git Commit ---
#export COMMIT=$(git log --pretty=format:'%h' -n 1)
# --- Git Commit ---
# --- Log File Name ---
export LOGFILE="log/${TIMESTAMP}-alpha_${ALPHA}-gamma_${GAMMA}-lambda1_${LAMBDA1}-lambda2_${LAMBDA2}-mean_${MEAN_BAG_LEN}-var_${VAR_BAG_LEN}-num_bag_train_${NUM_BAG_TRAIN}-num_bag_test_${NUM_BAG_TEST}.log"
# --- Log File Name ---
if [ "$1" == "generate" ] || [ "$1" == 'all' ];
then
    if [ -d bags ];
    then
        rm -rf bags
    fi
    echo "-------------- Generate Bags ... --------------"
    sleep 5s
    python create_bags.py --data_base ${DATA_BASE} \
                          --mean_bag_length ${MEAN_BAG_LEN} \
                          --var_bag_length ${VAR_BAG_LEN} \
                          --num_bag_train ${NUM_BAG_TRAIN} \
                          --num_bag_test ${NUM_BAG_TEST}
    python check_bag_balance.py
fi
if [ "$1" == "train" ] || [ "$1" == 'all' ];
then
    sleep 3s
    echo "-------------- Train DDN Model ... --------------" >> ${LOGFILE}
    python main.py --base_lr ${BASE_LR} \
                   --epochs ${EPOCHS} \
                   --weight_decay ${WEIGHT_DECAY} \
                   --alpha ${ALPHA} \
                   --gamma ${GAMMA} \
                   --lambda1 ${LAMBDA1} \
                   --lambda2 ${LAMBDA2} \
                   >> ${LOGFILE}
fi
if [ "$1" == "extract" ] || [ "$1" == 'all' ];
then
    sleep 30s
    echo "-------------- Extract Noise ... --------------" >> ${LOGFILE}
    python noise_extractor.py --data_base ${DATA_BASE} >> ${LOGFILE}
fi
