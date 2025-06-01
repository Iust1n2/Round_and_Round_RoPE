#!/bin/bash

run_training() {
    config=$1
    wandb_flag=$2
    run_name=$(basename "$config" .py)

    echo "[`date`] START: $run_name"
    if [[ "$wandb_flag" == "true" ]]; then
        python train_model.py -c "$config" -w
    else
        python train_model.py -c "$config"
    fi
    echo "[`date`] DONE: $run_name"
}

if [[ "$1" == "--use_wandb" ]]; then
    run_training ./config/attn_only_rope.py true &
    run_training ./config/full_model_rope.py true &
    run_training ./config/full_model_alibi.py true &
else
    run_training ./config/attn_only_rope.py false &
    run_training ./config/full_model_rope.py false &
    run_training ./config/full_model_alibi.py false &
fi

wait
echo "[`date`] All training jobs finished."
