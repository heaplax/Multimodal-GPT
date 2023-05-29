source /nobackup/users/zitian/code/Heaplax/anaconda3/bin/activate && conda activate && conda activate llava
NODE_RANK=${SLURM_PROCID}
ip2=node${SLURM_NODELIST:5:4}
NODE_LENGTH=${#SLURM_NODELIST}
if [[ ${NODE_LENGTH} == 8  ]]; then
    ip2=node${SLURM_NODELIST:4:4}
else
    ip2=node${SLURM_NODELIST:5:4}
fi
echo $ip2
echo $NODE_RANK
echo $SLURM_JOB_NUM_NODES
torchrun --nproc_per_node=4\
    --master_addr ${ip2} \
    --node_rank ${NODE_RANK} \
    --nnodes $SLURM_JOB_NUM_NODES \
    mmgpt/train/instruction_finetune.py \
    --lm_path checkpoints/llama-7b_hf \
    --tokenizer_path checkpoints/llama-7b_hf \
    --pretrained_path checkpoints/OpenFlamingo-9B/checkpoint.pt \
    --run_name train-my-gpt4 \
    --learning_rate 1e-5 \
    --lr_scheduler cosine \
    --batch_size 1 \
    --tuning_config configs/lora_config.py \
    --dataset_config configs/dataset_config_clevr.py \
    --report_to_wandb