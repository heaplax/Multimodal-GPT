torchrun --nproc_per_node=4 mmgpt/train/instruction_finetune.py \
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