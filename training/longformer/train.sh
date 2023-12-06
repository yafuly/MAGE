# MODEL=bert-base-cased
plm_dir="allenai/longformer-base-4096"
seed=42629309
data_path="./data/cross_domains_cross_models"
train_file="$data_path/train.csv"
valid_file="$data_path/valid.csv"
out_dir="./output_samples_${seed}_lfbase"
time=$(date +'%m:%d:%H:%M')
mkdir -p $out_dir

CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --do_train \
  --model_name_or_path $plm_dir \
  --do_eval \
  --train_file $train_file \
  --validation_file $valid_file \
  --max_seq_length 2048 \
  --per_device_train_batch_size  2 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 8 \
  --fp16 \
  --output_dir $out_dir 2>&1 | tee $out_dir/log.train.$time
