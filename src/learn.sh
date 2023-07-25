# The path to the tranformers repository
cd ../transformers

# Use a venv
source venv/bin/activate

# Use a local cache
export TRANSFORMERS_CACHE=./cache
export HF_DATASETS_CACHE=.cache/datasets
export HF_METRICS_CACHE=.cache/metrics

# The arguments to provide to the model
train_file=$1
test_file=$2
output_dir=$3

# GPT2 finetuning
python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2-large \
    --train_file $train_file \
    --validation_file $test_file \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 100000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --output_dir $output_dir
