# Copyright 2021 National Institute of Information and Communication Technology (Raj Dabre)
# 
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the
# Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall
# be included in all copies or substantial portions of the
# Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/bin/bash
# cd /path/to/this/toolkit
# source <your python virtual environment>/bin/activate
# export PYTHONPATH=$PYTHONPATH:/path/to/this/toolkit/transformers

# usage: bash examples/train_or_fine_tune_model.sh
# Uncomment lines as applicable

## Notes:
# General: Look at the arguments in the script "train_nmt.py" for a better understanding.
# 1. Ensure that the order of the language codes corresponds with that of the language files.
# 2. If you are running this for the first time then the argument "--shard_files" is mandatory.
# 3. When running separate experiments on the same GPU, please specify a unique port via "--port [port]".
# 4. A separate model checkpoint will be saved for a translation direction if the BLEU score for that pair exceeds the previous high. The checkpoint will be saved with an extension indicating the translation direction and the number of batches processed. A checkpoint will also be saved if the global score (average of scores for all translation directions) exceeds the previous high. This checkpoint will be saved with an extension indicating the number of batches processed. Additionally there will be a global checkpoint that will be overwritten every 1,000 batches by default.
# 5. Note the use of the --mnmt flag for multilingual models.

## Train a very small NMT model on a single GPU for a translation direction

export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

python train_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/nmt_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --train_slang hi --train_tlang en --dev_slang hi --dev_tlang en --train_src examples/data/train.hi --train_tgt examples/data/train.en --dev_src examples/data/dev.hi --dev_tgt examples/data/dev.en --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files

## Train a very small NMT model on a single GPU for a translation direction but initialize it with a previous checkpoint. We assume that the model training had crashed but fortunately we had saved a checkpoint every 1,000 batches.

# export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

# python train_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/nmt_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files --pretrained_model examples/models/nmt_model

## Train a very small NMT model on a single GPU for a translation direction but initialize it with a previous checkpoint. We assume that this is not a previous checkpoint but a pretrained model or a checkpoint whose optimizer, scheduler and counter information we want to ignore.

# export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

# python train_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/nmt_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files --pretrained_model examples/models/mbart_model --no_reload_optimizer_ctr_and_scheduler

## Train a very small multilingual NMT model on a single GPU for a multiple translation directions.

# export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

# python train_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/nmt_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --train_slang hi,vi,en,en --train_tlang en,en,hi,vi --dev_slang hi,vi,en,en --dev_tlang en,en,hi,vi --train_src examples/data/train.hi,examples/data/train.vi,examples/data/train.en,examples/data/train.en --train_tgt examples/data/train.en,examples/data/train.en,examples/data/train.hi,examples/data/train.vi --dev_src examples/data/dev.hi,examples/data/dev.vi,examples/data/dev.en,examples/data/dev.en --dev_tgt examples/data/dev.en,examples/data/dev.en,examples/data/dev.hi,examples/data/dev.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files --mnmt


## Train a very small nmt model on multiple GPUs. Use --pretrained_model and --no_reload_optimizer_ctr_and_scheduler (as applicable) if you have a previously trained model.

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Change to the GPU IDs corresponding to GPUs that are free.

# python train_nmt.py -n 1  -nr 0 -g 8 --model_path examples/models/nmt_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --train_slang hi --train_tlang en --dev_slang hi --dev_tlang en --train_src examples/data/train.hi --train_tgt examples/data/train.en --dev_src examples/data/dev.hi --dev_tgt examples/data/dev.en --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files

## Train a very small nmt model on multiple GPUs scattered across multiple machines on a shared network. Use --pretrained_model and --no_reload_optimizer_ctr_and_scheduler (as applicable) if you have a previously trained model.

## Warning: Do not attempt this if your network is not powerful enough.

# Assume 2 machines with 8 GPUs each

# On the first machine aka the head node:

# ipaddr=<get the 1st machine's IP address>
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Change to the GPU IDs corresponding to GPUs that are free.

# python train_nmt.py -n 2  -nr 0 -g 8 -a $ipaddr --model_path examples/models/nmt_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --train_slang hi --train_tlang en --dev_slang hi --dev_tlang en --train_src examples/data/train.hi --train_tgt examples/data/train.en --dev_src examples/data/dev.hi --dev_tgt examples/data/dev.en --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files

# On the second machine aka the follower node:

# python train_nmt.py -n 2  -nr 1 -g 8 -a $ipaddr ---model_path examples/models/nmt_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --train_slang hi --train_tlang en --dev_slang hi --dev_tlang en --train_src examples/data/train.hi --train_tgt examples/data/train.en --dev_src examples/data/dev.hi --dev_tgt examples/data/dev.en --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files