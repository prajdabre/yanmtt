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

# usage: bash examples/train_mbart_model.sh
# Uncomment lines as applicable

## Notes:
# General: Look at the arguments in the script "pretrain_nmt.py" for a better understanding. Advanced functionalities can be used by using other arguments.
# 1. Ensure that the order of the language codes corresponds with that of the language files.
# 2. If you are running this for the first time then the argument "--shard_files" is mandatory.
# 3. When running separate experiments on the same GPU, please specify a unique port via "--port [port]".
# 4. A separate model checkpoint will be saved every 10,000 batches by default with an extension indicating the number of batches processed. Additionally there will be a global checkpoint that will be overwritten every 1,000 batches by default.

## Train a very small MBART model on a single GPU

export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

python pretrain_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files

## Train a very small MBART model on a single GPU but initialize it with a previous checkpoint. We assume that the model training had crashed but fortunately we had saved a checkpoint every 1,000 batches.

# export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

# python pretrain_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files --pretrained_model examples/models/mbart_model

## Train a very small MBART model on a single GPU but initialize it with a previous checkpoint. We assume that the model training had not crashed and this is a checkpoint from another pre-training phase. We thus need to ignore the optimizer information if it is present.

# export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

# python pretrain_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files --pretrained_model examples/models/mbart_model --no_reload_optimizer_ctr_and_scheduler


## Train a very small MBART model on multiple GPUs. Use --pretrained_model and --no_reload_optimizer_ctr_and_scheduler (as applicable) if you have a previously trained mbart model.

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Change to the GPU IDs corresponding to GPUs that are free.

# python pretrain_nmt.py -n 1  -nr 0 -g 8 --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files

## Train a very small MBART model on multiple GPUs scattered across multiple machines on a shared network. Use --pretrained_model and --no_reload_optimizer_ctr_and_scheduler (as applicable) if you have a previously trained mbart model.
## Warning: Do not attempt this if your network is not powerful enough.

# Assume 2 machines with 8 GPUs each

# On the first machine aka the head node:

# ipaddr=<get the 1st machine's IP address>
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Change to the GPU IDs corresponding to GPUs that are free.

# python pretrain_nmt.py -n 2  -nr 0 -g 8 -a $ipaddr --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files

# On the second machine aka the follower node:

# python pretrain_nmt.py -n 2  -nr 1 -g 8 -a $ipaddr --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files