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

# Note 1: The --data_sampling_temperature argument helps balance the size of the corpora. The default value is 1.0 which means smaller corpora are not oversampled. A value of 5 will oversample the smaller corpora a lot more frequently. A very high value of 100 means all corpora will be sampled equally often.
# Note 2: --token_masking_lambda and --token_masking_probs_range arguments will control the lengths of spans to be masked and the percentage of tokens to be masked.
# Note 3: If you wish to use this model with a GUI then look at the --supported_languages argument.
# Note 4: If each line in the corpora are documents then look at the --is_document and --document_level_sentence_delimiter arguments. If you also want to use the future prediction objective then look into the --future_prediction argument.
# Note 5: If you want to use MASS or mT5 style denosing then look at the --span_prediction. For a reversed objective look at --span_to_sentence_prediction.
# Note 5: If you want a contrastive objective  where negative examples are taken from the same batch then look at --contrastive_decoder_training.
# Note 6: Look at the --save_every and --long_save_every arguments to choose how often and which major checkpoints are to be saved, respectively.


## Train a very small MBART model on a single GPU but simulate a 8-gpu setup. The argument --multistep_optimizer_steps is to be used.

# export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

# python pretrain_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files --multistep_optimizer_steps 8

## Train a very small MBART model on a single GPU but initialize it with a previous checkpoint. We assume that the model training had crashed but fortunately we had saved a checkpoint every 1,000 batches.

# export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

# python pretrain_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files --pretrained_model examples/models/mbart_model

## Train a very small MBART model on a single GPU but initialize it with a previous checkpoint. We assume that the model training had not crashed and this is a checkpoint from another pre-training phase. We thus need to ignore the optimizer information if it is present.

# export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

# python pretrain_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files --pretrained_model examples/models/mbart_model --no_reload_optimizer_ctr_and_scheduler

# Note 1: Currently, if you want to use bart-base, bart-large, mBART-50, mBART-25 or IndicBART then you may use facebook/bart-base, facebook/bart-large, facebook/mbart-large-cc25, facebook/mbart-large-50, and ai4bharat/IndicBART or ai4bharat/IndicBARTSS, respectively for --tokenizer_name_or_path and --pretrained_model. Additionally, pass the flag --use_official_pretrained. In this case you will also have to pass the language tokens recognized by the tokenizer. For example, in the above example, langs was hi,en,vi because it was not an official model we are fine-tuning on. However, if you use an official model then you should use tokens like: en for bart, hi_IN for mbart, <2hi> for IndicBART. So look at the tokenizer and identify the language indicator tokens. bart does not support anything other than English. If you have multiple languages for training then use --langs <2hi>,<2mr> in the case of IndicBART and --langs hi_IN,mr_IN for mbart.
# Note 2: If you want to freeze certain components of the model prior to fine-tuning then for embeddings use --freeze_embeddings and for encoder use --freeze_encoder. 
# Note 3: If you want to freeze specific params whose names you know use --freeze_exception_list and pass a comma separated list of names like "encoder_attn,self_attn" for freeze ALL self- and cross-attentions. If you want to want to freeze the self-attention of the encoder only then use "encoder.self_attn". Essentially, be more descriptive with the names.


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


## Train a very small MBART+NLG model on a single GPU. This is essentially joint denoising and NLG model. The most commonly used NLG task is NMT.

# export CUDA_VISIBLE_DEVICES=0 # Change to the GPU ID corresponding to a GPU that is free.

# python pretrain_nmt.py -n 1  -nr 0 -g 1 --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --train_slang hi,vi,en,en --train_tlang en,en,hi,vi --train_src examples/data/train.hi,examples/data/train.vi,examples/data/train.en,examples/data/train.en --train_tgt examples/data/train.en,examples/data/train.en,examples/data/train.hi,examples/data/train.vi --bilingual_train_frequency 0.5 --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files 

# Note 1: The bilingual training frequency is the ratio of the number of training examples that are bilingual to the total number of training examples. This is to balance the NLG and the denoising ovjectives.
# Note 2: If you want to additionally mask the source sentences in the NLG objective then use the flag --source_masking_for_bilingual 
# Note 3: There is no evaluation for the NLG objective so when training reaches a the predefined number of iterations then the training will stop.