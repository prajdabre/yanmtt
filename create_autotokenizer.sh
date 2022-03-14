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

src_files=$1 # Has to be a comma separated list
vocab_size=$2
tgt_folder=$3
type=$4
user_tokens=$5
num_train_sentences=$6
character_coverage=$7
mkdir -p $tgt_folder

SPM_TRAIN=../sentencepiece/build/src/spm_train ## Change this to your spm_train path

if [[ $type == "albert" ]]
then
echo "ALBERT tokenizer"
$SPM_TRAIN  --max_sentence_length 20000 --input $src_files   --model_prefix=$tgt_folder/spiece --vocab_size=$vocab_size   --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1   --user_defined_symbols="[CLS],[SEP],[MASK]"   --shuffle_input_sentence=true   --character_coverage=$character_coverage --model_type=unigram --input_sentence_size=$num_train_sentences #Apart from the CLS, SEP and MASK tokens, I was unable to add other user defined tokens here for some reason so I used the create_autoconfig.py to do the same.
elif [[ $type == "mbart" ]]
then
echo "MBART tokenizer."
$SPM_TRAIN  --max_sentence_length 20000 --input $src_files   --model_prefix=$tgt_folder/sentencepiece.bpe --vocab_size=$vocab_size   --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1   --user_defined_symbols="[CLS],[SEP],[MASK]"   --shuffle_input_sentence=true   --character_coverage=$character_coverage --model_type=bpe --input_sentence_size=$num_train_sentences #Apart from the CLS, SEP and MASK tokens, I was unable to add other user defined tokens here for some reason so I used the create_autoconfig.py to do the same.
else
echo "Unknown tokenizer."
exit
fi

## names of the bpe and spm models are set to sentencepiece.bpe and spiece as those are the names that the tokenizer implementations expect.

python create_autoconfig.py $tgt_folder $type $src_files $user_tokens

# Notes: the create_autotokenizer.sh script takes 5 arguments:
# paths to files: this should be a comma separated list
# vocabulary size: an integer
# path to the saved tokenizer: remember to use the prefixes albert or mbart for when you want sentencepiece or bpe subword segmentation, respectively, encapsulated by the AlbertTokenizer or the MBartTokenizer, respectively. Why? Because the AutoTokenizer class likes having the prefix to the tokenizer name so that it can resolve which tokenizer class should be used internally. Note that I personally prefer sentencepiece over BPE have not played with the MbartTokenizer that much.
# tokenizer type: albert or mbart for sentencepiece or BPE, respectively.
# user defined tokens: keep as "." (full stop) if you dont want to specify any special tokens. Specify a comma separated list of tokens such as <2x>,<2y> if you plan to use special tokens. Note that you will have to mess with the data and batch generation method to ensure that these tokens are appropriately used but I have faith in you, King/Queen.
