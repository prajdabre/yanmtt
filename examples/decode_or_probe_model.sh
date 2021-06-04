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

# usage: bash examples/decode_or_probe_model.sh
# Uncomment lines as applicable

## Notes:
# General: Look at the arguments in the script "decode_nmt.py" for a better understanding.

## Greedy decode a trained NMT model on a single GPU for a translation direction. Use --beam_size and --length_pentaly for beam search.

dec_mod=examples/models/nmt_model ## Replace this with the path to your NMT model

python decode_nmt.py -n 1  -nr 0 -g 1 --model_path $dec_mod --slang hi --tlang en --test_src examples/data/test.hi --test_tgt examples/translations/translation.en --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --test_ref examples/data/test.en


## Score source and translation using a trained NMT model on a single GPU for a translation direction.

# dec_mod=examples/models/nmt_model ## Replace this with the path to your NMT model

# python decode_nmt.py -n 1  -nr 0 -g 1 --model_path $dec_mod --slang hi --tlang en --test_src examples/data/test.hi --test_tgt examples/translations/translation.en.score --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --test_ref examples/data/test.en --decode_type score

## Perform forced decoding of a source sentence and its translation using a trained NMT model on a single GPU for a translation direction.

# dec_mod=examples/models/nmt_model ## Replace this with the path to your NMT model

# python decode_nmt.py -n 1  -nr 0 -g 1 --model_path $dec_mod --slang hi --tlang en --test_src examples/data/test.hi --test_tgt examples/translations/translation.en.score --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --test_ref examples/data/test.en --decode_type teacher_forced_decoding

## Get encoder and decoder representations from a trained NMT model on a single GPU for a translation direction. The representations are computed by averaging the representations of all non-padding tokens and saved one per translation pair and the values of the hidden representation are tab separated. "layer_id" goes from 0 (embedding outputs) to N (N'th layer)

# dec_mod=examples/models/nmt_model ## Replace this with the path to your NMT model

# python decode_nmt.py -n 1  -nr 0 -g 1 --model_path $dec_mod --slang hi --tlang en --test_src examples/data/test.hi --test_tgt examples/translations/translation.en.encoder_representations --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --test_ref examples/data/test.en --decode_type get_enc_representations --layer_id 1

# python decode_nmt.py -n 1  -nr 0 -g 1 --model_path $dec_mod --slang hi --tlang en --test_src examples/data/test.hi --test_tgt examples/translations/translation.en.encoder_representations --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --test_ref examples/data/test.en --decode_type get_dec_representations --layer_id 1


## Get attention heatmaps from a trained NMT model on a single GPU for a translation direction. "layer_id" goes from 0 (1'st layer) to N-1 (N'th layer). "att_head_id" indicates the H'th attention head and goes from 0 (1'st head) to H-1 (H'th head). 

# dec_mod=examples/models/nmt_model ## Replace this with the path to your NMT model

# python decode_nmt.py -n 1  -nr 0 -g 1 --model_path $dec_mod --slang hi --tlang en --test_src examples/data/test.hi --test_tgt examples/translations/translation.en.encoder_representations --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --test_ref examples/data/test.en --decode_type get_attention --layer_id 0 --att_head_id 0

