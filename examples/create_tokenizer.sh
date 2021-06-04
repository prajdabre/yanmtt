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

# usage: bash examples/create_tokenizer.sh
# Uncomment lines as applicable

# Note: Make sure that your corpora files have the language code as a suffix of the file name else you might have issues.

## An SPM tokenizer without any special user defined tokens.
bash create_autotokenizer.sh examples/data/train.vi,examples/data/train.en,examples/data/train.hi 16000 examples/tokenizers/albert-vienhi16k albert .

## An SPM tokenizer with tokens indicating some domains.
# bash create_autotokenizer.sh examples/data/train.vi,examples/data/train.en,examples/data/train.hi 16000 examples/tokenizers/albert-vienhi16k albert "<2alt>,<2wikinews>"

## A BPE tokenizer without any special user defined tokens.
# bash create_autotokenizer.sh examples/data/train.vi,examples/data/train.en,examples/data/train.hi 16000 examples/tokenizers/mbart-vienhi16k mbart .

## A BPE tokenizer with tokens indicating some domains.
# bash create_autotokenizer.sh examples/data/train.vi,examples/data/train.en,examples/data/train.hi 16000 examples/tokenizers/mbart-vienhi16k mbart "<2alt>,<2wikinews>"