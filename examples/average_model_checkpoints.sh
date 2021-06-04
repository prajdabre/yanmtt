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

# usage: bash examples/average_model_checkpoints.sh
# Uncomment lines as applicable

## Notes:
# General: Look at the arguments in the script "average_checkpoints.py" for a better understanding.
# 1. Use --geometric_mean flag for multilingual models.

## Arithmetic average two checkpoints (actually the same checkpoint twice as this is just an example)

python average_checkpoints.py --inputs examples/models/nmt_model examples/models/nmt_model --output examples/models/averaged.nmt_model

## Geometric average two checkpoints (actually the same checkpoint twice as this is just an example)

# python average_checkpoints.py --inputs examples/models/nmt_model examples/models/nmt_model --output examples/models/averaged.nmt_model --geometric_mean