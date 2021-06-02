# YANMTT

YANMTT is short for Yet Another Neural Machine Translation Toolkit. For a backstory how I ended up creating this toolkit scroll to the bottom of this README. Although the name says that it is yet another toolkit, it was written with the purpose of better understanding of the flow of training, starting from data pre-processing, sharding, batching, distributed training and decoding. There is a significant emphashis on multilingualism and on cross-lingual learning.

**List of features:**
1. **Basic NMT pre-training, fine-tuning, decoding** <br>
    Distributed training (tested on up to 48 GPUs. We dont have that much money.).<br>
    Mixed precision training (optimization issues on multiple GPUs).<br>
    Tempered softmax training, entropy maximization training.<br>
    Joint training using monolingual and parallel corpora.<br>
    MBART pre-training with cross-lingual constraints.<br>
    Sentence representation and attention extraction.<br>
    Scoring translations using trained NMT models. (for reranking, filtering or quality estimation)<br>
2. **Multilingual training**<br>
    Fine-grained control over checkpoint saving for optimising per language pair performance.<br>
3. **Fine-grained parameter transfer** <br>
    Remap embeddings and layers between pre-trained and fine-tuned models. <br>
    Eliminate compoents or layers prior to decoding or fine-tuning. <br>
4. **Model compression** <br>
    Training compact models from scratch via recurrently stacked layers (similar to what is used in ALBERT). <br>
    Distillation of pre-trained and fine-tuned models. Distillation styles supported: label cross-entropy, attention cross-entropy, layer similarity. <br>
5. **Simultaneous NMT** <br>
    Simulated Wait-k NMT where we train and decode wait-K models or decode full-sentence models using wait-k. <br>
6. **Multi-source and Document NMT** <br>
    Vanilla multi-source with two input sentences belonging to different languages. <br>
    Document level NMT where one input is the current sentence and the other one is the context. <br>
    Can be combined with wait-k NMT <br>
    
**Prerequisites (core):** <br>
    Pytorch v1.7.1 <br>
    HuggingFace Transformers v4.3.2 (install the modified copy of the transformers library provided with this toolkit) <br>
    tensorflow-gpu v2.3.0 <br>
    sentencepiece v0.1.95 <br>
    gputil v1.4.0 <br>

**How to install:**
1. Install the required packages via: pip -r requirements.txt
2. Install the modified version of transformers provided along with this repo by: "cd transformers && python setup.py install"

**Scripts and their functionality:**
1. **create_autotokenizer.sh** and **create_autotokenizer.py**: These scripts govern the creation of a unigram SPM or BPE tokenizer. The shell script creates the subword segmenter using sentencepiece which can make both SPM and BPE models. All you need is a monolingual corpus for the languages you are interested in. The python script wraps this around an AlbertTokenizer (for SPM) or MBartTokenizer (for BPE), adds special user defined tokens and saves a configuration file for use in the future via an AutoTokenizer. <br>
**Usage:** see examples/create_tokenizer.sh

2. **pretrain_nmt.py**: This is used to train an MBART model. At the very least you need a monolingual corpus for the languages you are interested in and a tokenizer trained for those languages. This script can also be used to do joint MBART style training jointly with regular NMT training although the NMT training is rather basic because there is no evaluation during training. If you want to do advanced NMT training then you should use the "train_nmt.py" script. Ultimately, you should not use the outcome of this script to perform final translations. Additional advanced usages involve: simulated wait-k simultaneous NMT, knowledge distillation, fine-tuning pre-existing MBART models with fine-grained control over what should be initialized or tuned etc. Read the code and the command line arguments for a better understanding of the advanced features.  <br>
**Usage:** see examples/train_mbart_model.sh

3. **train_nmt.py**: This is used to either train a NMT model from scratch or fine-tune a pre-existing MBART or NMT model. At the very least you need a parallel corpus (preferrably split into train, dev and test sets although we can make do with only a train set) for the language pairs you are interested in. There are several advanced features such as: simulated wait-k simultaneous NMT, knowledge distillation, fine-grained control over what should be initialized or tuned, document NMT, multi-source NMT, multilingual NMT training. <br>
**Usage:** see examples/train_or_fine_tune_model.sh

4. **decode_model.py**: This is used to decode sentences using a trained model. Additionally you can do translation pair scoring, forced decoding, forced alignment (experimental), encoder/decoder representation extraction and alignment visualization. <br>
**Usage:** see examples/decode_or_probe_model.sh

5. **common_utils.py**: This contains all housekeeping functions such as corpora splitting, batch generation, loss computation etc. Do take a look at all the methods since you may need to modify them. <br>

6. **average_checkpoints.py**: You can average the specified checkpoints using either arithmetic or geometric averaging. <br>
**Usage:** see examples/avergage_model_checkpoints.sh

7. **gpu_blocker.py**: This is used to temporarily occupy a gpu in case you use a shared GPU environment. Run this in the background before launching the training processes so that while the training scripts are busy doing preprocessing like sharding or model loading, the GPU you aim for is not occupied by someone else. ****
 

**Backstory: Why I made this toolkit**

Despite the fact that I enjoy coding, I never really pushed myself throughout my Masters and Ph.D. towards writing a self contained toolkit. I had always known that coding is an important part of research and although I had made plenty of meaningful changes to several code bases, I never felt like I owned any of those changes. Fast forward to 2020 where I wanted to play with MBART/BART/MASS. It would have been easy to use fairseq or tensor2tensor but then again the feeling of lack of ownership would remain. Huggingface provides a lot of implementations but (at the time) had no actual script to easily do MBART pre-training. All I had was this single comment "https://github.com/huggingface/transformers/issues/5096#issuecomment-645860271". After a bit of hesitation I decided to get my hands dirty and make a quick notebook for MBART pretraining. That snowballed into me writing my own pipeline for data sharding, preprocessing and training. Since I was at it I wrote a pipeline for tine tuning. Why not go further and write a pipeline for decoding and analysis? Fine-grined control over fine-tuning? Distillation? Multi-source NMT? Document NMT? Simultaneous Wait-K NMT? 3 momnths later I ended up with this toolkit which I wanted to share with everyone. Since I have worked in low-resource MT and efficent MT this toolkit will mostly contain implementations that somehow involve transfer learning, compression/distillation, simultaneous NMT. I am pretty sure its not as fast or perfect like the ones written by the awesome people at GAFA but I will be more than happy if a few people use my script.
