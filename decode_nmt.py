# -*- coding: utf-8 -*-
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

## Basic imports
import os
import sys
import argparse
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
##

## Huggingface imports
import transformers
from transformers import AutoTokenizer
from transformers import MBartForConditionalGeneration, MBartConfig, get_linear_schedule_with_warmup
from transformers import AdamW
##

## Pytorch imports
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
##

## Our imports
from common_utils import *
##

## Other imports
import math
import random
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
import gc
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # drawing heat map of attention weights
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Source Han Sans TW',
                                   'sans-serif',
                                   "FreeSerif"  # fc-list :lang=hi family
                                   ]
##

## Unused imports
#from torch.utils.tensorboard import SummaryWriter
##


def model_create_load_decode(gpu, args):
    """The main function which does the overall decoding, visualization etc. Should be split into multiple parts in the future. Currently monolithc intentionally."""
    rank = args.nr * args.gpus + gpu ## The rank of the current process out of the total number of processes indicated by world_size. This need not be done using DDP but I am leaving it as is for consistency with my other code. In the future, I plan to support sharding the decoding data into multiple shards which will then be decoded in a distributed fashion.
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=False, use_fast=False, keep_accents=True)

    print("Tokenizer is:", tok)

    print(f"Running DDP checkpoint example on rank {rank}.")

    config = MBartConfig(vocab_size=len(tok), encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_dim, decoder_ffn_dim=args.decoder_ffn_dim, d_model=args.d_model, add_final_layer_norm=args.add_final_layer_norm, normalize_before=args.normalize_before, normalize_embedding=args.normalize_embedding, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], static_position_embeddings=True, encoder_tying_config=args.encoder_tying_config, decoder_tying_config=args.decoder_tying_config, multilayer_softmaxing=args.multilayer_softmaxing, wait_k=args.wait_k, additional_source_wait_k=args.additional_source_wait_k, unidirectional_encoder=args.unidirectional_encoder, multi_source=args.multi_source, multi_source_method=args.multi_source_method) ## Configuration.
    model = MBartForConditionalGeneration(config)
    model.eval()
    torch.cuda.set_device(gpu)
    
    model.cuda(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu])
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint_dict = torch.load(args.model_path, map_location=map_location)
    
                    
    if type(checkpoint_dict) == dict:
        model.load_state_dict(remap_embeddings_eliminate_components_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict['model'], 4, args), args), strict=True if (args.remap_encoder == "" and args.remap_decoder == "" and not args.eliminate_encoder_before_initialization and not args.eliminate_decoder_before_initialization and not args.eliminate_embeddings_before_initialization) else False) ## Modification needed if we want to load a partial model trained using multilayer softmaxing.
    else:
        model.load_state_dict(remap_embeddings_eliminate_components_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict, 3, args), args), strict=True if (args.remap_encoder == "" and args.remap_decoder == "" and not args.eliminate_encoder_before_initialization and not args.eliminate_decoder_before_initialization and not args.eliminate_embeddings_before_initialization) else False) ## Modification needed if we want to load a partial model trained using multilayer softmaxing.
    model.eval()        
    ctr = 0
    outf = open(args.test_tgt, 'w')
    if args.decode_type == "decode": ## Standard NMT decoding.
        print("Decoding file")
        hyp = []
        if args.test_ref is not None:
            refs = [[refline.strip() for refline in open(args.test_ref)]]
        for input_ids, input_masks in generate_batches_for_decoding(tok, args): #infinite_same_sentence(10000):
            start = time.time()
            print("Processing batch:", ctr)
            if args.multi_source:
                input_ids_parent = input_ids[1]
                input_ids = input_ids[0]
                input_masks_parent = input_masks[1]
                input_masks = input_masks[0]
            with torch.no_grad():
                translations = model.module.generate(input_ids.to(gpu), use_cache=True, num_beams=args.beam_size, max_length=int(len(input_ids[0])*args.max_decode_length_multiplier), min_length=int(len(input_ids[0])*args.min_decode_length_multiplier), early_stopping=True, attention_mask=input_masks.to(gpu), pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], decoder_start_token_id=tok(["<2"+args.tlang+">"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], length_penalty=args.length_penalty, repetition_penalty=args.repetition_penalty, encoder_no_repeat_ngram_size=args.encoder_no_repeat_ngram_size, no_repeat_ngram_size=args.no_repeat_ngram_size, num_return_sequences=args.beam_size if args.return_all_sequences else 1, additional_input_ids=input_ids_parent.to(gpu) if args.multi_source else None, additional_input_ids_mask=input_masks_parent.to(gpu) if args.multi_source else None) ## We translate the batch.
            print(len(input_ids), "in and", len(translations), "out")
            for input_id, translation in zip(input_ids, translations):
                translation  = tok.decode(translation, skip_special_tokens=args.no_skip_special_tokens, clean_up_tokenization_spaces=False) 
                input_id  = tok.decode(input_id, skip_special_tokens=args.no_skip_special_tokens, clean_up_tokenization_spaces=False) ### Get the raw sentences.
                outf.write(translation+"\n")
                outf.flush()
                hyp.append(translation)
            ctr += 1
        if args.test_ref is not None:
            sbleu = get_sacrebleu(refs, hyp)
            print("BLEU score is:", sbleu)
    elif args.decode_type == "score" or args.decode_type == "teacher_forced_decoding": ## Here we will either score a sentence and its translation. The score will be the NLL loss. If not scoring then we will use the softmax to generate translations.
        print("Scoring translations or teacher forced decoding. Will print the log probability or (oracle) translations.")
        hyp = []
        if args.test_ref is not None:
            refs = [[refline.strip() for refline in open(args.test_ref)]]
        for input_ids, input_masks, decoder_input_ids, decoder_masks, labels in generate_batches_pair(tok, args):
            mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu))
            logits = mod_compute.logits
            softmax = torch.nn.functional.log_softmax(logits, dim=-1)
            print(softmax.size())
            if args.decode_type == "teacher_forced_decoding": ## Use the softmax for prediction instead of computing NLL loss.
                translations = torch.argmax(softmax, dim=-1)
                tgt_masks = (labels != tok.pad_token_id).int().to(gpu)
                translations = translations * tgt_masks
                print(translations.size())
                for input_id, translation in zip(input_ids, translations):
                    translation  = tok.decode(translation, skip_special_tokens=args.no_skip_special_tokens, clean_up_tokenization_spaces=False) 
                    input_id  = tok.decode(input_id, skip_special_tokens=args.no_skip_special_tokens, clean_up_tokenization_spaces=False) 
                    outf.write(translation+"\n")
                    outf.flush()
                    hyp.append(translation)
            else: ## Return the label smoothed loss.
                logprobs = label_smoothed_nll_loss(softmax, labels.to(gpu), args.label_smoothing, ignore_index=tok.pad_token_id)
                for logprob in logprobs:
                    print(logprob)
                    outf.write(str(logprob)+"\n")
                    outf.flush()
        
        if args.decode_type == "teacher_forced_decoding" and args.test_ref is not None:
            print(len(refs[0]), len(hyp))
            sbleu = get_sacrebleu(refs, hyp)
            print("BLEU score is:", sbleu)

    elif args.decode_type == "force_align": ## This is experimental. Source is A B C and target is X Y Z. If B and Z are aligned then the highest score should be for the source A MASK C and target X Y MASK. Works sometimes but not always. No detailed documentation yet.
        print("Getting alignments. Will print alignments for each source subword to target subword.")
        final_alignment_pos = ""
        final_alignment_str = ""
        final_src_str = ""
        final_tgt_str = ""
        for input_ids, input_masks, decoder_input_ids, tgt_masks, labels, src_sent_split, tgt_sent_split, enc_pos, dec_pos in generate_batches_pair_masked(tok, args):
            if enc_pos == 0:
                if final_alignment_pos != "":
                    print(final_alignment_pos)
                    print(final_alignment_str)
                    outf.write(final_src_str + "\t" + final_tgt_str + "\t" + final_alignment_pos.strip() + "\t" + final_alignment_str + "\n")
                    outf.flush()
                final_alignment_pos = ""
                final_alignment_str = ""
            mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu))
            logits = mod_compute.logits
            softmax = torch.nn.functional.log_softmax(logits, dim=-1)
            logprobs = nll_loss(softmax, labels.to(gpu), ignore_index=tok.pad_token_id)
            minprob = 1000
            minpos = 0
            for log_prob, dec_p in zip(logprobs, dec_pos):
                if log_prob < minprob:
                    minpos = dec_p
                    minprob = log_prob
            final_alignment_pos += str(enc_pos) + "-" + str(minpos) + " "
            final_alignment_str += src_sent_split[enc_pos] + "-" + tgt_sent_split[minpos] + " "
            final_src_str = " ".join(src_sent_split)
            final_tgt_str = " ".join(tgt_sent_split)
    elif args.decode_type == "get_enc_representations" or args.decode_type == "get_dec_representations": ## We want to extract the encoder or decoder representations for a given layer.
        print("Getting encoder or decoder representations for layer", args.layer_id, ". Will save representations for each input line.")
        for input_ids, input_masks, decoder_input_ids, decoder_masks, labels in generate_batches_pair(tok, args):
            mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu), output_hidden_states=True)
            #print(input_masks)
            if args.decode_type == "get_enc_representations":
                pad_mask = input_ids.to(gpu).eq(tok.pad_token_id).unsqueeze(2)
                hidden_state = mod_compute.encoder_hidden_states[args.layer_id]
            else:
                pad_mask = decoder_input_ids.to(gpu).eq(tok.pad_token_id).unsqueeze(2)
                hidden_state = mod_compute.decoder_hidden_states[args.layer_id]
            hidden_state.masked_fill_(pad_mask, 0.0)
            print(hidden_state.size())
            hidden_state = hidden_state.mean(dim=1)
            for idx, hidden_state_individual in enumerate(hidden_state):
                metadata=tok.decode(input_ids[idx] if args.decode_type == "get_enc_representations" else decoder_input_ids[idx], skip_special_tokens=args.no_skip_special_tokens, clean_up_tokenization_spaces=False)
                outf.write("\t".join([str(elem) for elem in hidden_state_individual.tolist()])+"\n")
                outf.flush()
    elif args.decode_type == "get_attention": ## We want to extract and visualize the self attention and cross attentions for a particular layer and particular head. TODO make this work with all layers and all heads in a single plot. Currently my IQ is low so I am unable to achieve it.
        print("Getting attention for layer ", args.layer_id)  
        sentence_id = 0
        for input_ids, input_masks, decoder_input_ids, decoder_masks, labels in generate_batches_pair(tok, args): 
            mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu), output_attentions=True)
            encoder_attentions = mod_compute.encoder_attentions[args.layer_id]
            decoder_attentions = mod_compute.decoder_attentions[args.layer_id]
            cross_attentions = mod_compute.cross_attentions[args.layer_id]
            for idx, (input_sent, tgt_sent) in enumerate(zip(input_ids, decoder_input_ids)):
                input_sent = tok.convert_ids_to_tokens(input_sent, skip_special_tokens=args.no_skip_special_tokens)
                input_len = len(input_sent)
                tgt_sent = tok.convert_ids_to_tokens(tgt_sent, skip_special_tokens=args.no_skip_special_tokens)
                tgt_len = len(tgt_sent)
                print("Processing for ", input_sent, tgt_sent)
                num_heads = 1
                encoder_sizes = encoder_attentions[idx].size()
                decoder_sizes = decoder_attentions[idx].size()
                cross_sizes = cross_attentions[idx].size()
                print(encoder_sizes, decoder_sizes, cross_sizes)
                encoder_attention = encoder_attentions[args.att_head_id].view(-1, encoder_sizes[-1]).cpu().detach().numpy()
                encoder_attention = encoder_attention[0:input_len*num_heads,0:input_len]
                decoder_attention = decoder_attentions[args.att_head_id].view(-1, decoder_sizes[-1]).cpu().detach().numpy()
                decoder_attention = decoder_attention[0:tgt_len*num_heads,0:tgt_len]
                cross_attention = cross_attentions[args.att_head_id].view(-1, cross_sizes[-1]).cpu().detach().numpy()
                cross_attention = cross_attention[0:tgt_len*num_heads,0:input_len]
                ## Enc Enc plot
                plot_attention(encoder_attention, input_sent, input_sent*num_heads, num_heads, args.test_tgt+".sentence-"+str(sentence_id)+".layer-"+str(args.layer_id)+".head-"+str(args.att_head_id)+".enc_enc.png", "Encoder Encoder Attention")
                ## Dec Dec plot
                plot_attention(decoder_attention, tgt_sent, tgt_sent*num_heads, num_heads, args.test_tgt+".sentence-"+str(sentence_id)+".layer-"+str(args.layer_id)+".head-"+str(args.att_head_id)+".dec_dec.png", "Decoder Decoder Attention")
                ## Enc Dec plot
                plot_attention(cross_attention, input_sent, tgt_sent*num_heads, num_heads, args.test_tgt+".sentence-"+str(sentence_id)+".layer-"+str(args.layer_id)+".head-"+str(args.att_head_id)+".enc_dec.png", "Encoder Decoder Attention")
                sentence_id += 1
                
    outf.close()
    
    
    dist.destroy_process_group()

def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-a', '--ipaddr', default='localhost', type=str, 
                        help='IP address of the main node')
    parser.add_argument('-p', '--port', default='26023', type=str, 
                        help='Port main node')
    parser.add_argument('-m', '--model_path', default='pytorch.bin', type=str, 
                        help='Path to save the fine tuned model')
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size in terms of number of sentences')
    parser.add_argument('--beam_size', default=4, type=int, 
                        help='Size of beam search')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, 
                        help='To prevent repetition during decoding. 1.0 means no repetition. 1.2 was supposed to be a good value for some settings according to some researchers.')
    parser.add_argument('--no_repeat_ngram_size', default=0, type=int, 
                        help='N-grams of this size will never be repeated in the decoder. Lets play with 2-grams as default.')
    parser.add_argument('--length_penalty', default=1.0, type=float, 
                        help='Set to more than 1.0 for longer sentences.')
    parser.add_argument('--encoder_no_repeat_ngram_size', default=0, type=int, 
                        help='N-gram sizes to be prevented from being copied over from encoder. Lets play with 2-grams as default.')
    parser.add_argument('--encoder_layers', default=6, type=int, help="The value for number of encoder layers")
    parser.add_argument('--decoder_layers', default=6, type=int, help="The value for number of decoder layers")
    parser.add_argument('--label_smoothing', default=0.1, type=float, help="The value for label smoothing")
    parser.add_argument('--dropout', default=0.1, type=float, help="The value for embedding dropout")
    parser.add_argument('--layer_id', default=6, type=int, help="The id of the layer from 0 to num_layers. Note that the implementation returns the embedding layer output at index 0 so the output of layer 1 is actually at index 1.")
    parser.add_argument('--att_head_id', default=0, type=int, help="The id of the attention head from 0 to encoder_attention_heads-1 or decoder_attention_heads-1")
    parser.add_argument('--attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--encoder_attention_heads', default=8, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--decoder_attention_heads', default=8, type=int, help="The value for number of decoder attention heads")
    parser.add_argument('--wait_k', default=-1, type=int, help="The value for k in wait-k snmt. Keep as -1 for non-snmt aka vanilla NMT.")
    parser.add_argument('--additional_source_wait_k', default=-1, type=int, help="The value for k in wait-k snmt. Keep as -1 for non-snmt aka vanilla NMT. This is the wait-k for the additional source language. Can be used for simultaneous mutlisource NMT.")
    parser.add_argument('--future_prediction', action='store_true', 
                        help='This assumes that we dont mask token sequences randomly but only after the latter half of the sentence. We do this to make the model more robust towards missing future information. Granted we can achieve this using wait-k but methinks this may be a better way of training.')
    parser.add_argument('--unidirectional_encoder', action='store_true', 
                        help='This assumes that we use a unidirectional encoder. This is simulated via a lower-triangular matrix mask in the encoder. Easy peasy lemon squeazy.')
    parser.add_argument('--decoder_ffn_dim', default=2048, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--encoder_ffn_dim', default=2048, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--d_model', default=512, type=int, help="The value for model hidden size")
    parser.add_argument('--max_decode_length_multiplier', default=2.0, type=float, 
                        help='This multiplied by the source sentence length will be the maximum decoding length.')
    parser.add_argument('--min_decode_length_multiplier', default=0.1, type=float, 
                        help='This multiplied by the source sentence length will be the minimum decoding length.')
    parser.add_argument('--hard_truncate_length', default=512, type=int, 
                        help='Should we perform a hard truncation of the batch? This will be needed to eliminate cuda caching errors for when sequence lengths exceed a particular limit. This means self attention matrices will be massive and I used to get errors. Choose this value empirically.')
    parser.add_argument('--token_masking_lambda', default=3.5, type=float, help="The value for the poisson sampling lambda value")
    parser.add_argument('--token_masking_probs_range', nargs='+', type=float, default=[0.3], help="The range of probabilities with which the token will be masked. If you want a fixed probability then specify one argument else specify ONLY 2.")
    parser.add_argument('--add_final_layer_norm', action='store_true', 
                        help='Should we add a final layer norm?')
    parser.add_argument('--normalize_before', action='store_true', 
                        help='Should we normalize before doing attention?')
    parser.add_argument('--normalize_embedding', action='store_true', 
                        help='Should we normalize embeddings?')
    parser.add_argument('--scale_embedding', action='store_true', 
                        help='Should we scale embeddings?')
    parser.add_argument('--max_src_length', default=256, type=int, 
                        help='Maximum token length for source language')
    parser.add_argument('--max_tgt_length', default=256, type=int, 
                        help='Maximum token length for target language')
    parser.add_argument('--encoder_tying_config', default=None, type=str,
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--decoder_tying_config', default=None, type=str,
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--slang', default='en', type=str, 
                        help='Source language')
    parser.add_argument('--decode_type', default='decode', type=str, 
                        help='One of decode, score, force_align, get_enc_representation, get_dec_representation, teacher_forced_decoding or get_attention. When getting representations or attentions you must specify the index of the layer which you are interested in. By default the last layer is considered.')
    parser.add_argument('--tokenizer_name_or_path', default='ai4bharat/indic-bert', type=str, 
                        help='Name of or path to the tokenizer')
    parser.add_argument('--pretrained_tokenizer_name_or_path', default=None, type=str, 
                        help='Name of or path to the tokenizer of the pretrained model if its different from the current model. This tokenizer will be used for remapping embeddings so as to reuse as many pretrained embeddings as possible.')
    parser.add_argument('--tlang', default='hi', type=str, 
                        help='Target language')
    parser.add_argument('--test_src', default='', type=str, 
                        help='Source language test sentences')
    parser.add_argument('--test_tgt', default='', type=str, 
                        help='Target language translated sentences')
    parser.add_argument('--test_ref', default=None, type=str, 
                        help='Target language reference sentences')
    parser.add_argument('--multi_source', action='store_true', 
                        help='Are we doing multisource NMT? In that case you should specify the train_src as a hyphen separated pair indicating the parent language and the child language. You should also ensure that the source file is a tab separated file where each line contains "the parent pair source sentence[tab]child pair source sentence".')
    parser.add_argument('--multi_source_method', default=None, type=str, 
                        help='How to merge representations from multiple sources? Should be one of self_relevance_and_merge_after_attention, self_relevance_and_merge_before_attention, merge_after_attention, merge_before_attention. We also need to implement averaging methods such as early averaging (average encoder representations) and late averaging (average softmaxes). Relevance mechanisms should have a separate flag in the future.')
    parser.add_argument('--mask_input', action='store_true', 
                        help='Should we mask words in the input sentence? We should use this for hallucinating variations of the input sentences.')
    parser.add_argument('--return_all_sequences', action='store_true', 
                        help='Should we return all beam sequences?')
    parser.add_argument('--no_skip_special_tokens', action='store_false', 
                        help='Should we return outputs without special tokens? We may need this to deal with situations where the user specified control tokens must be in the output.')
    parser.add_argument('--multilayer_softmaxing', action='store_true', 
                        help='Should we apply a softmax for each decoder layer? Unsupported for distillation. Only for vanilla training.')
    parser.add_argument('--remap_encoder', default='', type=str, 
                        help='This indicates the remappings for the layer. Example: 1-2,2-4,3-6. The plan is to use these remappings to cut down the model prior to decoding or training. Suppose we have a 6 layer model but we only want to utilize the 2nd, 4th and 6th layer then we will copy the content of the 2nd, 4th and 6th layers to the 1st, 2nd and 3rd layer and delete the former layers from the parameter dictionary. This counts as layer pruning. IMPORTANT NOTE: Ensure that you specify ALL child layer indices you wish mapped. For example if you want 1-2,2-1,3-3 you MUST NOT skip the 3-3 part else it will be deleted from the model dictionary and will be randomly initialized. The loading mechanism is not strict so it will ignore missing or non matching keys. ADDITIONAL NOTE: Load a checkpoint with only the model and not the optimizer to prevent failure as we are not sure if remapping optimizers and learning rate schedulers make sense or not.')
    parser.add_argument('--remap_decoder', default='', type=str, 
                        help='This indicates the remappings for the layer. Example: 1-2,2-4,3-6. The plan is to use these remappings to cut down the model prior to decoding or training. Suppose we have a 6 layer model but we only want to utilize the 2nd, 4th and 6th layer then we will copy the content of the 2nd, 4th and 6th layers to the 1st, 2nd and 3rd layer and delete the former layers from the parameter dictionary. This counts as layer pruning. IMPORTANT NOTE: Ensure that you specify ALL child layer indices you wish mapped. For example if you want 1-2,2-1,3-3 you MUST NOT skip the 3-3 part else it will be deleted from the model dictionary and will be randomly initialized. The loading mechanism is not strict so it will ignore missing or non matching keys. ADDITIONAL NOTE: Load a checkpoint with only the model and not the optimizer to prevent failure as we are not sure if remapping optimizers and learning rate schedulers make sense or not.')
    parser.add_argument('--eliminate_encoder_before_initialization', action='store_true', 
                        help='Lets wipe out the encoder params from the pretrained model before we use it to initialize the current model. This means we have random encoder initialization.')
    parser.add_argument('--eliminate_decoder_before_initialization', action='store_true', 
                        help='Lets wipe out the decoder params from the pretrained model before we use it to initialize the current model. This means we have random decoder initialization.')
    parser.add_argument('--eliminate_embeddings_before_initialization', action='store_true', 
                        help='Lets wipe out the embedding params from the pretrained model before we use it to initialize the current model. This means we have random embedding initialization.')
    
    args = parser.parse_args()
    assert len(args.token_masking_probs_range) <= 2
    print("IP address is", args.ipaddr)
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = args.ipaddr              #
    os.environ['MASTER_PORT'] = args.port                      #
    mp.spawn(model_create_load_decode, nprocs=args.gpus, args=(args,))         #
    #########################################################
    
if __name__ == "__main__":
    run_demo()