# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
import time

from transformers import MBartForConditionalGeneration, MBartConfig, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AdamW
# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import os

from torch.utils.tensorboard import SummaryWriter

import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import sys
import torch.distributed as dist
import numpy as np


import matplotlib
import numpy as np


matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # drawing heat map of attention weights

#plt.rcParams['font.sans-serif'] = ['SimSun']  # set font family
# matplotlib.rcParams['font.sans-serif'] = ['Source Han Sans TW',
#                                    'sans-serif',
#                                    "Lohit Devanagari"  # fc-list :lang=hi family
#                                    ]

from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Tahoma']

rcParams['font.sans-serif'] = ['Source Han Sans TW',
                                   'sans-serif',
                                   "FreeSerif"  # fc-list :lang=hi family
                                   ]

# from matplotlib.font_manager import FontProperties


# font_prop = FontProperties(fname='Mangal.ttf', size=18)



import random

import sacrebleu

def get_sacrebleu(refs, hyp):
    """Returns the sacrebleu score."""
    bleu = sacrebleu.corpus_bleu(hyp, refs)
    return bleu.score

def generate_batches_pair(tok, args):
    """Generates the source, target and source attention masks for the training set."""
    src_file = open(args.test_src)
    tgt_file = open(args.test_ref)
    corpus = [(src_line, tgt_line) for src_line, tgt_line in zip(src_file, tgt_file)]
    epoch_counter = 0
    curr_batch_count = 0
    encoder_input_batch = []
    decoder_input_batch = []
    decoder_label_batch = []
    max_src_sent_len = 0
    max_tgt_sent_len = 0
    for src_sent, tgt_sent in corpus:
        src_sent = src_sent.strip()
        tgt_sent = tgt_sent.strip()
        start = time.time()
        slang = "<2"+args.slang+">"
        tlang = "<2"+args.tlang+">"
        src_sent_split = src_sent.split(" ")
        tgt_sent_split = tgt_sent.split(" ")
        tgt_sent_len = len(tgt_sent_split)
        src_sent_len = len(src_sent_split)
        if src_sent_len <=1 or src_sent_len >= 256 or tgt_sent_len <=1 or tgt_sent_len >= 256:
            continue
        iids = tok(src_sent + " </s> " + slang, add_special_tokens=False, return_tensors="pt").input_ids
        curr_src_sent_len = len(iids[0])

        iids = tok(tlang + " " + tgt_sent, add_special_tokens=False, return_tensors="pt").input_ids
        curr_tgt_sent_len = len(iids[0])
        if curr_src_sent_len <= 1 or curr_src_sent_len >= 256 or curr_tgt_sent_len <= 1 or curr_tgt_sent_len >= 256:
            continue
        if curr_src_sent_len > max_src_sent_len:
            max_src_sent_len = curr_src_sent_len

        if curr_tgt_sent_len > max_tgt_sent_len:
            max_tgt_sent_len = curr_tgt_sent_len

        encoder_input_batch.append(src_sent + " </s> " + slang)
        decoder_input_batch.append(tlang + " " + tgt_sent)
        decoder_label_batch.append(tgt_sent + " </s>")
        curr_batch_count += 1
        if curr_batch_count == args.batch_size:
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
            input_masks = (input_ids != tok.pad_token_id).int()
            decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
            decoder_masks = (decoder_input_ids != tok.pad_token_id).int()
            labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
            end = time.time()
            yield input_ids, input_masks, decoder_input_ids, decoder_masks, labels
            curr_batch_count = 0
            encoder_input_batch = []
            decoder_input_batch = []
            decoder_label_batch = []
            max_src_sent_len = 0
            max_tgt_sent_len = 0

    if len(encoder_input_batch) != 0:
        input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
        input_masks = (input_ids != tok.pad_token_id).int()
        decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
        decoder_masks = (decoder_input_ids != tok.pad_token_id).int()
        labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
        yield input_ids, input_masks, decoder_input_ids, decoder_masks, labels

def generate_batches_pair_masked(tok, args):
    """Generates the source, target and source attention masks for the training set."""
    src_file = open(args.test_src)
    tgt_file = open(args.test_ref)
    corpus = [(src_line, tgt_line) for src_line, tgt_line in zip(src_file, tgt_file)]
    epoch_counter = 0
    curr_batch_count = 0
    for src_sent, tgt_sent in corpus:
        src_sent = src_sent.strip()
        tgt_sent = tgt_sent.strip()
        start = time.time()
        slang = "<2"+args.slang+">"
        tlang = "<2"+args.tlang+">"
        src_sent_split = src_sent.split(" ")
        tgt_sent_split = tgt_sent.split(" ")
        tgt_sent_len = len(tgt_sent_split)
        src_sent_len = len(src_sent_split)
        if src_sent_len <=1 or src_sent_len >= 100 or tgt_sent_len <=1 or tgt_sent_len >= 100:
            continue
        
        for pos_src in range(src_sent_len):
            encoder_input_batch = []
            decoder_input_batch = []
            decoder_label_batch = []
            dec_pos = []
            enc_pos = pos_src
            max_src_sent_len = 0
            max_tgt_sent_len = 0
            new_src_sent_split = list(src_sent_split)
            new_src_sent_split[pos_src] = "[MASK]"  #[] #"[MASK]" #"<pad>"
            new_src_sent = " ".join(new_src_sent_split)
            iids = tok(new_src_sent + " </s> " + slang, add_special_tokens=False, return_tensors="pt").input_ids
            curr_src_sent_len = len(iids[0])
            if curr_src_sent_len > max_src_sent_len:
                max_src_sent_len = curr_src_sent_len
            for pos_tgt in range(tgt_sent_len):
                dec_pos.append(pos_tgt)
                new_tgt_sent_split = list(tgt_sent_split)
                new_tgt_sent_split[pos_tgt] = "[MASK]" #[] #"[MASK]" #"<pad>"
                new_tgt_sent = " ".join(new_tgt_sent_split)
                iids = tok(tlang + " " + new_tgt_sent, add_special_tokens=False, return_tensors="pt").input_ids
                curr_tgt_sent_len = len(iids[0])
                if curr_tgt_sent_len > max_tgt_sent_len:
                    max_tgt_sent_len = curr_tgt_sent_len
                encoder_input_batch.append(new_src_sent + " </s> " + slang)
                decoder_input_batch.append(tlang + " " + new_tgt_sent)
                decoder_label_batch.append(new_tgt_sent + " </s>")
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
            input_masks = (input_ids != tok.pad_token_id).int()
            decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
            tgt_masks = (decoder_input_ids != tok.pad_token_id).int()
            labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
            end = time.time()

            yield input_ids, input_masks, decoder_input_ids, tgt_masks, labels, src_sent_split, tgt_sent_split, enc_pos, dec_pos
        
def generate_batches(tok, args):
    """Generates the source sentences for the test set."""
    src_file = open(args.test_src)
    curr_batch_count = 0
    encoder_input_batch = []
    max_src_sent_len = 0

    for src_line in src_file:
        start = time.time()
        src_sent = src_line.strip()
        lang = "<2"+args.slang+">"
        src_sent_split = src_sent.split(" ")
        sent_len = len(src_sent_split)
        if sent_len <1 or sent_len > 256:
            src_sent = " ".join(src_sent_split[:256])
        iids = tok(src_sent + " </s> " + lang, add_special_tokens=False, return_tensors="pt").input_ids
        curr_src_sent_len = len(iids[0])

        if curr_src_sent_len > max_src_sent_len:
            max_src_sent_len = curr_src_sent_len

        encoder_input_batch.append(src_sent + " </s> " + lang)
        curr_batch_count += 1
        if curr_batch_count == args.batch_size:
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
            input_masks = input_ids != tok.pad_token_id
            end = time.time()
            yield input_ids, input_masks
            curr_batch_count = 0
            encoder_input_batch = []
            max_src_sent_len = 0

    if len(encoder_input_batch) != 0:
        input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
        input_masks = input_ids != tok.pad_token_id
        yield input_ids, input_masks

def nll_loss(lprobs, target, ignore_index=0):
    """From fairseq. This returns the label smoothed loss."""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)

    nll_loss = nll_loss.sum(-1).mean(-1)
    return nll_loss

def plot_attention(data, X_label=None, Y_label=None, num_heads=None, file_name=None, plot_title=None):
    '''
      Plot the attention model heatmap
      Args:
        data: attn_matrix with shape [ty, tx], cut before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    print(len(X_label))
    print(len(Y_label))
    print(data.shape)
    fig, ax = plt.subplots(figsize=(20, 20*num_heads))  # set figure size
    im = ax.imshow(data)


    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(X_label)))
    ax.set_yticks(np.arange(len(Y_label)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(X_label)
    ax.set_yticklabels(Y_label)
    ax.xaxis.tick_top()


    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")
    
    for i in range(len(Y_label)):
        for j in range(len(X_label)):
            text = ax.text(j, i, "%.1f" % data[i, j],
                           ha="center", va="center", color="b",size=10.0)
    # Save Figure
    ax.set_title(plot_title)
    fig.tight_layout()

    print("Saving figures %s" % file_name)
    fig.savefig(file_name)  # save the figure to file
    plt.close(fig)  # close the figure

def model_create_load_run_save(gpu, args):
    """The main function which does the magic. Should be split into multiple parts in the future."""
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=False, use_fast=False, keep_accents=True)

#     files = {"as": "data/as/as.txt", "bn": "data/bn/bn.txt", "en": "data/en/en.txt", "gu": "data/gu/gu.txt", "hi": "data/hi/hi.txt", "kn": "data/kn/kn.txt", "ml": "data/ml/ml.txt", "mr": "data/mr/mr.txt", "or": "data/or/or.txt", "pa": "data/pa/pa.txt", "ta": "data/ta/ta.txt", "te": "data/te/te.txt"}  ## Get this from command line
    
#     special_tokens_dict = {'additional_special_tokens': ["<s>", "</s>"] + ["<2"+lang+">" for lang in files.keys()] + ["<2"+args.slang+">", "<2"+args.tlang+">"]}
#     num_added_toks = tok.add_special_tokens(special_tokens_dict)

    print("Tokenizer is:", tok)

    print(f"Running DDP checkpoint example on rank {rank}.")

    config = MBartConfig(vocab_size=len(tok), encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_dim, decoder_ffn_dim=args.decoder_ffn_dim, d_model=args.d_model, add_final_layer_norm=args.add_final_layer_norm, normalize_before=args.normalize_before, normalize_embedding=args.normalize_embedding, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], static_position_embeddings=True, encoder_tying_config=args.encoder_tying_config, decoder_tying_config=args.decoder_tying_config)
    model = MBartForConditionalGeneration(config)
    model.eval()
    torch.cuda.set_device(gpu)
    
    model.cuda(gpu)
    model = DistributedDataParallel(model, device_ids=[gpu])
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint_dict = torch.load(args.model_to_decode, map_location=map_location)
    if type(checkpoint_dict) == dict:
        model.load_state_dict(checkpoint_dict['model'])
    else:
        model.load_state_dict(checkpoint_dict)
            
    ctr = 0
    outf = open(args.test_tgt, 'w')
    if args.decode_type == "decode":
        print("Decoding file")
        hyp = []
        refs = [[refline.strip() for refline in open(args.test_ref)]]
        for input_ids, input_masks in generate_batches(tok, args): #infinite_same_sentence(10000):
            start = time.time()
            print("Processing batch:", ctr)
            with torch.no_grad():
                translations = model.module.generate(input_ids.to(gpu), use_cache=True, num_beams=args.beam_size, max_length=int(len(input_ids[0])*args.max_decode_length_multiplier), early_stopping=True, attention_mask=input_masks.to(gpu), pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], decoder_start_token_id=tok(["<2"+args.tlang+">"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], length_penalty=args.length_penalty, repetition_penalty=args.repetition_penalty, encoder_no_repeat_ngram_size=args.encoder_no_repeat_ngram_size, no_repeat_ngram_size=args.no_repeat_ngram_size)
            print(len(input_ids), "in and", len(translations), "out")
            for input_id, translation in zip(input_ids, translations):
                translation  = tok.decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
                input_id  = tok.decode(input_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
                #print(input_id, translation)
                outf.write(translation+"\n")
                outf.flush()
                hyp.append(translation)
            ctr += 1
        sbleu = get_sacrebleu(refs, hyp)
        print("BLEU score is:", sbleu)
    elif args.decode_type == "score":
        print("Scoring translations. Will print the log probability")
        for input_ids, input_masks, decoder_input_ids, decoder_masks, labels in generate_batches_pair(tok, args): #infinite_same_sentence(10000):
            mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu))
            logits = mod_compute.logits
            print(logits.size())
            softmax = torch.nn.functional.log_softmax(logits, dim=-1)
            logprobs = nll_loss(softmax, labels.to(gpu), ignore_index=tok.pad_token_id)
            for logprob in logprobs:
                print(logprob)
                outf.write(str(logprob)+"\n")
                outf.flush()
            
    elif args.decode_type == "force_align":
        print("Getting alignments. Will print alignments for each source subword to target subword.")
        final_alignment_pos = ""
        final_alignment_str = ""
        final_src_str = ""
        final_tgt_str = ""
        for input_ids, input_masks, decoder_input_ids, tgt_masks, labels, src_sent_split, tgt_sent_split, enc_pos, dec_pos in generate_batches_pair_masked(tok, args): #infinite_same_sentence(10000):
            if enc_pos == 0:
                if final_alignment_pos != "":
                    print(final_alignment_pos)
                    print(final_alignment_str)
                    outf.write(final_src_str + "\t" + final_tgt_str + "\t" + final_alignment_pos.strip() + "\t" + final_alignment_str + "\n")
                    outf.flush()
                final_alignment_pos = ""
                final_alignment_str = ""
            mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu)) #, decoder_attention_mask=tgt_masks.to(gpu)
            logits = mod_compute.logits
            softmax = torch.nn.functional.log_softmax(logits, dim=-1)
            logprobs = nll_loss(softmax, labels.to(gpu), ignore_index=tok.pad_token_id)
            minprob = 1000
            minpos = 0
            for log_prob, dec_p in zip(logprobs, dec_pos):
                #print(log_prob, src_sent_split[enc_pos], tgt_sent_split[dec_p], log_prob < minprob)
                if log_prob < minprob:
                    minpos = dec_p
                    minprob = log_prob
            final_alignment_pos += str(enc_pos) + "-" + str(minpos) + " "
            final_alignment_str += src_sent_split[enc_pos] + "-" + tgt_sent_split[minpos] + " "
            final_src_str = " ".join(src_sent_split)
            final_tgt_str = " ".join(tgt_sent_split)
    elif args.decode_type == "get_enc_representations" or args.decode_type == "get_dec_representations":
        print("Getting encoder or decoder representations for layer "+args.layer_id+". Will save representations for each input line.")
        # default `log_dir` is "runs" - we'll be more specific here
        #writer = SummaryWriter('runs/sent_emb_run')
        #all_meta = []
        #all_hids = []
        for input_ids, input_masks, decoder_input_ids, decoder_masks, labels in generate_batches_pair(tok, args): #infinite_same_sentence(10000):
            mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu), output_hidden_states=True)
            print(input_masks)
            if args.decode_type == "get_enc_representations":
                pad_mask = input_ids.to(gpu).eq(tok.pad_token_id).unsqueeze(2)
                hidden_state = mod_compute.encoder_hidden_states[args.layer_id]
            else:
                pad_mask = decoder_input_ids.to(gpu).eq(tok.pad_token_id).unsqueeze(2)
                hidden_state = mod_compute.decoder_hidden_states[args.layer_id]
            hidden_state.masked_fill_(pad_mask, 0.0)
            print(hidden_state.size())
            hidden_state = hidden_state.mean(dim=1)
            #all_hids.append(hidden_state)    
            for idx, hidden_state_individual in enumerate(hidden_state):
                metadata=tok.decode(input_ids[idx] if args.decode_type == "get_enc_representations" else decoder_input_ids[idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                #all_meta.append(metadata)
#                 print(hidden_state_individual.size())
#                 print(metadata)
#                 writer.add_embedding(hidden_state_individual.view(1,-1),
#                             metadata=[metadata], global_step=0)
                outf.write("\t".join([str(elem) for elem in hidden_state_individual.tolist()])+"\n")
                outf.flush()
#            break
#        print(all_hids, all_meta)
#        writer.add_embedding(torch.cat(all_hids, 0), metadata=None, global_step=0)
#        writer.close()
    elif args.decode_type == "get_attention":
        print("Getting attention for layer ", args.layer_id)  
        sentence_id = 0
        for input_ids, input_masks, decoder_input_ids, decoder_masks, labels in generate_batches_pair(tok, args): 
            mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu), output_attentions=True)
            encoder_attentions = mod_compute.encoder_attentions[args.layer_id]
            decoder_attentions = mod_compute.decoder_attentions[args.layer_id]
            cross_attentions = mod_compute.cross_attentions[args.layer_id]
            for idx, (input_sent, tgt_sent) in enumerate(zip(input_ids, decoder_input_ids)):
                input_sent = tok.convert_ids_to_tokens(input_sent, skip_special_tokens=True)
                input_len = len(input_sent)
                tgt_sent = tok.convert_ids_to_tokens(tgt_sent, skip_special_tokens=True)
                tgt_len = len(tgt_sent)
                print("Processing for ", input_sent, tgt_sent)
                num_heads = 1 #encoder_attentions.size()[0] #1
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
    parser.add_argument('-m', '--model_to_decode', default='pytorch.bin', type=str, 
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
    parser.add_argument('--layer_id', default=0, type=int, help="The id of the layer from 0 to num_layers-1")
    parser.add_argument('--att_head_id', default=0, type=int, help="The id of the attention head from 0 to encoder_attention_heads-1 or decoder_attention_heads-1")
    parser.add_argument('--attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--encoder_attention_heads', default=8, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--decoder_attention_heads', default=8, type=int, help="The value for number of decoder attention heads")
    parser.add_argument('--decoder_ffn_dim', default=2048, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--encoder_ffn_dim', default=2048, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--d_model', default=512, type=int, help="The value for model hidden size")
    parser.add_argument('--max_decode_length_multiplier', default=1.5, type=float, 
                        help='This multiplied by the source sentence length will be the maximum decoding length.')
    parser.add_argument('--add_final_layer_norm', action='store_true', 
                        help='Should we add a final layer norm?')
    parser.add_argument('--normalize_before', action='store_true', 
                        help='Should we normalize before doing attention?')
    parser.add_argument('--normalize_embedding', action='store_true', 
                        help='Should we normalize embeddings?')
    parser.add_argument('--scale_embedding', action='store_true', 
                        help='Should we scale embeddings?')
    parser.add_argument('--encoder_tying_config', default=None, type=str,
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--decoder_tying_config', default=None, type=str,
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--slang', default='en', type=str, 
                        help='Source language')
    parser.add_argument('--decode_type', default='decode', type=str, 
                        help='One of decode, score, force_align, get_enc_representation, get_dec_representation or get_attention. When getting representations or attentions you must specify the index of the layer which you are interested in. By default the last layer is considered.')
    parser.add_argument('--tokenizer_name_or_path', default='ai4bharat/indic-bert', type=str, 
                        help='Name of or path to the pre-trained indic language tokenizer')
    parser.add_argument('--tlang', default='hi', type=str, 
                        help='Target language')
    parser.add_argument('--test_src', default='', type=str, 
                        help='Source language test sentences')
    parser.add_argument('--test_tgt', default='', type=str, 
                        help='Target language translated sentences')
    parser.add_argument('--test_ref', default='', type=str, 
                        help='Target language reference sentences')
    args = parser.parse_args()
    print("IP address is", args.ipaddr)
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = args.ipaddr              #
    os.environ['MASTER_PORT'] = args.port                      #
    mp.spawn(model_create_load_run_save, nprocs=args.gpus, args=(args,))         #
    #########################################################
    
if __name__ == "__main__":
    run_demo()