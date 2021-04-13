# -*- coding: utf-8 -*-
## Basic imports
import os
import argparse
import time
import sys
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
from torch.optim import Adam
##

## Our imports
from common_utils import *
##

## Other imports
import random
import numpy as np
import math
import sacrebleu
from rouge_score import rouge_scorer
##

## Seed setting here
torch.manual_seed(621311)
##

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None):
    """From fairseq. This returns the label smoothed cross entropy loss."""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.mean()
    smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

def lmap(f, x):
    """list(map(f, x)). Converts a map into a list containing (key,value) pairs."""
    return list(map(f, x))

def compute_distillation_losses(child_mod_compute, parent_mod_compute, target, ignore_index, args):
    """Implemented by me. This is based on distill bert, distill mbart etc. This method is run when the 'distillation' argument is passed.
    There are 3 types of distillation losses for now: cross_entropy, hidden_layer_regression and attention_distillation.
    cross_entropy: This minimizes the cross entropy loss between the parent distribution and the child distribution. Essentially this is different from regular cross entropy loss in the following way. Regular cross entropy is -(label(Y)*log(p_child(Y/X))) whereas this distillation loss is -(p_parent(Y/X)*log(p_child(Y/X))). We expect that the child will mimic the parent distribution.
    hidden_layer_regression: Here we choose parent to child layer mappings and minimize the hidden layer differences via the L2 (regression) loss. Simply put, for the encoder and decoder, for each layer mapping, we compute (child_hidden_representation-parent_hidden_representation)**2.
    attention_distillation: This is a rather recent approach where we compute cross entropy loss between the attention distributions of the parent (as a label) and the child. The loss is -(parent_layer_x_attention*log(child_layer_x_attention))."""
    distillation_losses_to_compute = args.distillation_styles.split(",")
    all_distillation_losses = []
    pad_mask = target.eq(ignore_index)
    for distillation_loss_to_compute in distillation_losses_to_compute:
        if distillation_loss_to_compute == "cross_entropy":
            parent_logits = parent_mod_compute.logits
            parent_lprobs = torch.nn.functional.log_softmax(parent_logits/args.softmax_temperature, dim=-1)
            child_logits = child_mod_compute.logits
            child_lprobs = torch.nn.functional.log_softmax(child_logits/args.softmax_temperature, dim=-1)
            if target.dim() == child_lprobs.dim() - 1:
                target = target.unsqueeze(-1)
    
            parent_softmax = torch.exp(parent_lprobs)
            distillation_cross_entropy = parent_softmax*child_lprobs
            distillation_cross_entropy.masked_fill_(pad_mask, 0.0)
            distillation_cross_entropy = distillation_cross_entropy.sum(dim=-1)
            distillation_cross_entropy = distillation_cross_entropy.mean() * args.softmax_temperature**2
            all_distillation_losses.append(distillation_cross_entropy)
        if distillation_loss_to_compute == "hidden_layer_regression":
            all_regression_losses = []
            for layer_mapping in args.distillation_layer_mapping.strip().split(","):
                parent_layer_idx, child_layer_idx = layer_mapping.split("-")
                parent_layer_idx, child_layer_idx = int(parent_layer_idx)-1, int(child_layer_idx)-1
                parent_encoder_layer_state = parent_mod_compute.encoder_hidden_states[parent_layer_idx]
                child_encoder_layer_state = child_mod_compute.encoder_hidden_states[child_layer_idx]
                encoder_l2_loss = (parent_encoder_layer_state-child_encoder_layer_state)**2
                encoder_l2_loss.masked_fill_(pad_mask, 0.0)
                encoder_l2_loss = encoder_l2_loss.sum(dim=-1).mean()
                parent_decoder_layer_state = parent_mod_compute.decoder_hidden_states[parent_layer_idx]
                child_decoder_layer_state = child_mod_compute.decoder_hidden_states[child_layer_idx]
                decoder_l2_loss = (parent_decoder_layer_state-child_decoder_layer_state)**2
                decoder_l2_loss.masked_fill_(pad_mask, 0.0)
                decoder_l2_loss = decoder_l2_loss.sum(dim=-1).mean()
                all_regression_losses.append(encoder_l2_loss)
                all_regression_losses.append(decoder_l2_loss)
            regression_loss = torch.mean(torch.stack(all_regression_losses), dim=0)
            all_distillation_losses.append(regression_loss)
        if distillation_loss_to_compute == "attention_distillation":
            all_attention_distillation_losses = []
            for layer_mapping in args.distillation_layer_mapping.strip().split(","):
                parent_layer_idx, child_layer_idx = layer_mapping.split("-")
                parent_layer_idx, child_layer_idx = int(parent_layer_idx)-1, int(child_layer_idx)-1
                parent_encoder_self_attention = parent_mod_compute.encoder_attentions[parent_layer_idx]
                child_encoder_self_attention = child_mod_compute.encoder_attentions[child_layer_idx]
                # deal with padding here. We will need to access the source token padding information.
                encoder_sa_loss = parent_encoder_attention*torch.log(child_encoder_attention.masked_fill_(child_encoder_attention.eq(0.0), 1e-10))
                encoder_l2_loss = encoder_l2_loss.sum(dim=-1).mean()
                parent_decoder_self_attention = parent_mod_compute.decoder_attentions[parent_layer_idx]
                child_decoder_self_attention = child_mod_compute.decoder_attentions[child_layer_idx]
                decoder_sa_loss = parent_decoder_self_attention*torch.log(child_decoder_self_attention.masked_fill_(child_decoder_self_attention.eq(0.0), 1e-10))
                decoder_sa_loss = decoder_sa_loss.sum(dim=-1).mean()
                parent_decoder_cross_attention = parent_mod_compute.cross_attentions[parent_layer_idx]
                child_decoder_cross_attention = child_mod_compute.cross_attentions[child_layer_idx]
                decoder_ca_loss = parent_decoder_cross_attention*torch.log(child_decoder_cross_attention.masked_fill_(child_decoder_cross_attention.eq(0.0), 1e-10))
                decoder_ca_loss = decoder_ca_loss.sum(dim=-1).mean()
                all_attention_distillation_losses.append(encoder_sa_loss)
                all_attention_distillation_losses.append(decoder_sa_loss)
                all_attention_distillation_losses.append(decoder_ca_loss)
            all_attention_distillation_losses = torch.mean(torch.stack(all_attention_distillation_losses), dim=0)
            all_distillation_losses.append(all_attention_distillation_losses)
        
    return -torch.mean(torch.stack(all_distillation_losses), dim=0)

def remap_layers(model, idx, args): ### Cut this code into half.
    """This method is used to remap the layers from a pretrained model to the current model. The remapping info comes in the form of 2-1,... which means, map the second layer of the pretrained model to the first layer of the current model."""
    print("Remapping layers from parent to child.")
    if args.remap_encoder != "":
        keys_to_consider = [key for key in model.keys() if "encoder" in key]
        for mapping in args.remap_encoder.split(","):
            slayer, tlayer = mapping.split("-")
            for key in keys_to_consider:
                key = key.strip().split(".")
                key_copy = list(key)
                if key[idx] == slayer:
                    key_copy[idx] =tlayer
                    key = ".".join(key)
                    key_copy = ".".join(key_copy)
                    model[key] = model[key_copy]
                    del model[key_copy]
    if args.remap_decoder != "":
        keys_to_consider = [key for key in model.keys() if "decoder" in key]
        for mapping in args.remap_encoder.split(","):
            slayer, tlayer = mapping.split("-")
            for key in keys_to_consider:
                key = key.strip().split(".")
                key_copy = list(key)
                if key[idx] == slayer:
                    key_copy[idx] =tlayer
                    key = ".".join(key)
                    key_copy = ".".join(key_copy)
                    model[key] = model[key_copy]
                    del model[key_copy]
    return model

def remap_embeddings(our_model_dict, model_to_load_dict, args):
    """This method will consider two tokenizers, one for the pretrained model and one for the current model. It will then remap the embeddings"""
    print("Remapping embeddings.")
    if args.pretrained_tokenizer_name_or_path is None:
        return model_to_load_dict
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=False, use_fast=False, keep_accents=True).get_vocab()
    tok_pre = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_name_or_path, do_lower_case=False, use_fast=False, keep_accents=True).get_vocab()
    for token in tok:
        tok_idx = tok[token]
        if token in tok_pre:
            pre_tok_idx = tok_pre[token]
            our_model_dict["module.model.shared.weight"][tok_idx] = model_to_load_dict["module.model.shared.weight"][pre_tok_idx]
    model_to_load_dict["module.model.shared.weight"] = our_model_dict["module.model.shared.weight"]
    return model_to_load_dict

def remap_embeddings_and_eliminate_mismatches(our_model_dict, model_to_load_dict, args):
    """This method first remaps embeddings from pretrained to current model and then eliminates mismatched layers between the pretrained model and the current model. A mismatch is when the size of the pretrained parameter is not the same as the parameter of the current model."""
    model_to_load_dict = remap_embeddings(our_model_dict, model_to_load_dict, args)
    print("Eliminating matched params with mismatched sizes from the initial model.")
    for our_model_key in our_model_dict:
        if our_model_key in model_to_load_dict:
            if our_model_dict[our_model_key].size() != model_to_load_dict[our_model_key].size():
                print("Eliminating", our_model_key)
                del model_to_load_dict[our_model_key]
    return model_to_load_dict

def init_weights(module, in_features, out_features):
    """Method to initialize model weights. Not used for now but might be used in the future. Tries to mimic t2t initialization.
    TODO: Incorporate this into the flow so as to give users an option to do their own initialization."""
    if isinstance(module, nn.Linear):
        init_std = (3.0/(in_features+out_features))**(0.5)
        module.weight.data.normal_(mean=0.0, std=init_std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        init_std = (3.0/(out_features))**(0.5)
        module.weight.data.normal_(mean=0.0, std=init_std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
            
def shard_files_mono(files, world_size):
    """This method shards files into N parts containing the same number of lines. Each shard will go to a different GPU which may even be located on another machine. This method is run when the 'shard_files' argument is passed."""
    print("Sharding files into", world_size, "parts")
    for lang in files:
        infile = open(files[lang]).readlines()
        num_lines = len(infile)
        lines_per_shard = math.ceil(num_lines/world_size)
        print("For language:",lang," the total number of lines are:", num_lines, "and number of lines per shard are:", lines_per_shard)
        for shard_id in range(world_size):
            outfile = open(files[lang]+"."+"%02d" % shard_id, "w")
            for line in infile[shard_id*lines_per_shard:(shard_id+1)*lines_per_shard]:
                outfile.write(line)
            outfile.flush()
            outfile.close()
        print("File for language", lang, "has been sharded.")
        sys.stdout.flush()

def shard_files_bi(files, world_size):
    """This method shards files into N parts containing the same number of lines. Each shard will go to a different GPU which may even be located on another machine. This method is run when the 'shard_files' argument is passed."""
    print("Sharding files into", world_size, "parts")
    for pair in files:
        infile = list(zip(open(files[pair][0]).readlines(), open(files[pair][1]).readlines()))
        num_lines = len(infile)
        lines_per_shard = math.ceil(num_lines/world_size)
        print("For language pair:",pair," the total number of lines are:", num_lines, "and number of lines per shard are:", lines_per_shard)
        for shard_id in range(world_size):
            srcoutfile = open(files[pair][0]+"."+"%02d" % shard_id, "w")
            tgtoutfile = open(files[pair][1]+"."+"%02d" % shard_id, "w")
            for src_line, tgt_line in infile[shard_id*lines_per_shard:(shard_id+1)*lines_per_shard]:
                srcoutfile.write(src_line)
                tgtoutfile.write(tgt_line)
            srcoutfile.flush()
            srcoutfile.close()
            tgtoutfile.flush()
            tgtoutfile.close()
        print("File for language pair", pair, "has been sharded.")
        sys.stdout.flush()
        
def get_sacrebleu(refs, hyp):
    """Returns sacrebleu score. Sacrebleu is a reliable implementation for computing corpus level BLEU scores."""
    bleu = sacrebleu.corpus_bleu(hyp, refs)
    return bleu.score

def yield_corpus_indefinitely_mono(corpus, lang):
    """This shuffles the corpus or corpus shard at the beginning of each epoch and returns sentences indefinitely."""
    epoch_counter = 0
    try:
        while True:
            print("Shuffling corpus!")
            sys.stdout.flush()
            random.shuffle(corpus)
            for src_line in corpus:
                yield src_line

            epoch_counter += 1
            print("Finished epoch", epoch_counter, "for language:", lang)
    except Exception as e:
        print(e)
        print("Catastrophic data gen failure")
    return None

def yield_corpus_indefinitely_bi(corpus, language):
    """This shuffles the corpus at the beginning of each epoch and returns sentences indefinitely."""
    epoch_counter = 0
    while True:
        print("Shuffling corpus:", language)
        random.shuffle(corpus)
        for src_line, tgt_line in corpus:
            yield src_line, tgt_line
        
        epoch_counter += 1
        print("Finished epoch", epoch_counter, "for language:", language)
    return None, None ## We should never reach this point.
