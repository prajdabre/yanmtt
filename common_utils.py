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
import argparse
import time
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
##

## Huggingface imports
import transformers
from transformers import AutoTokenizer, MBartTokenizer, MBart50Tokenizer, BartTokenizer
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
import torch.nn.functional as F
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
import functools
import matplotlib.pyplot as plt  # drawing heat map of attention weights
from matplotlib import rcParams
import matplotlib.colors as mcolors
rcParams['font.sans-serif'] = ['Source Han Sans TW',
                                   'sans-serif',
                                   "FreeSerif"  # fc-list :lang=hi family
                                   ]
from copy import deepcopy
##

## Seed setting here
torch.manual_seed(621311)
##


class EWC(object):
    def __init__(self, model, dataset, gpu, label_smoothing, ignore_index=None):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.gpu = gpu
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self._precision_matrices = self._diag_fisher()
        
        for n, p in deepcopy(self.params).items():
            self._means[n] = torch.tensor(p.detach().cpu().numpy()).to(gpu)
            self._means[n].requires_grad = False

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = torch.tensor(p.detach().cpu().numpy()).to(self.gpu)            

        self.model.eval()
        num_samples = 0
        for input_ids, input_masks, decoder_input_ids, labels in self.dataset:
            self.model.zero_grad(set_to_none=True)
            input_ids = input_ids.to(self.gpu)
            input_masks = input_masks.to(self.gpu)
            decoder_input_ids = decoder_input_ids.to(self.gpu)
            labels = labels.to(self.gpu)
            num_samples += input_ids.size(0)
            output = self.model(input_ids=input_ids, attention_mask=input_masks ,decoder_input_ids=decoder_input_ids)
            lprobs = torch.nn.functional.log_softmax(output.logits, dim=-1) ## Softmax tempering of logits if needed.
            loss = label_smoothed_nll_loss(lprobs, labels, self.label_smoothing, self.ignore_index)
            loss.backward()
            loss.detach()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2
        
        for n, p in self.model.named_parameters():
            precision_matrices[n].data = precision_matrices[n].data / num_samples
        
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        for n, p in precision_matrices.items():
            precision_matrices[n].requires_grad = False
        
        self.model.zero_grad(set_to_none=True)    
        self.model.train()
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

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
        denominator = (1.0 - 1.0*pad_mask)
        denominator = denominator.sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        denominator = 1.0
    
    if ignore_index is not None:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    else:
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
        
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    loss = loss/denominator
    return loss

def prune_weights(model_weight_dict, prune_ratio):
    """Prunes the weights of the model.
    Args:
        model_weight_dict: The weights of the model.
        prune_ratio: The ratio of the weights to be pruned.
    Returns:
        The pruned weights which will be used to initialize the model.
        
    Logic:
        1. For each key in the model_weight_dict, get the weight tensor and flatten it.
        2. Sort the weight tensor in ascending order and get the indices of the top prune_ratio*len(weight_tensor) elements.
        3. Set the weights to zero at the indices obtained in step 2."""
    if prune_ratio <= 0:
        return model_weight_dict
    
    print("Pruning weights. Pruning percent: {}%".format(prune_ratio*100))
    for key in model_weight_dict:
        weight_tensor = model_weight_dict[key]
        weight_tensor = weight_tensor.view(-1)
        weight_tensor = weight_tensor.cpu().numpy()
        indices = np.argsort(weight_tensor)
        indices = indices[:int(prune_ratio*len(weight_tensor))]
        weight_tensor[indices] = 0
        weight_tensor = torch.from_numpy(weight_tensor)
        print("Pruned percentage: {}%".format(torch.sum(weight_tensor == 0)/len(weight_tensor)*100))
        weight_tensor = weight_tensor.view_as(model_weight_dict[key])
        model_weight_dict[key] = weight_tensor
    return model_weight_dict


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
    if target.dim() == child_mod_compute.logits.dim() - 1:
        target = target.unsqueeze(-1)
    pad_mask = target.eq(ignore_index)
            
    for distillation_loss_to_compute in distillation_losses_to_compute:
        if distillation_loss_to_compute == "cross_entropy":
            parent_logits = parent_mod_compute.logits
            parent_lprobs = torch.nn.functional.log_softmax(parent_logits/args.distillation_temperature, dim=-1)
            child_logits = child_mod_compute.logits
            child_lprobs = torch.nn.functional.log_softmax(child_logits/args.distillation_temperature, dim=-1)
            parent_softmax = torch.exp(parent_lprobs)
            distillation_cross_entropy = parent_softmax*child_lprobs
            distillation_cross_entropy.masked_fill_(pad_mask, 0.0)
            distillation_cross_entropy = distillation_cross_entropy.sum(dim=-1)
            distillation_cross_entropy = distillation_cross_entropy.mean() * args.distillation_temperature**2
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
            all_distillation_losses.append(-regression_loss) ## We will take a negative later so this minus sign here is to negate its effect. We want to minimize the L2 loss after all.
        if distillation_loss_to_compute == "attention_distillation":
            all_attention_distillation_losses = []
            for layer_mapping in args.distillation_layer_mapping.strip().split(","):
                parent_layer_idx, child_layer_idx = layer_mapping.split("-")
                parent_layer_idx, child_layer_idx = int(parent_layer_idx)-1, int(child_layer_idx)-1
                parent_encoder_self_attention = parent_mod_compute.encoder_attentions[parent_layer_idx]
                child_encoder_self_attention = child_mod_compute.encoder_attentions[child_layer_idx]
                # deal with padding here. We will need to access the source token padding information.
                encoder_sa_loss = parent_encoder_self_attention*torch.log(child_encoder_self_attention.masked_fill_(child_encoder_self_attention.eq(0.0), 1e-10))
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
    model_copy = model.copy()
    if args.remap_encoder != "":
        keys_to_consider = [key for key in model.keys() if ".encoder.layers" in key]
        keys_to_keep = set() ## Keys to keep in the model dict once remapping is done as we assume that the user always specifies ALL desired target model keys to be remapped.
        for mapping in args.remap_encoder.split(","):
            slayer, tlayer = mapping.split("-")
            slayer = str(int(slayer)-1) # Zero indexing
            tlayer = str(int(tlayer)-1) # Zero indexing
            keys_to_keep.add(slayer) ## Key remapped so it should not be deleted
            for key in keys_to_consider:
                key = key.strip().split(".")
                key_copy = list(key)
                if key[idx] == slayer:
                    print("Remapping", key)
                    key_copy[idx] =tlayer
                    key = ".".join(key)
                    key_copy = ".".join(key_copy)
                    model[key] = model_copy[key_copy]
        for key in keys_to_consider: ## Purge all unspecified keys.
            key = key.strip().split(".")
            if key[idx] not in keys_to_keep:
                key = ".".join(key)
                print("Deleting", key)
                del model[key]

    if args.remap_decoder != "":
        keys_to_consider = [key for key in model.keys() if ".decoder.layers" in key]
        keys_to_keep = set() ## Keys to keep in the model dict once remapping is done as we assume that the user always specifies ALL desired target model keys to be remapped.
        for mapping in args.remap_encoder.split(","):
            slayer, tlayer = mapping.split("-")
            slayer = str(int(slayer)-1) # Zero indexing
            tlayer = str(int(tlayer)-1) # Zero indexing
            keys_to_keep.add(slayer) ## Key remapped so it should not be deleted
            for key in keys_to_consider:
                key = key.strip().split(".")
                key_copy = list(key)
                if key[idx] == slayer:
                    print("Remapping", key)
                    key_copy[idx] =tlayer
                    key = ".".join(key)
                    key_copy = ".".join(key_copy)
                    model[key] = model_copy[key_copy]
        for key in keys_to_consider: ## Purge all unspecified keys.
            key = key.strip().split(".")
            if key[idx] not in keys_to_keep:
                key = ".".join(key)
                print("Deleting", key)
                del model[key]
    print("Final model dictionary after remapping is:", model.keys())
    return model

def remap_embeddings(our_model_dict, model_to_load_dict, args):
    """This method will consider two tokenizers, one for the pretrained model and one for the current model. It will then remap the embeddings. When we remapt embeddings we not only remap input embeddings to the encoder and decoder but also the lm head parameters which is a kind of embedding consisting of a weight matrix and biases. Note that embed positions remapping makes no sense."""
    if args.pretrained_tokenizer_name_or_path is None:
        return model_to_load_dict
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=False, use_fast=False, keep_accents=True).get_vocab()
    tok_pre = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_name_or_path, do_lower_case=False, use_fast=False, keep_accents=True).get_vocab()
    for token in tok:
        tok_idx = tok[token]
        if token in tok_pre:
            pre_tok_idx = tok_pre[token]
            our_model_dict["module.model.shared.weight"][tok_idx] = model_to_load_dict["module.model.shared.weight"][pre_tok_idx]
            our_model_dict["module.model.encoder.embed_tokens.weight"][tok_idx] = model_to_load_dict["module.model.encoder.embed_tokens.weight"][pre_tok_idx]
            our_model_dict["module.model.decoder.embed_tokens.weight"][tok_idx] = model_to_load_dict["module.model.decoder.embed_tokens.weight"][pre_tok_idx]
            our_model_dict["module.lm_head.weight"][tok_idx] = model_to_load_dict["module.lm_head.weight"][pre_tok_idx]
            our_model_dict["module.final_logits_bias"][tok_idx] = model_to_load_dict["module.final_logits_bias"][pre_tok_idx]
    model_to_load_dict["module.model.shared.weight"] = our_model_dict["module.model.shared.weight"]
    model_to_load_dict["module.model.encoder.embed_tokens.weight"] = our_model_dict["module.model.encoder.embed_tokens.weight"]
    model_to_load_dict["module.model.decoder.embed_tokens.weight"] = our_model_dict["module.model.decoder.embed_tokens.weight"]
    model_to_load_dict["module.lm_head.weight"] = our_model_dict["module.lm_head.weight"]
    model_to_load_dict["module.final_logits_bias"] = our_model_dict["module.final_logits_bias"]
    return model_to_load_dict

def remap_embeddings_eliminate_components_and_eliminate_mismatches(our_model_dict, model_to_load_dict, args):
    """This method first remaps embeddings from pretrained to current model and then eliminates mismatched layers between the pretrained model and the current model. A mismatch is when the size of the pretrained parameter is not the same as the parameter of the current model."""
    print("Remapping embeddings.")
    model_to_load_dict = remap_embeddings(our_model_dict, model_to_load_dict, args)
    
    if args.eliminate_encoder_before_initialization:
        print("Eliminating encoder from the model to load")
        for load_model_key in model_to_load_dict:
            if "encoder" in load_model_key:
                del model_to_load_dict[load_model_key]
    if args.eliminate_decoder_before_initialization:
        print("Eliminating decoder from the model to load")
        for load_model_key in model_to_load_dict:
            if "decoder" in load_model_key:
                del model_to_load_dict[load_model_key]
    if args.eliminate_embeddings_before_initialization:
        print("Eliminating embeddings from the model to load")
        for load_model_key in model_to_load_dict:
            if "embed" in load_model_key:
                del model_to_load_dict[load_model_key]            
    
                
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
            
def shard_files_mono(files, args):
    """This method shards files into N parts containing the same number of lines. Each shard will go to a different GPU which may even be located on another machine. This method is run when the 'shard_files' argument is passed."""
    print("Sharding files into", args.world_size, "parts")
    for lang in files:
        infile = open(files[lang][0]).readlines() if args.num_domains_for_domain_classifier > 1 else open(files[lang]).readlines()
        num_lines = len(infile)
        lines_per_shard = math.ceil(num_lines/args.world_size)
        print("For language:",lang," the total number of lines are:", num_lines, "and number of lines per shard are:", lines_per_shard)
        for shard_id in range(args.world_size):
            outfile = open(files[lang][0]+"."+"%02d" % shard_id, "w") if args.num_domains_for_domain_classifier > 1 else open(files[lang]+"."+"%02d" % shard_id, "w")
            for line in infile[shard_id*lines_per_shard:(shard_id+1)*lines_per_shard]:
                outfile.write(line)
            outfile.flush()
            outfile.close()
        print("File for language", lang, "has been sharded.")
        sys.stdout.flush()

def shard_files_mono_lm(files, args):
    """This method shards files into N parts containing the same number of lines. Each shard will go to a different GPU which may even be located on another machine. This method is run when the 'shard_files' argument is passed."""
    print("Sharding files into", args.world_size, "parts")
    for lang in files:
        infile = open(files[lang]).readlines()
        num_lines = len(infile)
        lines_per_shard = math.ceil(num_lines/args.world_size)
        print("For language:",lang," the total number of lines are:", num_lines, "and number of lines per shard are:", lines_per_shard)
        for shard_id in range(args.world_size):
            outfile = open(files[lang]+"."+"%02d" % shard_id, "w")
            for line in infile[shard_id*lines_per_shard:(shard_id+1)*lines_per_shard]:
                outfile.write(line)
            outfile.flush()
            outfile.close()
        print("File for language", lang, "has been sharded.")
        sys.stdout.flush()
        
def shard_files_bi(files, args):
    """This method shards files into N parts containing the same number of lines. Each shard will go to a different GPU which may even be located on another machine. This method is run when the 'shard_files' argument is passed."""
    print("Sharding files into", args.world_size, "parts")
    for pair in files:
        infile = list(zip(open(files[pair][0]).readlines(), open(files[pair][1]).readlines()))
        num_lines = len(infile)
        lines_per_shard = math.ceil(num_lines/args.world_size)
        print("For language pair:",pair," the total number of lines are:", num_lines, "and number of lines per shard are:", lines_per_shard)
        for shard_id in range(args.world_size):
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

def yield_corpus_indefinitely_mono(corpus, lang, sorted_batching):
    """This shuffles the corpus or corpus shard at the beginning of each epoch and returns sentences indefinitely."""
    epoch_counter = 0
    num_lines = len(corpus)
    num_sentences_before_sort = 20000
    num_sorted_segments = (num_lines // num_sentences_before_sort) + 1
    try:
        while True:
            print("Shuffling corpus!")
            sys.stdout.flush()
            random.shuffle(corpus)
            if sorted_batching:
                for curr_segment_id in range(num_sorted_segments):
                    curr_segment = corpus[curr_segment_id*num_sentences_before_sort:(curr_segment_id+1)*num_sentences_before_sort]
                    for src_line in sorted(curr_segment, key=len):
                        yield src_line
            else:
                for src_line in corpus:
                    yield src_line
            epoch_counter += 1
            print("Finished epoch", epoch_counter, "for language:", lang)
    except Exception as e:
        print(e)
        print("Catastrophic data gen failure")
    return None

def yield_corpus_indefinitely_bi(corpus, language, sorted_batching):
    """This shuffles the corpus at the beginning of each epoch and returns sentences indefinitely."""
    epoch_counter = 0
    num_lines = len(corpus)
    num_sentences_before_sort = 20000
    num_sorted_segments = (num_lines // num_sentences_before_sort) + 1
    while True:
        print("Shuffling corpus:", language)
        random.shuffle(corpus)
        sys.stdout.flush()
        if sorted_batching:
            for curr_segment_id in range(num_sorted_segments):
                curr_segment = corpus[curr_segment_id*num_sentences_before_sort:(curr_segment_id+1)*num_sentences_before_sort]
                for src_line, tgt_line in sorted(curr_segment, key=lambda x: len(x[1])):
                    yield src_line, tgt_line
        else:
            for src_line, tgt_line in corpus:
                yield src_line, tgt_line
        epoch_counter += 1
        print("Finished epoch", epoch_counter, "for language:", language)
    return None, None ## We should never reach this point.

def sub_sample_and_permute_document(sentence, document_level_sentence_delimiter, max_length):
    """Here we start at a particular random index and select the rest of the sentences. This is to make sure that we dont always see only the initial part of each document all the time."""
    sentence_split = sentence.split(" "+document_level_sentence_delimiter+" ")
    sentence_split_length = len(sentence_split)
    num_delimiters = sentence_split_length - 1
    start_idx = random.randint(0, sentence_split_length-1)
    sentence_split = sentence_split[start_idx:]
    sentence = (" "+document_level_sentence_delimiter+" ").join(sentence_split)
    sentence_split = sentence.split(" ")
    sent_len = len(sentence_split)
    if sent_len > max_length: ## Initial truncation
        sentence_split = sentence_split[:max_length]
        sentence = " ".join(sentence_split)
        sent_len = max_length
    sentence_split = sentence.split(" "+document_level_sentence_delimiter+" ")
    sentence_split_shuffled = random.sample(sentence_split, len(sentence_split))
    sentence_split_shuffled = (" "+document_level_sentence_delimiter+" ").join(sentence_split_shuffled)
    sentence_split_shuffled = sentence_split_shuffled.split(" ")
    return sentence_split_shuffled, sentence, sent_len

    
def generate_batches_monolingual_masked(tok, args, files, rank):
    """Generates the source, target and source attention masks for denoising. Long sequences are truncated and short sequences are ignored."""
    
    if args.tokenization_sampling:
        print("Stochastic tokenizer will be used.")
        if "mbart" in args.tokenizer_name_or_path:
            print("BPE dropout with a dropout probability of", args.tokenization_alpha_or_dropout, "will be used.")
        else:
            print("Sentencepiece regularization with an alpha value of", args.tokenization_alpha_or_dropout, "will be used.")
    
    batch_count = 0
    if args.use_official_pretrained:
        mask_tok = "<mask>"
    else:
        mask_tok = "[MASK]"
    if len(args.token_masking_probs_range) == 1:
        mp_val_or_range = args.token_masking_probs_range[0]
    elif len(args.token_masking_probs_range) == 2:
        mp_val_or_range = args.token_masking_probs_range
    print("Masking ratio:", mp_val_or_range)
    language_list = list(files.keys())
    print("Training for:", language_list)
    language_file_dict = {}
    probs = {}
    for l in language_list:
        file_content = open(files[l][0]+"."+"%02d" % rank).readlines() if args.num_domains_for_domain_classifier > 1 else open(files[l]+"."+"%02d" % rank).readlines()
        probs[l] = len(file_content)
        language_file_dict[l] = yield_corpus_indefinitely_mono(file_content, l, args.sorted_batching)
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = probs_temp
    probs_temp = {lang: probs[lang]**(1.0/args.data_sampling_temperature) for lang in probs} ## Temperature sampling probabilities.
    probs = probs_temp
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = [probs_temp[lang] for lang in language_list]
    dropped_sentence = "" ## We will save the sentence to be dropped this batch and add it to the next batch.
    dropped_language = "" ## We will save the language to be dropped this batch and add it to the next batch.
    while batch_count != args.num_batches:
        curr_batch_count = 0
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_label_batch = []
        batch_count += 1
        max_src_sent_len = 0
        max_tgt_sent_len = 0
        start = time.time()
        sents_in_batch = 0
        if args.num_domains_for_domain_classifier > 1:
            domain_classifier_labels = []
        while True:
            if dropped_sentence != "":
                language = dropped_language # Reuse the previous language
                sentence = dropped_sentence # Reuse the previous sentence
                dropped_sentence = ""
                dropped_language = ""
            else:
                language = random.choices(language_list, probs)[0]
                sentence = next(language_file_dict[language]).strip()
            if args.num_domains_for_domain_classifier > 1: ## Careful when handling domains for monolingual corpora.
                lang = language.strip().split("-")[0]
                lang = lang if args.use_official_pretrained else "<2"+lang+">"
            else:
                lang = language if args.use_official_pretrained else "<2"+language+">"
            if type(mp_val_or_range) is float:
                mask_percent = mp_val_or_range
            else:
                mask_percent = random.uniform(mp_val_or_range[0], mp_val_or_range[1])
            if args.is_document:
                sentence_split, sentence, sent_len = sub_sample_and_permute_document(sentence, args.document_level_sentence_delimiter, args.max_length)
            else:
                sentence_split = sentence.split(" ")
                sent_len = len(sentence_split)
                if sent_len < 1: 
                    continue
                if sent_len > args.max_length: ## Initial truncation
                    sentence_split = sentence_split[:args.max_length]
                    sentence = " ".join(sentence_split)
                    sent_len = args.max_length
            mask_count = 0
            max_mask_count = int(mask_percent*sent_len)
            spans_to_mask = list(np.random.poisson(args.token_masking_lambda, 1000))
            curr_sent_len = sent_len
            while mask_count < max_mask_count:
                try:
                    span_to_mask = spans_to_mask[0]
                    del spans_to_mask[0]
                    if span_to_mask > (max_mask_count-mask_count): ## Cant mask more than the allowable number of tokens.
                        continue
                    idx_to_mask = random.randint(sent_len//2 if args.future_prediction else 0, (curr_sent_len-1)-(span_to_mask-1)) ## We mask only the remaining half of the sentence to encourage the model to learn representations that can make do without most of the future tokens.
                    if mask_tok not in sentence_split[idx_to_mask:idx_to_mask+span_to_mask] and args.document_level_sentence_delimiter not in sentence_split[idx_to_mask:idx_to_mask+span_to_mask]:
                        actually_masked_length = len(sentence_split[idx_to_mask:idx_to_mask+span_to_mask]) ## If at the end of the sentence then we have likely masked fewer tokens.
                        sentence_split[idx_to_mask:idx_to_mask+span_to_mask] = [mask_tok]
                        mask_count += actually_masked_length # We assume that with a low probability there are mask insertions when span lengths are 0 which may cause more mask tokens than planned. I have decided not to count these insersions towards the maximum maskable limit. This means that the total number of mask tokens will be a bit higher than what it should be. 
                        curr_sent_len -= (actually_masked_length-1)
                except:
                    break ## If we cannot get a properly masked sentence despite all our efforts then we just give up and continue with what we have so far.
            
            masked_sentence = " ".join(sentence_split)
            if args.span_prediction or args.span_to_sentence_prediction: ## We only predict the masked spans and not other tokens.
                masked_sentence_split = masked_sentence.split(mask_tok)
                final_sentence = ""
                prev_idx = 0
                for span in masked_sentence_split:
                    if span.strip() != "":
                        extracted = sentence[prev_idx:prev_idx+sentence[prev_idx:].index(span)]
                        final_sentence += extracted + " <s> " ## Separate the predictions.
                        prev_idx=prev_idx+len(extracted) + len(span)
                        
                final_sentence += sentence[prev_idx:]
                final_sentence = final_sentence.strip()
                if args.span_to_sentence_prediction:
                    masked_sentence = final_sentence
                else:
                    sentence = final_sentence
            
            
            if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                iids = tok(masked_sentence, return_tensors="pt").input_ids
                curr_src_sent_len = len(iids[0])
                if curr_src_sent_len > args.hard_truncate_length:
                    masked_sentence = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_src_sent_len = args.hard_truncate_length
                iids = tok(sentence, return_tensors="pt").input_ids
                curr_tgt_sent_len = len(iids[0])
                if curr_tgt_sent_len > args.hard_truncate_length:
                    sentence = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_tgt_sent_len = args.hard_truncate_length
            else:
                iids = tok(lang + " " + masked_sentence + " </s>", add_special_tokens=False, return_tensors="pt").input_ids
                curr_src_sent_len = len(iids[0])
                if curr_src_sent_len > args.hard_truncate_length:
                    masked_sentence = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_src_sent_len = args.hard_truncate_length
                iids = tok("<s> " + sentence, add_special_tokens=False, return_tensors="pt").input_ids
                curr_tgt_sent_len = len(iids[0])
                if curr_tgt_sent_len > args.hard_truncate_length:
                    sentence = tok.decode(iids[0][1:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_tgt_sent_len = args.hard_truncate_length
            
                        
            if curr_src_sent_len > max_src_sent_len:
                max_src_sent_len = curr_src_sent_len
            
            if curr_tgt_sent_len > max_tgt_sent_len:
                max_tgt_sent_len = curr_tgt_sent_len
            
            if args.batch_size_indicates_lines: ## Batch a fixed number of sentences. We can safely add the current example because we assume that the user knows the max batch size.
                if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                    encoder_input_batch.append(masked_sentence)
                    decoder_input_batch.append(sentence)
                else:
                    if args.use_official_pretrained and "50" in args.pretrained_model: ## mbart-50 model has a different input representation
                        encoder_input_batch.append(lang + " " + masked_sentence + " </s>")
                    else:
                        encoder_input_batch.append(masked_sentence + " </s> " + lang)
                    decoder_input_batch.append(lang + " " + sentence)
                    decoder_label_batch.append(sentence + " </s>")
                    if args.tokenization_sampling:
                        decoder_input_batch[-1] += " </s>" ## In case of stochastic subword segmentation we have to generate the decoder label ids from the decoder input ids. This will make the model behavior sliiiiiightly different from how it was when stochastic decoding is not done. However it should have no major difference. In the non stochastic case, the </s> or EOS token is never fed as the input but in the stochastic case that token is fed as the input. So this means in the non stochastic case, the end of decoder and labels will be something like: "a good person <pad> <pad> <pad>" and "good person </s> <pad <pad> <pad>". In the stochastic case, the end of decoder and labels will be something like: "a good person </s> <pad> <pad>" and "good person </s> <pad> <pad <pad>". Now think about how we compute loss. When the label is a pad, the loss is never propagated through it. So the model will never learn anything when </s> or <pad> will be the input. Furthermore, the generation of </s> is what dictates the end of generation. When the model generates a </s> the sequence is taken out of the computation process and therefore whatever it generates after that will not be considered at all towards its final score. In conclusion, there should be no practical difference in outcomes between the two batching approaches.
                sents_in_batch += 1
                if args.num_domains_for_domain_classifier > 1:
                    domain_classifier_labels.append(files[language][1])
                if sents_in_batch == args.batch_size:
                    break
            else:
                potential_batch_count = max(max_src_sent_len, max_tgt_sent_len)*(sents_in_batch+1) ## Note that this will be unreliable when we do stochastic subword segmentation.
                if potential_batch_count > args.batch_size: ## We will drop this sentence for now because we may go over the limit of what the GPU can handle. It may be used in a future iteration. Note that this will be unreliable when we do stochastic subword segmentation.
                    if curr_src_sent_len > args.batch_size or curr_tgt_sent_len > args.batch_size:
                        dropped_sentence = "" ## Dangerous sentence detected. Exterminate with extreme prejudice!
                        dropped_language = ""
                    else:
                        dropped_sentence = sentence
                        dropped_language = language
                    break
                if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                    encoder_input_batch.append(masked_sentence)
                    decoder_input_batch.append(sentence)
                else:
                    if args.use_official_pretrained and "50" in args.pretrained_model: ## mbart-50 model has a different input representation
                        encoder_input_batch.append(lang + " " + masked_sentence + " </s>")
                    else:
                        encoder_input_batch.append(masked_sentence + " </s> " + lang)
                    decoder_input_batch.append(lang + " " + sentence)
                    decoder_label_batch.append(sentence + " </s>")
                    if args.tokenization_sampling:
                        decoder_input_batch[-1] += " </s>" ## In case of stochastic subword segmentation we have to generate the decoder label ids from the decoder input ids. This will make the model behavior sliiiiiightly different from how it was when stochastic decoding is not done. However it should have no major difference. In the non stochastic case, the </s> or EOS token is never fed as the input but in the stochastic case that token is fed as the input. So this means in the non stochastic case, the end of decoder and labels will be something like: "a good person <pad> <pad> <pad>" and "good person </s> <pad <pad> <pad>". In the stochastic case, the end of decoder and labels will be something like: "a good person </s> <pad> <pad>" and "good person </s> <pad> <pad <pad>". Now think about how we compute loss. When the label is a pad, the loss is never propagated through it. So the model will never learn anything when </s> or <pad> will be the input. Furthermore, the generation of </s> is what dictates the end of generation. When the model generates a </s> the sequence is taken out of the computation process and therefore whatever it generates after that will not be considered at all towards its final score. In conclusion, there should be no practical difference in outcomes between the two batching approaches.
                if args.num_domains_for_domain_classifier > 1:
                    domain_classifier_labels.append(files[language][1])
                sents_in_batch += 1
        
        if len(encoder_input_batch) == 0:
            print("Zero size batch due to an abnormal example. Skipping empty batch.")
            continue
        if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit. No support for stochastic tokenizer because the roberta tokenizer which is inherited from GPT2 tokenizer does its onw weird BPE and I dont want to mess with it.
            input_ids = tok(encoder_input_batch, return_tensors="pt", padding=True).input_ids
        else:
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
        # if len(input_ids[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
        #     input_ids = input_ids[:,:args.hard_truncate_length]
        input_masks = (input_ids != tok.pad_token_id).int()
        if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit. No support for stochastic tokenizer because the roberta tokenizer which is inherited from GPT2 tokenizer does its onw weird BPE and I dont want to mess with it.
            decoder_input_ids = tok(decoder_input_batch, return_tensors="pt", padding=True).input_ids
        else:
            decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
        # if len(decoder_input_ids[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
        #     decoder_input_ids = decoder_input_ids[:,:args.hard_truncate_length]
        if (args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model) or args.tokenization_sampling: ## We have to be careful when using stochastic segmentation. Note again that there will be no stoachastic segmentation with the official bart model. IT JUST WONT WORK.
            labels = decoder_input_ids[:,1:]
            decoder_input_ids = decoder_input_ids[:,:-1]
        else:
            labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        # if len(labels[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
        #     labels = labels[:,:args.hard_truncate_length]
        end = time.time()
#         if rank == 0:
#             print(input_ids.size(), functools.reduce(lambda x,y: x*y, input_ids.size()), decoder_input_ids.size(), functools.reduce(lambda x,y: x*y, decoder_input_ids.size()))
        if args.num_domains_for_domain_classifier > 1:
            yield input_ids, input_masks, decoder_input_ids, [labels, domain_classifier_labels] ## We are going to pass the domain indicator batch along with the labels
        else:
            yield input_ids, input_masks, decoder_input_ids, labels
    

def generate_batches_lm(tok, args, rank, files): ## Address compatibilities of the meta tokens when using official models
    """Generates the source, target and source attention masks for denoising. Long sequences are truncated and short sequences are ignored."""
    
    batch_count = 0
    language_list = list(files.keys())
    print("Training for:", language_list)
    language_file_dict = {}
    probs = {}
    for l in language_list:
        file_content = open(files[l]+"."+"%02d" % rank).readlines()
        probs[l] = len(file_content)
        language_file_dict[l] = yield_corpus_indefinitely_mono(file_content, l, args.sorted_batching)
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = probs_temp
    probs_temp = {lang: probs[lang]**(1.0/args.data_sampling_temperature) for lang in probs} ## Temperature sampling probabilities.
    probs = probs_temp
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = [probs_temp[lang] for lang in language_list]
    num_langs = len(language_list)
    language_indices = list(range(num_langs))
    has_rem=False
    while batch_count != args.num_batches:
        curr_batch_count = 0
        input_batch = []
        batch_count += 1
        max_sent_len = 0
        prev_max_sent_len = 0
        start = time.time()
        sents_in_batch = 0
        while True:
            if not has_rem:
                language_idx = random.choices(language_indices, probs)[0]
                sentence = next(language_file_dict[language_list[language_idx]]).strip()
                lang = "<2"+language_list[language_idx]+">"
                sentence_split = sentence.split(" ")
                sent_len = len(sentence_split)
                if sent_len < 10: ## Extremely short docs.
                    continue
            if args.train_with_meta and not has_rem and random.random() <= 0.2: ## Use the first part of the document only 20% of the time.
                randidx = 0
                sentence_split_curr = sentence_split[randidx:randidx+args.max_length]
                sentence_curr=" ".join(sentence_split)

                if args.use_official_pretrained:
                    input_batch.append(sentence_curr)
                else:
                    input_batch.append(lang + " " + sentence_curr)

                sents_in_batch += 1
                if sents_in_batch == args.batch_size: ## We will drop this sentence for now. It may be used in a future iteration.
                    has_rem=True
                    break
                    
            has_rem=False
            randidx = random.randint(0,max(sent_len-args.max_length,0))
            sentence_split = sentence_split[randidx:randidx+args.max_length]
            sentence=" ".join(sentence_split)
            
            if args.use_official_pretrained:
                input_batch.append(sentence)
            else:
                input_batch.append(lang + " " + sentence)

            sents_in_batch += 1
            if sents_in_batch == args.batch_size: ## We will drop this sentence for now. It may be used in a future iteration.
                break
        
        if args.use_official_pretrained:
            input_ids = tok(input_batch, return_tensors="pt", padding=True).input_ids
        else:
            input_ids = tok(input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        if len(input_ids[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
            input_ids = input_ids[:,:args.hard_truncate_length+1]
        labels = input_ids[:,1:]
        input_ids = input_ids[:,:-1]
        if input_ids.size()[1] == 0:
            continue
        end = time.time()
        yield input_ids, labels
    
    
def assert_all_frozen(model):
    """Checks if frozen parameters are all linked to each other or not. Ensures no disjoint components of graphs."""
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

def grad_status(model):
    """Checks whether the parameter needs gradient or not. Part of asserting that the correct parts of the model are frozen."""
    return (par.requires_grad for par in model.parameters())


def freeze_params(model, exception="none"):
    """Set requires_grad=False for each of model.parameters() thereby freezing those parameters. We use this when we want to prevent parts of the model from being trained."""
    if exception is "none":
        for par in model.parameters():
            par.requires_grad = False
    elif exception is not None:
        exception = exception.split(",")
        for name, par in model.named_parameters():
            if not any(individual_exception in name for individual_exception in exception):
                print("Freezing", name)
                par.requires_grad = False

def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for mbart."""
    try:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    except AttributeError:
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)

def generate_batches_eval_bilingual(tok, args, file, slang):
    """Generates the source sentences for the dev set. This ensures that long sentences are truncated and then batched. The batch size is the number of sentences and not the number of tokens."""
    src_file = file
    curr_batch_count = 0
    encoder_input_batch = []
    if args.multi_source: ## Additional source batch and length info
        encoder_input_batch_parent = []
        slang = slang.split("-")
        slang_parent = slang[0]
        slang = slang[1]
        lang_parent = slang_parent if args.use_official_pretrained else "<2"+slang_parent+">"
    lang = slang if args.use_official_pretrained else "<2"+slang+">"
    for src_line in src_file:
        start = time.time()
        src_sent = src_line.strip()
        if args.multi_source: ## We assume that we use a N-way corpus of 3 languages X, Y and Z. We want to distill Y-Z behavior into X-Z where the Y-Z pair also has additional larger corpora but X-Z does not. As such the source sentence should be a tab separated sentence consisting of X[tab]Y.
            src_sent = src_sent.split("\t")
            src_sent_parent = src_sent[0].strip() ## This is the sentence for Y
            src_sent = src_sent[1] ## This is the sentence for X
        src_sent_split = src_sent.split(" ")
        sent_len = len(src_sent_split)
        if sent_len > args.max_src_length: ## Initial truncation
            src_sent_split=src_sent_split[:args.max_src_length]
            src_sent = " ".join(src_sent_split)
            sent_len = args.max_src_length
        
        if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            iids = tok(src_sent, return_tensors="pt").input_ids
            sent_len = len(iids[0])
            if sent_len > args.hard_truncate_length:
                src_sent = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                sent_len = args.hard_truncate_length
            encoder_input_batch.append(src_sent)
        else: ## Truncation code is the same as in the "if case" but replicating it just for clarity. Coincidentally the truncation indices are the same.
            iids = tok(lang + " " + src_sent + " </s>", add_special_tokens=False, return_tensors="pt").input_ids
            sent_len = len(iids[0])
            if sent_len > args.hard_truncate_length:
                src_sent = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                sent_len = args.hard_truncate_length
            if args.use_official_pretrained and "50" in args.pretrained_model: ## mbart-50 model has a different input representation
                encoder_input_batch.append(lang + " " + src_sent + " </s>")
            else:
                encoder_input_batch.append(src_sent + " </s> " + lang)
        if args.multi_source: ## Process the batch for the additional source as well.
            src_sent_split_parent = src_sent_parent.split(" ")
            sent_len_parent = len(src_sent_split_parent)
            if sent_len_parent > args.max_src_length: ## Initial truncation
                src_sent_split_parent=src_sent_split_parent[:args.max_src_length]
                src_sent_parent = " ".join(src_sent_split_parent)
                sent_len_parent = args.max_src_length
            if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                iids = tok(src_sent_parent, return_tensors="pt").input_ids
                sent_len_parent = len(iids[0])
                if sent_len_parent > args.hard_truncate_length:
                    src_sent_parent = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    sent_len_parent = args.hard_truncate_length
                encoder_input_batch_parent.append(src_sent_parent)
            else: ## Truncation code is the same as in the "if case" but replicating it just for clarity. Coincidentally the truncation indices are the same.
                iids = tok(lang_parent + " " + src_sent_parent + " </s>", add_special_tokens=False, return_tensors="pt").input_ids
                sent_len_parent = len(iids[0])
                if sent_len_parent > args.hard_truncate_length:
                    src_sent_parent = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    sent_len_parent = args.hard_truncate_length
                if args.use_official_pretrained and "50" in args.pretrained_model: ## mbart-50 model has a different input representation
                    encoder_input_batch_parent.append(lang_parent + " " + src_sent_parent + " </s>")
                else:
                    encoder_input_batch_parent.append(src_sent_parent + " </s> " + lang_parent)

            

        curr_batch_count += 1
        if curr_batch_count == args.dev_batch_size:
            if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                input_ids = tok(encoder_input_batch, return_tensors="pt", padding=True).input_ids
            else:
                input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            # if len(input_ids[0]) > args.hard_truncate_length:
            #     input_ids = input_ids[:,:args.hard_truncate_length]
            input_masks = (input_ids != tok.pad_token_id).int()
            
            if args.multi_source: ## Process the batch for the additional source as well.
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
                # if len(input_ids_parent[0]) > args.hard_truncate_length:
                #     input_ids_parent = input_ids_parent[:,:args.hard_truncate_length]
                input_masks_parent = (input_ids_parent != tok.pad_token_id).int()
                #print(input_ids.size(), input_ids_parent.size())
                yield [input_ids, input_ids_parent], [input_masks, input_masks_parent]
            else:
                yield input_ids, input_masks
            end = time.time()
            curr_batch_count = 0
            encoder_input_batch = []
            if args.multi_source: ## Additional source batch and length info
                encoder_input_batch_parent = []

    if len(encoder_input_batch) != 0:
        if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            input_ids = tok(encoder_input_batch, return_tensors="pt", padding=True).input_ids
        else:
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        # if len(input_ids[0]) > args.hard_truncate_length:
        #     input_ids = input_ids[:,:args.hard_truncate_length]
        input_masks = (input_ids != tok.pad_token_id).int()
        
        if args.multi_source: ## Process the batch for the additional source as well.
            input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            # if len(input_ids_parent[0]) > args.hard_truncate_length:
            #     input_ids_parent = input_ids_parent[:,:args.hard_truncate_length]
            input_masks_parent = (input_ids_parent != tok.pad_token_id).int()
            yield [input_ids, input_ids_parent], [input_masks, input_masks_parent]
        else:
            yield input_ids, input_masks


def generate_batches_bilingual(tok, args, files, rank):
    """Generates the source, target and source attention masks for the training set. The source and target sentences are ignored if empty and are truncated if longer than a threshold. The batch size in this context is the maximum number of tokens in the batch post padding."""
    if args.tokenization_sampling:
        print("Stochastic tokenizer will be used.")
        if "mbart" in args.tokenizer_name_or_path:
            print("BPE dropout with a dropout probability of", args.tokenization_alpha_or_dropout, "will be used.")
        else:
            print("Sentencepiece regularization with an alpha value of", args.tokenization_alpha_or_dropout, "will be used.")

    batch_count = 0
    if args.use_official_pretrained:
        mask_tok = "<mask>"
    else:
        mask_tok = "[MASK]"

    if len(args.token_masking_probs_range) == 1:
        mp_val_or_range = args.token_masking_probs_range[0]
    elif len(args.token_masking_probs_range) == 2:
        mp_val_or_range = args.token_masking_probs_range
    if not args.is_summarization or args.source_masking_for_bilingual:
        print("Masking ratio:", mp_val_or_range)

    language_list = list(files.keys())
    print("Training for:", language_list)
    language_file_dict = {}
    probs = {}
    for l in language_list:
        src_file_content = open(files[l][0]+"."+"%02d" % rank).readlines()
        tgt_file_content = open(files[l][1]+"."+"%02d" % rank).readlines()
        probs[l] = len(src_file_content)
        file_content = list(zip(src_file_content, tgt_file_content))
        language_file_dict[l] = yield_corpus_indefinitely_bi(file_content, l, args.sorted_batching)
    print("Corpora stats:", probs)
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = probs_temp
    probs_temp = {lang: probs[lang]**(1.0/args.data_sampling_temperature) for lang in probs} ## Temperature sampling probabilities.
    probs = probs_temp
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = [probs_temp[lang] for lang in language_list]
    dropped_source_sentence = "" ## We will save the source sentence to be dropped this batch and add it to the next batch.
    dropped_target_sentence = "" ## We will save the target sentence to be dropped this batch and add it to the next batch.
    dropped_language = "" ## We will save the language indicator to be dropped this batch and add it to the next batch.
    if args.cross_distillation or args.multi_source: ## We assume an additional source language.
        dropped_source_sentence_parent = "" ## We will save the source parent sentence to be dropped this batch and add it to the next batch.
    while batch_count != args.num_batches:
        curr_batch_count = 0
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_label_batch = []
        batch_count += 1
        max_src_sent_len = 0
        max_tgt_sent_len = 0
        start = time.time()
        sents_in_batch = 0
        if args.cross_distillation or args.multi_source: ## We assume an additional source language.
            max_src_sent_len_parent = 0
            encoder_input_batch_parent = []
        if args.num_domains_for_domain_classifier > 1:
            domain_classifier_labels = []
        while True:
            if dropped_source_sentence != "":
                language = dropped_language # Reuse the previous language indicator
                src_sent = dropped_source_sentence # Reuse the previous source sentence
                tgt_sent = dropped_target_sentence # Reuse the previous target sentence
                dropped_language = ""
                dropped_source_sentence = ""
                dropped_target_sentence = ""
                if args.cross_distillation or args.multi_source: ## We assume an additional source language.
                    src_sent_parent = dropped_source_sentence_parent # Reuse the previous source sentence
                    dropped_source_sentence_parent = ""
            else:
                language = random.choices(language_list, probs)[0]
                src_sent, tgt_sent = next(language_file_dict[language])
                if args.cross_distillation or args.multi_source: ## We assume that we use a N-way corpus of 3 languages X, Y and Z. We want to distill Y-Z behavior into X-Z where the Y-Z pair also has additional larger corpora but X-Z does not. As such the source sentence should be a tab separated sentence consisting of X[tab]Y.
                    src_sent = src_sent.split("\t")
                    src_sent_parent = src_sent[0].strip() ## This is the sentence for Y
                    src_sent = src_sent[1] ## This is the sentence for X
                src_sent = src_sent.strip()
                tgt_sent = tgt_sent.strip()
            slangtlang = language.strip().split("-")
            if args.cross_distillation or args.multi_source: ## In this case only we provide a hyphen separated triplet to represent languages X, Y and Z.
                slang_parent = slangtlang[0] if args.use_official_pretrained else "<2"+slangtlang[0]+">"
                slang = slangtlang[1] if args.use_official_pretrained else "<2"+slangtlang[1]+">"
                tlang = slangtlang[2] if args.use_official_pretrained else "<2"+slangtlang[2]+">"
            else:
                slang = slangtlang[0] if args.use_official_pretrained else "<2"+slangtlang[0]+">"
                tlang = slangtlang[1] if args.use_official_pretrained else "<2"+slangtlang[1]+">"
            src_sent_split = src_sent.split(" ")
            tgt_sent_split = tgt_sent.split(" ")
            tgt_sent_len = len(tgt_sent_split)
            src_sent_len = len(src_sent_split)
            
            if src_sent_len < 1 or tgt_sent_len < 1:
                continue
            else:   # Initial truncation
                if src_sent_len >= args.max_src_length:
                    src_sent_split = src_sent_split[:args.max_src_length]
                    src_sent = " ".join(src_sent_split)
                    src_sent_len = args.max_src_length
                if tgt_sent_len >= args.max_tgt_length:
                    tgt_sent_split = tgt_sent_split[:args.max_tgt_length]
                    tgt_sent = " ".join(tgt_sent_split)
                    tgt_sent_len = args.max_tgt_length
            
            if args.cross_distillation or args.multi_source:
                src_sent_split_parent = src_sent_parent.split(" ")
                src_sent_len_parent = len(src_sent_split_parent)
                if src_sent_len_parent < 1:
                    continue
                else:   # Initial truncation
                    if src_sent_len_parent >= args.max_src_length: ## The same sentence length constraint applies to Y as it does to X.
                        src_sent_split_parent = src_sent_split_parent[:args.max_src_length]
                        src_sent_parent = " ".join(src_sent_split_parent)
                        src_sent_len_parent = args.max_src_length
                        
            if (slang == tlang and not args.is_summarization) or args.source_masking_for_bilingual: ## Copying task should DEFINITELY use source masking unless we are doing summarization. We wont bother using this condition for cross distillation. In fact a single condition based on a flag should be sufficient but I am too lazy to make a change. Come fight me if you disagree.
                if args.source_masking_for_bilingual:
                    mask_percent = random.uniform(0.0, mp_val_or_range[0]) ## Do less masking
                else:
                    if type(mp_val_or_range) is float:
                        mask_percent = mp_val_or_range
                    else:
                        mask_percent = random.uniform(mp_val_or_range[0], mp_val_or_range[1])
                mask_count = 0
                max_mask_count = int(mask_percent*src_sent_len)
                spans_to_mask = list(np.random.poisson(args.token_masking_lambda, 1000))
                curr_sent_len = src_sent_len
                while mask_count < max_mask_count:
                    try:
                        span_to_mask = spans_to_mask[0]
                        del spans_to_mask[0]
                        if span_to_mask > (max_mask_count-mask_count): ## Cant mask more than the allowable number of tokens.
                            continue
                        idx_to_mask = random.randint(curr_sent_len//2 if args.future_prediction else 0, (curr_sent_len-1)-(span_to_mask-1))
                        if mask_tok not in src_sent_split[idx_to_mask:idx_to_mask+span_to_mask]:
                            actually_masked_length = len(src_sent_split[idx_to_mask:idx_to_mask+span_to_mask]) ## If at the end of the sentence then we have likely masked fewer tokens.
                            src_sent_split[idx_to_mask:idx_to_mask+span_to_mask] = [mask_tok]
                            mask_count += actually_masked_length # We assume that with a low probability there are mask insertions when span lengths are 0 which may cause more mask tokens than planned. I have decided not to count these insersions towards the maximum maskable limit. This means that the total number of mask tokens will be a bit higher than what it should be. 
                            curr_sent_len -= (actually_masked_length-1)
                    except:
                        break ## If we cannot get a properly masked sentence despite all our efforts then we just give up and continue with what we have so far.
                src_sent = " ".join(src_sent_split)
                if args.span_prediction or args.span_to_sentence_prediction: ## We only predict the masked spans and not other tokens.
                    masked_sentence_split = src_sent.split(mask_tok)
                    final_sentence = ""
                    prev_idx = 0
                    for span in masked_sentence_split:
                        if span.strip() != "":
                            extracted = tgt_sent[prev_idx:prev_idx+tgt_sent[prev_idx:].index(span)]
                            final_sentence += extracted + " <s> "
                            prev_idx=prev_idx+len(extracted) + len(span)

                    final_sentence += tgt_sent[prev_idx:]
                    final_sentence = final_sentence.strip()
                    
                    if (slang == tlang) and args.span_to_sentence_prediction:
                        src_sent = final_sentence
                    else:
                        tgt_sent = final_sentence

            if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                iids = tok(src_sent, return_tensors="pt").input_ids
                curr_src_sent_len = len(iids[0])
                if curr_src_sent_len > args.hard_truncate_length:
                    src_sent = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_src_sent_len = args.hard_truncate_length
                
                iids = tok(tgt_sent, return_tensors="pt").input_ids
                curr_tgt_sent_len = len(iids[0])
                if curr_tgt_sent_len > args.hard_truncate_length:
                    tgt_sent = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_tgt_sent_len = args.hard_truncate_length
            else:
                iids = tok(src_sent + " </s> " + slang, add_special_tokens=False, return_tensors="pt").input_ids
                curr_src_sent_len = len(iids[0])
                if curr_src_sent_len > args.hard_truncate_length:
                    src_sent = tok.decode(iids[0][0:args.hard_truncate_length-2], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_src_sent_len = args.hard_truncate_length
                
                iids = tok(tlang + " " + tgt_sent, add_special_tokens=False, return_tensors="pt").input_ids
                curr_tgt_sent_len = len(iids[0])
                if curr_tgt_sent_len > args.hard_truncate_length:
                    tgt_sent = tok.decode(iids[0][1:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_tgt_sent_len = args.hard_truncate_length
            
            if args.cross_distillation or args.multi_source:
                if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                    iids = tok(src_sent_parent, add_special_tokens=False, return_tensors="pt").input_ids
                    curr_src_sent_len_parent = len(iids[0])
                    if curr_src_sent_len_parent > args.hard_truncate_length:
                        src_sent_parent = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        curr_src_sent_len_parent = args.hard_truncate_length
                else: ## Truncation code is the same as in the "if case" but replicating it just for clarity. Coincidentally the truncation indices are the same.
                    iids = tok(src_sent_parent + " </s> " + slang_parent, add_special_tokens=False, return_tensors="pt").input_ids
                    curr_src_sent_len_parent = len(iids[0])
                    if curr_src_sent_len_parent > args.hard_truncate_length:
                        src_sent_parent = tok.decode(iids[0][0:args.hard_truncate_length-2], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        curr_src_sent_len_parent = args.hard_truncate_length
                
                if curr_src_sent_len_parent > max_src_sent_len_parent:
                    max_src_sent_len_parent = curr_src_sent_len_parent

            if curr_src_sent_len > max_src_sent_len:
                max_src_sent_len = curr_src_sent_len
            
            if curr_tgt_sent_len > max_tgt_sent_len:
                max_tgt_sent_len = curr_tgt_sent_len
            
            if args.batch_size_indicates_lines: ## Batch a fixed number of sentences. We can safely add the current example because we assume that the user knows the max batch size.
                if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                    encoder_input_batch.append(src_sent)
                    decoder_input_batch.append(tgt_sent)

                else:
                    if args.use_official_pretrained and "50" in args.pretrained_model: ## mbart-50 model has a different input representation
                        encoder_input_batch.append(slang + " " + src_sent + " </s>")
                    else:
                        encoder_input_batch.append(src_sent + " </s> " + slang)
                    if args.unify_encoder:
                        decoder_input_batch.append(tgt_sent + " </s> " + tlang)
                        decoder_label_batch.append(tgt_sent + " </s> " + tlang) ## This should not be used when we unify encoders.
                    else:
                        decoder_input_batch.append(tlang + " " + tgt_sent)
                        decoder_label_batch.append(tgt_sent + " </s>")
                        if args.tokenization_sampling:
                            decoder_input_batch[-1] += " </s>" ## In case of stochastic subword segmentation we have to generate the decoder label ids from the decoder input ids. This will make the model behavior sliiiiiightly different from how it was when stochastic decoding is not done. However it should have no major difference. In the non stochastic case, the </s> or EOS token is never fed as the input but in the stochastic case that token is fed as the input. So this means in the non stochastic case, the end of decoder and labels will be something like: "a good person <pad> <pad> <pad>" and "good person </s> <pad <pad> <pad>". In the stochastic case, the end of decoder and labels will be something like: "a good person </s> <pad> <pad>" and "good person </s> <pad> <pad <pad>". Now think about how we compute loss. When the label is a pad, the loss is never propagated through it. So the model will never learn anything when </s> or <pad> will be the input. Furthermore, the generation of </s> is what dictates the end of generation. When the model generates a </s> the sequence is taken out of the computation process and therefore whatever it generates after that will not be considered at all towards its final score. In conclusion, there should be no practical difference in outcomes between the two batching approaches.
                    if args.cross_distillation or args.multi_source:
                        encoder_input_batch_parent.append(src_sent_parent + " </s> " + slang_parent)

                sents_in_batch += 1
                if args.num_domains_for_domain_classifier > 1:
                    domain_classifier_labels.append(files[language][2])
                if sents_in_batch == args.batch_size:
                    break
            else:
                if args.cross_distillation or args.multi_source:
                    potential_batch_count = max(max_src_sent_len, max_src_sent_len_parent, max_tgt_sent_len)*(sents_in_batch+1) ## We limit ourselves based on the maximum of either source or target.
                else:
                    potential_batch_count = max(max_src_sent_len, max_tgt_sent_len)*(sents_in_batch+1) ## We limit ourselves based on the maximum of either source or target.
                if potential_batch_count > args.batch_size: ## We will drop this sentence for now. It may be used in a future iteration. Note that this will be unreliable when we do stochastic subword segmentation.
                    if curr_src_sent_len > args.batch_size or curr_tgt_sent_len > args.batch_size: ## Dangerous sentences. Drop them no matter what.
                        dropped_source_sentence = ""
                        dropped_target_sentence = ""
                        dropped_source_sentence_parent = ""
                        dropped_language = ""
                    else:
                        dropped_source_sentence = src_sent
                        dropped_target_sentence = tgt_sent
                        dropped_language = language 
                    if args.cross_distillation or args.multi_source:
                        if curr_src_sent_len_parent > args.batch_size: ## Dangerous sentences. Drop them no matter what.
                            dropped_source_sentence = ""
                            dropped_target_sentence = ""
                            dropped_source_sentence_parent = ""
                            dropped_language = ""
                        else:
                            dropped_source_sentence_parent = src_sent_parent
                    break
                if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                    encoder_input_batch.append(src_sent)
                    decoder_input_batch.append(tgt_sent)

                else:
                    if args.use_official_pretrained and "50" in args.pretrained_model: ## mbart-50 model has a different input representation
                        encoder_input_batch.append(slang + " " + src_sent + " </s>")
                    else:
                        encoder_input_batch.append(src_sent + " </s> " + slang)
                    if args.unify_encoder:
                        decoder_input_batch.append(tgt_sent + " </s> " + tlang)
                        decoder_label_batch.append(tgt_sent + " </s> " + tlang) ## This should not be used when we unify encoders.
                    else:
                        decoder_input_batch.append(tlang + " " + tgt_sent)
                        decoder_label_batch.append(tgt_sent + " </s>")
                        if args.tokenization_sampling:
                            decoder_input_batch[-1] += " </s>" ## In case of stochastic subword segmentation we have to generate the decoder label ids from the decoder input ids. This will make the model behavior sliiiiiightly different from how it was when stochastic decoding is not done. However it should have no major difference. In the non stochastic case, the </s> or EOS token is never fed as the input but in the stochastic case that token is fed as the input. So this means in the non stochastic case, the end of decoder and labels will be something like: "a good person <pad> <pad> <pad>" and "good person </s> <pad <pad> <pad>". In the stochastic case, the end of decoder and labels will be something like: "a good person </s> <pad> <pad>" and "good person </s> <pad> <pad <pad>". Now think about how we compute loss. When the label is a pad, the loss is never propagated through it. So the model will never learn anything when </s> or <pad> will be the input. Furthermore, the generation of </s> is what dictates the end of generation. When the model generates a </s> the sequence is taken out of the computation process and therefore whatever it generates after that will not be considered at all towards its final score. In conclusion, there should be no practical difference in outcomes between the two batching approaches.
                    if args.cross_distillation or args.multi_source:
                        encoder_input_batch_parent.append(src_sent_parent + " </s> " + slang_parent)

                sents_in_batch += 1
                if args.num_domains_for_domain_classifier > 1:
                    domain_classifier_labels.append(files[language][2])
                
        if len(encoder_input_batch) == 0:
            print("Zero size batch due to an abnormal example. Skipping empty batch.")
            continue    

        if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            input_ids = tok(encoder_input_batch, return_tensors="pt", padding=True).input_ids
        else:
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
        # if len(input_ids[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
        #     input_ids = input_ids[:,:args.hard_truncate_length]
        input_masks = (input_ids != tok.pad_token_id).int()
        if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            decoder_input_ids = tok(decoder_input_batch, return_tensors="pt", padding=True).input_ids
        else:
            decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
        # if len(decoder_input_ids[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
        #     decoder_input_ids = decoder_input_ids[:,:args.hard_truncate_length]
        if (args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model) or args.tokenization_sampling: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            labels = decoder_input_ids[:,1:]
            decoder_input_ids = decoder_input_ids[:,:-1]
        else:
            labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            # if len(labels[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
            #     labels = labels[:,:args.hard_truncate_length]
        if args.cross_distillation or args.multi_source:
            if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            else:
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
            # if len(input_ids_parent[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
            #     input_ids_parent = input_ids_parent[:,:args.hard_truncate_length]
            input_masks_parent = (input_ids_parent != tok.pad_token_id).int()
            end = time.time()
            #print(input_ids.size(), input_ids_parent.size(), decoder_input_ids.size())
            yield [input_ids, input_ids_parent], [input_masks, input_masks_parent], decoder_input_ids, labels
        else:
            end = time.time()
            #print(input_ids.size(), input_masks.size(), decoder_input_ids.size(), labels.size())
            if args.num_domains_for_domain_classifier > 1:
                yield input_ids, input_masks, decoder_input_ids, [labels, domain_classifier_labels]
            else:
                yield input_ids, input_masks, decoder_input_ids, labels

            
def generate_batches_pair(tok, args): ## TODO: Fix for mbart and bart variants
    """Generates the source, target and source attention masks for the training set."""
    src_file = open(args.test_src)
    tgt_file = open(args.test_ref)
    corpus = [(src_line, tgt_line) for src_line, tgt_line in zip(src_file, tgt_file)]
    epoch_counter = 0
    curr_batch_count = 0
    encoder_input_batch = []
    decoder_input_batch = []
    decoder_label_batch = []
    for src_sent, tgt_sent in corpus:
        src_sent = src_sent.strip()
        tgt_sent = tgt_sent.strip()
        start = time.time()
        slang = args.slang if args.use_official_pretrained else "<2"+args.slang+">"
        tlang = args.tlang if args.use_official_pretrained else "<2"+args.tlang+">"
        src_sent_split = src_sent.split(" ")
        tgt_sent_split = tgt_sent.split(" ")
        tgt_sent_len = len(tgt_sent_split)
        src_sent_len = len(src_sent_split)
        if src_sent_len < 1 or tgt_sent_len < 1:
            print("Big problem")
            #continue
        else:   # Initial truncation
            if src_sent_len >= args.max_src_length:
                src_sent_split = src_sent_split[:args.max_src_length]
                src_sent = " ".join(src_sent_split)
                src_sent_len = args.max_src_length
            if tgt_sent_len >= args.max_tgt_length:
                tgt_sent_split = tgt_sent_split[:args.max_tgt_length]
                tgt_sent = " ".join(tgt_sent_split)
                tgt_sent_len = args.max_tgt_length

        encoder_input_batch.append(src_sent + " </s> " + slang)
        decoder_input_batch.append(tlang + " " + tgt_sent)
        decoder_label_batch.append(tgt_sent + " </s>")
        if args.tokenization_sampling:
            decoder_input_batch[-1] += " </s>" ## In case of stochastic subword segmentation we have to generate the decoder label ids from the decoder input ids. This will make the model behavior sliiiiiightly different from how it was when stochastic decoding is not done. However it should have no major difference. In the non stochastic case, the </s> or EOS token is never fed as the input but in the stochastic case that token is fed as the input. So this means in the non stochastic case, the end of decoder and labels will be something like: "a good person <pad> <pad> <pad>" and "good person </s> <pad <pad> <pad>". In the stochastic case, the end of decoder and labels will be something like: "a good person </s> <pad> <pad>" and "good person </s> <pad> <pad <pad>". Now think about how we compute loss. When the label is a pad, the loss is never propagated through it. So the model will never learn anything when </s> or <pad> will be the input. Furthermore, the generation of </s> is what dictates the end of generation. When the model generates a </s> the sequence is taken out of the computation process and therefore whatever it generates after that will not be considered at all towards its final score. In conclusion, there should be no practical difference in outcomes between the two batching approaches.
        curr_batch_count += 1
        if curr_batch_count == args.batch_size:
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            if len(input_ids[0]) > args.hard_truncate_length:
                input_ids = input_ids[:,:args.hard_truncate_length]
            input_masks = (input_ids != tok.pad_token_id).int()
            decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            if len(decoder_input_ids[0]) > args.hard_truncate_length:
                decoder_input_ids = decoder_input_ids[:,:args.hard_truncate_length]
            if args.tokenization_sampling: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                labels = decoder_input_ids[:,1:]
                decoder_input_ids = decoder_input_ids[:,:-1]
            else:
                labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
                if len(labels[0]) > args.hard_truncate_length:
                    labels = labels[:,:args.hard_truncate_length]
            decoder_masks = (decoder_input_ids != tok.pad_token_id).int()
            end = time.time()
            yield input_ids, input_masks, decoder_input_ids, decoder_masks, labels
            curr_batch_count = 0
            encoder_input_batch = []
            decoder_input_batch = []
            decoder_label_batch = []

    if len(encoder_input_batch) != 0:
        input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        if len(input_ids[0]) > args.hard_truncate_length:
            input_ids = input_ids[:,:args.hard_truncate_length]
        input_masks = (input_ids != tok.pad_token_id).int()
        decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        if args.tokenization_sampling: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            labels = decoder_input_ids[:,1:]
            decoder_input_ids = decoder_input_ids[:,:-1]
        else:
            labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            if len(labels[0]) > args.hard_truncate_length:
                labels = labels[:,:args.hard_truncate_length]
        decoder_masks = (decoder_input_ids != tok.pad_token_id).int()
        yield input_ids, input_masks, decoder_input_ids, decoder_masks, labels

def generate_batches_pair_masked(tok, args): ## TODO: Implement hard truncation logic here if something bugs out. In general dont use this its highly untested. ## TODO: Fix for mbart and bart variants
    """Generates the source, target and source attention masks for the training set."""
    if args.use_official_pretrained:
        mask_tok = "<mask>"
    else:
        mask_tok = "[MASK]"
    src_file = open(args.test_src)
    tgt_file = open(args.test_ref)
    corpus = [(src_line, tgt_line) for src_line, tgt_line in zip(src_file, tgt_file)]
    epoch_counter = 0
    curr_batch_count = 0
    for src_sent, tgt_sent in corpus:
        src_sent = src_sent.strip()
        tgt_sent = tgt_sent.strip()
        start = time.time()
        slang = args.slang if args.use_official_pretrained else "<2"+args.slang+">"
        tlang = args.tlang if args.use_official_pretrained else "<2"+args.tlang+">"
        src_sent_split = src_sent.split(" ")
        tgt_sent_split = tgt_sent.split(" ")
        tgt_sent_len = len(tgt_sent_split)
        src_sent_len = len(src_sent_split)
        if src_sent_len < 1 or src_sent_len >= 100 or tgt_sent_len < 1 or tgt_sent_len >= 100:
            continue
        
        for pos_src in range(src_sent_len):
            encoder_input_batch = []
            decoder_input_batch = []
            decoder_label_batch = []
            dec_pos = []
            enc_pos = pos_src
            new_src_sent_split = list(src_sent_split)
            new_src_sent_split[pos_src] = mask_tok
            new_src_sent = " ".join(new_src_sent_split)
            for pos_tgt in range(tgt_sent_len):
                dec_pos.append(pos_tgt)
                new_tgt_sent_split = list(tgt_sent_split)
                new_tgt_sent_split[pos_tgt] = mask_tok
                new_tgt_sent = " ".join(new_tgt_sent_split)
                encoder_input_batch.append(new_src_sent + " </s> " + slang)
                decoder_input_batch.append(tlang + " " + new_tgt_sent)
                decoder_label_batch.append(new_tgt_sent + " </s>")
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            if len(input_ids[0]) > args.hard_truncate_length:
                input_ids = input_ids[:,:args.hard_truncate_length]
            input_masks = (input_ids != tok.pad_token_id).int()
            decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            if len(decoder_input_ids[0]) > args.hard_truncate_length:
                decoder_input_ids = decoder_input_ids[:,:args.hard_truncate_length]
            tgt_masks = (decoder_input_ids != tok.pad_token_id).int()
            labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            if len(labels[0]) > args.hard_truncate_length:
                labels = labels[:,:args.hard_truncate_length]
            end = time.time()

            yield input_ids, input_masks, decoder_input_ids, tgt_masks, labels, src_sent_split, tgt_sent_split, enc_pos, dec_pos
            


def generate_batches_for_decoding(tok, args):
    """Generates the source sentences for the test set."""
    if args.tokenization_sampling:
        print("Stochastic tokenizer will be used.")
        if "mbart" in args.tokenizer_name_or_path:
            print("BPE dropout with a dropout probability of", args.tokenization_alpha_or_dropout, "will be used.")
        else:
            print("Sentencepiece regularization with an alpha value of", args.tokenization_alpha_or_dropout, "will be used.")

    if args.use_official_pretrained:
        mask_tok = "<mask>"
    else:
        mask_tok = "[MASK]"
    src_file = open(args.test_src)
    slang = args.slang
    curr_batch_count = 0
    encoder_input_batch = []
    if args.multi_source: ## Additional source batch and length info
        encoder_input_batch_parent = []
        slang = slang.split("-")
        slang_parent = slang[0]
        slang = slang[1]
        lang_parent = slang_parent if args.use_official_pretrained else "<2"+slang_parent+">"
    lang = slang if args.use_official_pretrained else "<2"+slang+">"
    
    if len(args.token_masking_probs_range) == 1:
        mp_val_or_range = args.token_masking_probs_range[0]
    elif len(args.token_masking_probs_range) == 2:
        mp_val_or_range = args.token_masking_probs_range
    print("Masking ratio:", mp_val_or_range)

    for src_line in src_file:
        start = time.time()
        src_sent = src_line.strip()
        if args.multi_source: ## We assume that we use a N-way corpus of 3 languages X, Y and Z. We want to distill Y-Z behavior into X-Z where the Y-Z pair also has additional larger corpora but X-Z does not. As such the source sentence should be a tab separated sentence consisting of X[tab]Y.
            src_sent = src_sent.split("\t")
            src_sent_parent = src_sent[0].strip() ## This is the sentence for Y
            src_sent = src_sent[1] ## This is the sentence for X
        src_sent_split = src_sent.split(" ")
        sent_len = len(src_sent_split)
        if sent_len > args.max_src_length: ## Initial truncation
            src_sent_split = src_sent_split[:args.max_src_length]
            src_sent = " ".join(src_sent_split)
            sent_len = args.max_src_length
        
        if args.mask_input:
            if type(mp_val_or_range) is float:
                mask_percent = mp_val_or_range
            else:
                mask_percent = random.uniform(mp_val_or_range[0], mp_val_or_range[1])
            mask_count = 0
            max_mask_count = int(mask_percent*sent_len)
            spans_to_mask = list(np.random.poisson(args.token_masking_lambda, 1000))
            curr_sent_len = sent_len
            while mask_count < max_mask_count:
                try:
                    span_to_mask = spans_to_mask[0]
                    del spans_to_mask[0]
                    if span_to_mask > (max_mask_count-mask_count): ## Cant mask more than the allowable number of tokens.
                        continue
                    idx_to_mask = random.randint(sent_len//2 if args.future_prediction else 0, (curr_sent_len-1)-(span_to_mask-1))
                    if mask_tok not in src_sent_split[idx_to_mask:idx_to_mask+span_to_mask]:
                        actually_masked_length = len(src_sent_split[idx_to_mask:idx_to_mask+span_to_mask]) ## If at the end of the sentence then we have likely masked fewer tokens.
                        src_sent_split[idx_to_mask:idx_to_mask+span_to_mask] = [mask_tok]
                        mask_count += actually_masked_length # We assume that with a low probability there are mask insertions when span lengths are 0 which may cause more mask tokens than planned. I have decided not to count these insersions towards the maximum maskable limit. This means that the total number of mask tokens will be a bit higher than what it should be. 
                        curr_sent_len -= (actually_masked_length-1)
                except:
                    break ## If we cannot get a properly masked sentence despite all our efforts then we just give up and continue with what we have so far.
            src_sent = " ".join(src_sent_split)
        
        if args.use_official_pretrained and ("bart" in args.model_path or "barthez" in args.model_path) and "mbart" not in args.model_path: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            iids = tok(src_sent, return_tensors="pt").input_ids
            sent_len = len(iids[0])
            if sent_len > args.hard_truncate_length:
                src_sent = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                sent_len = args.hard_truncate_length
            encoder_input_batch.append(src_sent)
        else: ## Truncation code is the same as in the "if case" but replicating it just for clarity. Coincidentally the truncation indices are the same.
            iids = tok(src_sent + " </s> " + lang, add_special_tokens=False, return_tensors="pt").input_ids
            sent_len = len(iids[0])
            if sent_len > args.hard_truncate_length:
                src_sent = tok.decode(iids[0][0:args.hard_truncate_length-2], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                sent_len = args.hard_truncate_length
            if args.use_official_pretrained and "50" in args.model_path: ## mbart-50 model has a different input representation
                encoder_input_batch.append(lang + " " + src_sent + " </s>")
            else:
                encoder_input_batch.append(src_sent + " </s> " + lang)

        if args.multi_source: ## Process the batch for the additional source as well.
            src_sent_split_parent = src_sent_parent.split(" ")
            sent_len_parent = len(src_sent_split_parent)
            if sent_len_parent > args.max_src_length: ## Initial truncation
                src_sent_split_parent=src_sent_split_parent[:args.max_src_length]
                src_sent_parent = " ".join(src_sent_split_parent)
                sent_len_parent = args.max_src_length
            if args.use_official_pretrained and ("bart" in args.model_path or "barthez" in args.model_path) and "mbart" not in args.model_path: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                iids = tok(src_sent_parent, return_tensors="pt").input_ids
                sent_len_parent = len(iids[0])
                if sent_len_parent > args.hard_truncate_length:
                    src_sent_parent = tok.decode(iids[0][1:args.hard_truncate_length-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    sent_len_parent = args.hard_truncate_length
                encoder_input_batch.append(src_sent_parent)
            else: ## Truncation code is the same as in the "if case" but replicating it just for clarity. Coincidentally the truncation indices are the same.
                iids = tok(src_sent_parent + " </s> " + lang_parent, add_special_tokens=False, return_tensors="pt").input_ids
                sent_len_parent = len(iids[0])
                if sent_len_parent > args.hard_truncate_length:
                    src_sent_parent = tok.decode(iids[0][0:args.hard_truncate_length-2], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    sent_len_parent = args.hard_truncate_length
                if args.use_official_pretrained and "50" in args.model_path: ## mbart-50 model has a different input representation
                    encoder_input_batch_parent.append(lang_parent + " " + src_sent_parent + " </s>")
                else:
                    encoder_input_batch_parent.append(src_sent_parent + " </s> " + lang_parent)
            
            
        curr_batch_count += 1
        if curr_batch_count == args.batch_size:
            if args.use_official_pretrained and ("bart" in args.model_path or "barthez" in args.model_path) and "mbart" not in args.model_path: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                input_ids = tok(encoder_input_batch, return_tensors="pt", padding=True).input_ids
            else:
                input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            # if len(input_ids[0]) > args.hard_truncate_length:
            #     input_ids = input_ids[:,:args.hard_truncate_length]
            input_masks = input_ids != tok.pad_token_id
            end = time.time()
            if args.multi_source: ## Process the batch for the additional source as well.
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
                # if len(input_ids_parent[0]) > args.hard_truncate_length:
                #     input_ids_parent = input_ids_parent[:,:args.hard_truncate_length]
                input_masks_parent = (input_ids_parent != tok.pad_token_id).int()
                yield [input_ids, input_ids_parent], [input_masks, input_masks_parent]
            else:
                yield input_ids, input_masks
            curr_batch_count = 0
            encoder_input_batch = []
            if args.multi_source: ## Additional source batch and length info
                encoder_input_batch_parent = []

    if len(encoder_input_batch) != 0:
        if args.use_official_pretrained and ("bart" in args.model_path or "barthez" in args.model_path) and "mbart" not in args.model_path: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            input_ids = tok(encoder_input_batch, return_tensors="pt", padding=True).input_ids
        else:
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
        # if len(input_ids[0]) > args.hard_truncate_length:
        #     input_ids = input_ids[:,:args.hard_truncate_length]
        input_masks = input_ids != tok.pad_token_id
        if args.multi_source: ## Process the batch for the additional source as well.
            if args.use_official_pretrained and ("bart" in args.model_path or "barthez" in args.model_path) and "mbart" not in args.model_path: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            else:
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
            # if len(input_ids_parent[0]) > args.hard_truncate_length:
            #     input_ids_parent = input_ids_parent[:,:args.hard_truncate_length]
            input_masks_parent = (input_ids_parent != tok.pad_token_id).int()
            yield [input_ids, input_ids_parent], [input_masks, input_masks_parent]
        else:
            yield input_ids, input_masks

def generate_batches_for_decoding_lm(tok, args):
    """Generates the source sentences for the test set."""
    src_file = open(args.test_src)
    lang = args.lang
    lang = lang if args.use_official_pretrained else "<2"+lang+">"
    
    for line in src_file:
        start = time.time()
        sent = line.strip()
        
        if args.use_official_pretrained:
            input_ids = tok([sent], return_tensors="pt", padding=True).input_ids
        else:
            input_ids = tok([lang + " " + sent], add_special_tokens=False, return_tensors="pt", padding=True).input_ids

        end = time.time()
        
        yield input_ids
        

def plot_attention(data, X_label=None, Y_label=None, num_layers=None, num_heads=None, file_name=None, plot_title=None):
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
    fig, ax = plt.subplots(figsize=(10*num_layers, 10*num_heads))  # set figure size
    im = ax.imshow(data, cmap=plt.cm.Blues)


    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(X_label)))
    ax.set_yticks(np.arange(len(Y_label)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(X_label)
    ax.set_yticklabels(Y_label)
    ax.xaxis.tick_top()
    ax.margins(x=0)
    ax.margins(y=0)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")
    
#     for i in range(len(Y_label)):
#         for j in range(len(X_label)):
#             text = ax.text(j, i, "%.1f" % data[i, j],
#                            ha="center", va="center", color="b",size=10.0)
    # Save Figure
    ax.set_title(plot_title)
    fig.tight_layout()

    print("Saving figures %s" % file_name)
    fig.savefig(file_name, bbox_inches='tight')  # save the figure to file
    plt.close(fig)  # close the figure


def generate_batches_monolingual_masked_or_bilingual(tok, args, rank, files, train_files):
    """This will return masked monolingual or bilingual batches according to a fixed ratio."""
    bilingual_generator = generate_batches_bilingual(tok, args, train_files, rank)
    monolingual_generator = generate_batches_monolingual_masked(tok, args, files, rank)
    while True:
        if args.bilingual_train_frequency != 0.0 and random.random() <= args.bilingual_train_frequency:
            yield next(bilingual_generator), True
        else:
            yield next(monolingual_generator), False