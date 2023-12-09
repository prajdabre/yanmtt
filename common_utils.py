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
from torch.optim import Optimizer
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
from typing import Iterable, Tuple
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


class AdamWScale(Optimizer): ### Taken from nanot5 library (https://github.com/PiotrNawrot/nanoT5)
    """
    This AdamW implementation is copied from Huggingface.
    We modified it with Adagrad scaling by rms of a weight tensor

    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # /Adapt Step from Adafactor
                step_size = step_size * max(1e-3, self._rms(p.data))
                # /Adapt Step from Adafactor

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


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
                if p.requires_grad:
                    precision_matrices[n].data += p.grad.data ** 2
        
        for n, p in self.model.named_parameters():
            if p.requires_grad:
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
            if p.requires_grad:
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

def remap_layers(model, idx, args, rank): ### Cut this code into half.
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
                    if rank == 0:
                        print("Remapping", key)
                    key_copy[idx] =tlayer
                    key = ".".join(key)
                    key_copy = ".".join(key_copy)
                    model[key] = model_copy[key_copy]
        for key in keys_to_consider: ## Purge all unspecified keys.
            key = key.strip().split(".")
            if key[idx] not in keys_to_keep:
                key = ".".join(key)
                if rank == 0:
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
                    if rank == 0:
                        print("Remapping", key)
                    key_copy[idx] =tlayer
                    key = ".".join(key)
                    key_copy = ".".join(key_copy)
                    model[key] = model_copy[key_copy]
        for key in keys_to_consider: ## Purge all unspecified keys.
            key = key.strip().split(".")
            if key[idx] not in keys_to_keep:
                key = ".".join(key)
                if rank == 0:
                    print("Deleting", key)
                del model[key]
    if rank == 0:
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
            
def shard_files_mono(files, tokenizer, args):
    """This method shards files into N parts containing the same number of lines. Each shard will go to a different GPU which may even be located on another machine. This method is run when the 'shard_files' argument is passed."""
    print("Sharding files into", args.world_size, "parts")
    for lang, file_details in files:
        infile = open(file_details[0]).readlines() if args.num_domains_for_domain_classifier > 1 else open(file_details).readlines()
        num_lines = len(infile)
        lines_per_shard = math.ceil(num_lines/args.world_size)
        print("For language:",lang," the total number of lines are:", num_lines, "and number of lines per shard are:", lines_per_shard)
        for shard_id in range(args.world_size):
            outfile = open(file_details[0]+"."+"%02d" % shard_id, "w") if args.num_domains_for_domain_classifier > 1 else open(file_details+"."+"%02d" % shard_id, "w")
            for line in infile[shard_id*lines_per_shard:(shard_id+1)*lines_per_shard]:
                if args.sliding_window_shard: ## This is for sliding window sharding. Now note that each shard wont contain the same number of lines. This is because we are sliding the window. But thats not going to change our overall story.
                    if args.sliding_sharding_delimiter == " ": ## Handle this case specially.
                        line = line.strip()
                        line_tok = tokenizer(line, add_special_tokens=False).input_ids
                        num_blocks = math.ceil(len(line_tok)/args.hard_truncate_length)
                        for block_id in range(num_blocks): # Extract the block, detokenize it and write it to the file.
                            block = line_tok[block_id*args.hard_truncate_length:(block_id+1)*args.hard_truncate_length]
                            block = tokenizer.decode(block, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            outfile.write(block+"\n")
                    else: # First split the line based on the args.sliding_sharding_delimiter and then tokenize each constituent. Keep adding tokens until you reach the hard truncate length. Then write the block to the file.
                        line = line.strip()
                        line_split = line.split(args.sliding_sharding_delimiter)
                        block = []
                        for sub_line in line_split:
                            sub_line_tok = tokenizer(sub_line, add_special_tokens=False).input_ids
                            if (len(block) + len(sub_line_tok)) > args.hard_truncate_length:
                                if len(block) > 0:
                                    block = tokenizer.decode(block, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                    outfile.write(block+"\n")
                                    block = []
                                else:
                                    pass
                            block.extend(sub_line_tok)
                        if len(block) > 0:
                            block = tokenizer.decode(block, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            outfile.write(block+"\n")
                else:
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
        
def shard_files_bi(files, tokenizer, args, additional_tokenizer=None):
    """This method shards files into N parts containing the same number of lines. Each shard will go to a different GPU which may even be located on another machine. This method is run when the 'shard_files' argument is passed."""
    print("Sharding files into", args.world_size, "parts")
    if args.sliding_window_shard and args.sliding_sharding_delimiter == " ":
        print("Sliding window sharding with a space delimiter with parallel corpora is not a good idea. You have chosen violence. Now you must live with it.")
    elif args.sliding_window_shard and args.sliding_sharding_delimiter != " ":
        print("Sliding window sharding with a non-space delimiter with parallel corpora will make sense only if the number of delimiter separated sentences per pair of lines read are equal. Be careful.")

    if additional_tokenizer is None:
        additional_tokenizer = tokenizer
    for pair, file_details in files:
        infile = list(zip(open(file_details[0]).readlines(), open(file_details[1]).readlines()))
        num_lines = len(infile)
        lines_per_shard = math.ceil(num_lines/args.world_size)
        print("For language pair:",pair," the total number of lines are:", num_lines, "and number of lines per shard are:", lines_per_shard)
        for shard_id in range(args.world_size):
            srcoutfile = open(file_details[0]+"."+"%02d" % shard_id, "w")
            tgtoutfile = open(file_details[1]+"."+"%02d" % shard_id, "w")
            for src_line, tgt_line in infile[shard_id*lines_per_shard:(shard_id+1)*lines_per_shard]:
                if args.sliding_window_shard: ## This is for sliding window sharding. Now note that each shard wont contain the same number of lines. This is because we are sliding the window. But thats not going to change our overall story.
                    if args.sliding_sharding_delimiter == " ": ## Handle this case specially. For a parallel corpus setting, this will be bad. Very very bad. Dont use this EVER.
                        src_line = src_line.strip()
                        tgt_line = tgt_line.strip()
                        src_line_tok = tokenizer(src_line, add_special_tokens=False).input_ids
                        tgt_line_tok = additional_tokenizer(tgt_line, add_special_tokens=False).input_ids
                        num_blocks = math.ceil(len(src_line_tok)/args.hard_truncate_length)
                        for block_id in range(num_blocks): # Extract the block, detokenize it and write it to the file. Assuming that the number of delimiter separated sentences per line is the same for both the files.
                            src_block = src_line_tok[block_id*args.hard_truncate_length:(block_id+1)*args.hard_truncate_length]
                            tgt_block = tgt_line_tok[block_id*args.hard_truncate_length:(block_id+1)*args.hard_truncate_length]
                            src_block = tokenizer.decode(src_block, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            tgt_block = additional_tokenizer.decode(tgt_block, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            srcoutfile.write(src_block+"\n")
                            tgtoutfile.write(tgt_block+"\n")
                    else: # First split the line based on the args.sliding_sharding_delimiter and then tokenize each constituent. Keep adding tokens until you reach the hard truncate length. Then write the block to the file. Use this only if you are sure that the number of delimiter separated sentences per line is the same for both the files.
                        src_line = src_line.strip()
                        tgt_line = tgt_line.strip()
                        src_line_split = src_line.split(args.sliding_sharding_delimiter)
                        tgt_line_split = tgt_line.split(args.sliding_sharding_delimiter)
                        src_block = []
                        tgt_block = []
                        for src_sub_line, tgt_sub_line in zip(src_line_split, tgt_line_split):
                            src_sub_line_tok = tokenizer(src_sub_line, add_special_tokens=False).input_ids
                            tgt_sub_line_tok = additional_tokenizer(tgt_sub_line, add_special_tokens=False).input_ids
                            if (len(src_block) + len(src_sub_line_tok)) > args.hard_truncate_length: ## Assuming that source and target have the same number of delimiter separated sentences per line.
                                if len(src_block) > 0:
                                    src_block = tokenizer.decode(src_block, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                    srcoutfile.write(src_block+"\n")
                                    src_block = []
                                    tgt_block = additional_tokenizer.decode(tgt_block, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                    tgtoutfile.write(tgt_block+"\n")
                                    tgt_block = []
                                else:
                                    pass
                            src_block.extend(src_sub_line_tok)
                            tgt_block.extend(tgt_sub_line_tok)
                        if len(src_block) > 0:
                            src_block = tokenizer.decode(src_block, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            srcoutfile.write(src_block+"\n")
                            tgt_block = additional_tokenizer.decode(tgt_block, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            tgtoutfile.write(tgt_block+"\n")
                else:
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

def get_bucket_indexed_indefinite_corpus_yielder_mono(corpus, lang, bucket_intervals, sorted_batching, tokenizer):
    # First we split the corpus into buckets. Each number in the bucket_intervals list is the upper bound of the bucket. The last bucket is the rest of the corpus.
    # We create a dictionary where the key is the bucket id and the value is the yielder for that bucket.
    corpus_split_by_length = {l: [] for l in bucket_intervals}
    for line in corpus:
        tokenized_sentence = tokenizer(line.strip(), add_special_tokens=False).input_ids
        line_length = len(tokenized_sentence)
        for bucket_id in range(len(bucket_intervals)-1):
            if line_length >= bucket_intervals[bucket_id] and line_length < bucket_intervals[bucket_id+1]:
                corpus_split_by_length[bucket_intervals[bucket_id]].append(line)
                break
        if line_length >= bucket_intervals[-1]:
            corpus_split_by_length[bucket_intervals[-1]].append(line)
    # Print bucket sizes
    bucket_distributions = []
    for bucket_id in corpus_split_by_length:
        curr_bucket_size = len(corpus_split_by_length[bucket_id])
        print("For language", lang, "the number of lines in bucket", bucket_id, "is", curr_bucket_size)
        bucket_distributions.append(curr_bucket_size)
    total_lines = sum(bucket_distributions)
    bucket_distributions = [x/total_lines for x in bucket_distributions]
    # Now we create a yielder for each bucket.
    bucket_yielders = {}
    for bucket_id in corpus_split_by_length:
        if len(corpus_split_by_length[bucket_id]) > 0:
            bucket_yielders[bucket_id] = yield_corpus_indefinitely_mono(corpus_split_by_length[bucket_id], lang, sorted_batching, bucketed_batching=True, bucket_id=bucket_id, tokenizer=tokenizer)
    
    return bucket_yielders, bucket_distributions



def yield_corpus_indefinitely_mono(corpus, lang, sorted_batching, bucketed_batching=False, bucket_id=None, tokenizer=None):
    """This shuffles the corpus or corpus shard at the beginning of each epoch and returns sentences indefinitely."""
    epoch_counter = 0
    num_lines = len(corpus)
    num_sentences_before_sort = 20000
    num_sorted_segments = (num_lines // num_sentences_before_sort) + 1
    try:
        while True:
            if bucketed_batching:
                print("Shuffling bucket!")
            else:
                print("Shuffling corpus!")
            
            sys.stdout.flush()
            random.shuffle(corpus) ## Add bucketing logic here
            if sorted_batching:
                for curr_segment_id in range(num_sorted_segments):
                    curr_segment = corpus[curr_segment_id*num_sentences_before_sort:(curr_segment_id+1)*num_sentences_before_sort]
                    for src_line in sorted(curr_segment, key=lambda x: len(tokenizer(x.strip(), add_special_tokens=False).input_ids)):
                        yield src_line
            else:
                for src_line in corpus:
                    yield src_line
            epoch_counter += 1
            if bucketed_batching:
                print("Finished round", epoch_counter, "for bucket:", bucket_id, "for language:", lang)
            else:
                print("Finished epoch", epoch_counter, "for language:", lang)
    except Exception as e:
        print(e)
        print("Catastrophic data gen failure")
    return None

def get_bucket_indexed_indefinite_corpus_yielder_bi(corpus, lang, bucket_intervals, sorted_batching, tokenizer, tgt_tokenizer):
    # First we split the corpus into buckets. Each number in the bucket_intervals list is the upper bound of the bucket. The last bucket is the rest of the corpus.
    # We create a dictionary where the key is the bucket id and the value is the yielder for that bucket.
    corpus_split_by_length = {l: [] for l in bucket_intervals}
    for src_line, tgt_line in corpus:
        tgt_line_tok = tgt_tokenizer(tgt_line.strip(), add_special_tokens=False).input_ids
        src_line_tok = tokenizer(src_line.strip(), add_special_tokens=False).input_ids
        tgt_line_length = len(tgt_line_tok)
        src_line_length = len(src_line_tok)
        average_line_length = (tgt_line_length + src_line_length) / 2 ## Hopefully this leads to a compromise between the two.
        # We will index the bucket by the target line length.
        for bucket_id in range(len(bucket_intervals)-1):
            if average_line_length >= bucket_intervals[bucket_id] and average_line_length < bucket_intervals[bucket_id+1]:
                corpus_split_by_length[bucket_intervals[bucket_id]].append((src_line, tgt_line))
                break
        if average_line_length >= bucket_intervals[-1]:
            corpus_split_by_length[bucket_intervals[-1]].append((src_line, tgt_line))
    # Print bucket sizes
    bucket_distributions = []
    for bucket_id in corpus_split_by_length:
        curr_bucket_size = len(corpus_split_by_length[bucket_id])
        print("For language", lang, "the number of lines in bucket", bucket_id, "is", curr_bucket_size)
        bucket_distributions.append(curr_bucket_size)
    total_lines = sum(bucket_distributions)
    bucket_distributions = [x/total_lines for x in bucket_distributions]
    # Now we create a yielder for each bucket.
    bucket_yielders = {}
    for bucket_id in corpus_split_by_length:
        if len(corpus_split_by_length[bucket_id]) > 0:
            bucket_yielders[bucket_id] = yield_corpus_indefinitely_bi(corpus_split_by_length[bucket_id], lang, sorted_batching, bucketed_batching=True, bucket_id=bucket_id, tokenizer=tokenizer, tgt_tokenizer=tgt_tokenizer)
    
    return bucket_yielders, bucket_distributions

def yield_corpus_indefinitely_bi(corpus, language, sorted_batching, bucketed_batching=False, bucket_id=None, tokenizer=None, tgt_tokenizer=None):
    """This shuffles the corpus at the beginning of each epoch and returns sentences indefinitely."""
    epoch_counter = 0
    num_lines = len(corpus)
    num_sentences_before_sort = 20000
    num_sorted_segments = (num_lines // num_sentences_before_sort) + 1
    while True:
        if bucketed_batching:
            print("Shuffling bucket ID:", bucket_id, "for language:", language)
        else:
            print("Shuffling corpus:", language)
        random.shuffle(corpus)
        sys.stdout.flush()
        if sorted_batching:
            for curr_segment_id in range(num_sorted_segments):
                curr_segment = corpus[curr_segment_id*num_sentences_before_sort:(curr_segment_id+1)*num_sentences_before_sort]
                for src_line, tgt_line in sorted(curr_segment, key=lambda x: (len(tokenizer(x[0].strip(), add_special_tokens=False).input_ids) + len(tgt_tokenizer(x[1].strip(), add_special_tokens=False).input_ids)) / 2):
                    yield src_line, tgt_line
        else:
            for src_line, tgt_line in corpus:
                yield src_line, tgt_line
        epoch_counter += 1
        if bucketed_batching:
            print("Finished round", epoch_counter, "for bucket:", bucket_id, "for language:", language)
        else:
            print("Finished epoch", epoch_counter, "for language:", language)
    return None, None ## We should never reach this point.

def sub_sample_and_permute_document(sentence, document_level_sentence_delimiter, max_length):
    """Here we start at a particular random index and select the rest of the sentences. This is to make sure that we dont always see only the initial part of each document all the time."""
    sentence_split = sentence.split(" "+document_level_sentence_delimiter+" ")
    length_histogram = []
    current_length = 0
    for sent in sentence_split:
        current_length += len(sent.split(" "))
        length_histogram.append(current_length)
    len_space_sentence_split = length_histogram[-1]
    if len_space_sentence_split > max_length:
        max_token_idx = len_space_sentence_split - max_length
        for idx, token_idx in enumerate(length_histogram):
            max_last_sentence_idx = idx
            if token_idx > max_token_idx:
                break
    else:
        max_last_sentence_idx = 0
    start_idx = random.randint(0, max_last_sentence_idx)
    sentence_split = sentence_split[start_idx:]
    sentence = " ".join(sentence_split)
    sent_len = len(sentence.split(" "))
    sentence_split_shuffled = random.sample(sentence_split, len(sentence_split))
    sentence_split_shuffled = " ".join(sentence_split_shuffled)
    sentence_split_shuffled = sentence_split_shuffled.split(" ")
    return sentence_split_shuffled, sentence, sent_len


def generate_ul2_input_and_output(split_sentence, args, current_ul2_denoising_style = None):
    """Generates the input and output for the UL2 objective model."""
    # Randomly select a denoising style
    denoising_style = random.choice(args.denoising_styles) if current_ul2_denoising_style is None else current_ul2_denoising_style
    max_retries = args.max_masking_retries
    curr_retries = 0
    if denoising_style == "R": 
        original_sentence_length = len(split_sentence)
        masked_so_far = 0
        spans = []
        max_to_mask = int(args.ul2_r_max_to_mask * original_sentence_length)
        # While 15% of the tokens are not masked, do the following.
        while True:
            # Choose an index at random.
            possible_spans = [np.random.normal(s, args.ul2_r_spans_std) for s in args.ul2_r_spans_mean]
            # Choose a span length at random.
            span_length = int(random.choice(possible_spans))
            # If the span length is greater than the length of the sentence, then choose a new span length.
            if span_length > len(split_sentence) or span_length == 0:
                curr_retries += 1
                if curr_retries >= max_retries:
                    break
                continue
            # Else extract the span and replace it with a mask token.
            else:
                random_idx = random.randint(0, len(split_sentence)-span_length)
                span = split_sentence[random_idx:random_idx+span_length]
                # if the mask token is already present, then choose a new span.
                if "[MASK]" in span:
                    curr_retries += 1
                    if curr_retries >= max_retries:
                        break
                    continue
                spans.append((random_idx, span, span_length))
                split_sentence[random_idx:random_idx+span_length] = ["[MASK]"] * span_length
                masked_so_far += len(span)
                if masked_so_far >= max_to_mask:
                    break
        # Concatenate the spans to form the output. Use <s> as the separator.
        output_sentence = ""
        for _, span, _ in sorted(spans, key=lambda x: x[0]):
            output_sentence += " ".join(span) + " <s> "
        output_sentence = output_sentence.strip()
        # Concatenate the tokens to form the input.
        for pos, _, span_length in sorted(spans, key=lambda x: x[0], reverse=True):
            split_sentence[pos: pos + span_length] = ["[MASK]"]
        input_sentence = " ".join(split_sentence)
        # Prepend the input with the denoising style.
        if args.ignore_paradigm_token:
            pass
        else:
            input_sentence = "<R> " + input_sentence
    elif denoising_style == "S":
        # Choose an index uniformly between beginning and end of sentence subject to maximum and minimum masking percentage.
        random_idx = random.randint(int(args.ul2_s_min_prefix_to_keep * len(split_sentence)), int(args.ul2_s_max_prefix_to_keep * len(split_sentence)))
        # Extract extract the span from the index to the end of the sentence.
        span = split_sentence[random_idx:]
        # Replace the span with a mask token.
        split_sentence[random_idx:] = ["[MASK]"]
        # Concatenate the tokens to form the input.
        input_sentence = " ".join(split_sentence)
        # Concatenate the span to form the output. Use <s> as the separator.
        output_sentence = " ".join(span)
        # Prepend the input with the denoising style.
        if args.ignore_paradigm_token:
            pass
        else:
            input_sentence = "<S> " + input_sentence 
    elif denoising_style == "X":
        original_sentence_length = len(split_sentence)
        masked_so_far = 0
        spans = []
        # Choose a subtype of X denoising style.
        X_denoising_style = random.choice(args.x_denoising_styles)
        if X_denoising_style == "LL": # Same as R denoising style but with span lengths with mean of 64 and standard deviation of 1.
            max_to_mask = int(args.ul2_x_ll_max_to_mask * original_sentence_length)
            while True:
                span_length = int(np.random.normal(args.ul2_x_ll_span_mean, args.ul2_x_ll_span_std))
                if span_length > len(split_sentence) or span_length == 0:
                    curr_retries += 1
                    if curr_retries >= max_retries:
                        break
                    continue
                else:
                    random_idx = random.randint(0, len(split_sentence)-span_length)
                    span = split_sentence[random_idx:random_idx+span_length]
                    if "[MASK]" in span:
                        curr_retries += 1
                        if curr_retries >= max_retries:
                            break
                        continue
                    spans.append((random_idx, span, span_length))
                    split_sentence[random_idx:random_idx+span_length] = ["[MASK]"] * span_length
                    masked_so_far += len(span)
                    if masked_so_far >= max_to_mask:
                        break
        elif X_denoising_style == "LH": # Same as above but with masking limit of 0.5.
            max_to_mask = int(args.ul2_x_lh_max_to_mask * original_sentence_length)
            while True:
                span_length = int(np.random.normal(args.ul2_x_lh_span_mean, args.ul2_x_lh_span_std))
                if span_length > len(split_sentence) or span_length == 0:
                    curr_retries += 1
                    if curr_retries >= max_retries:
                        break
                    continue
                else:
                    random_idx = random.randint(0, len(split_sentence)-span_length)
                    span = split_sentence[random_idx:random_idx+span_length]
                    if "[MASK]" in span:
                        curr_retries += 1
                        if curr_retries >= max_retries:
                            break
                        continue
                    spans.append((random_idx, span, span_length))
                    split_sentence[random_idx:random_idx+span_length] = ["[MASK]"] * span_length
                    masked_so_far += len(span)
                    if masked_so_far >= max_to_mask:
                        break
        elif X_denoising_style == "SH": # Same as R denoising style but with masking limit of 0.5.
            max_to_mask = int(args.ul2_x_sh_max_to_mask * original_sentence_length)
            while True:
                possible_spans = [np.random.normal(s, args.ul2_x_sh_spans_std) for s in args.ul2_x_sh_spans_mean]
                # Choose a span length at random.
                span_length = int(random.choice(possible_spans))
                # If the span length is greater than the length of the sentence, then choose a new span length.
                if span_length > len(split_sentence) or span_length == 0:
                    if curr_retries >= max_retries:
                        break
                    continue
                # Else extract the span and replace it with a mask token.
                else:
                    random_idx = random.randint(0, len(split_sentence)-span_length)
                    span = split_sentence[random_idx:random_idx+span_length]
                    # if the mask token is already present, then choose a new span.
                    if "[MASK]" in span:
                        if curr_retries >= max_retries:
                            break
                        continue
                    spans.append((random_idx, span, span_length))
                    split_sentence[random_idx:random_idx+span_length] = ["[MASK]"] * span_length
                    masked_so_far += len(span)
                    if masked_so_far >= max_to_mask:
                        break
        # Concatenate the spans to form the output. Use <s> as the separator.
        output_sentence = ""
        for _, span, _ in sorted(spans, key=lambda x: x[0]):
            output_sentence += " ".join(span) + " <s> "
        output_sentence = output_sentence.strip()
        # Concatenate the tokens to form the input.
        for pos, _, span_length in sorted(spans, key=lambda x: x[0], reverse=True):
            split_sentence[pos: pos + span_length] = ["[MASK]"]
        input_sentence = " ".join(split_sentence)
        # Prepend the input with the denoising style.
        if args.ignore_paradigm_token:
            pass
        else:
            input_sentence = "<X> " + input_sentence
    
    return input_sentence, output_sentence

def generate_mbart_or_mt5_input_and_output(split_sentence, original_sentence, mask_percent, mask_tok, args):
    original_sentence_length = len(split_sentence)
    masked_so_far = 0
    spans = []
    max_to_mask = int(mask_percent * original_sentence_length)
    max_retries = args.max_masking_retries
    curr_retries = 0
    # While X% of the tokens are not masked, do the following.
    while True:
        # Choose an index at random.
        span_length = np.random.poisson(args.token_masking_lambda)
        # If the span length is greater than the length of the sentence, then choose a new span length.
        if span_length > len(split_sentence) or span_length == 0:
            curr_retries += 1
            if curr_retries >= max_retries:
                break
            continue
        # Else extract the span and replace it with a mask token.
        else:
            random_idx = random.randint(0, len(split_sentence)-span_length)
            span = split_sentence[random_idx:random_idx+span_length]
            # if the mask token is already present, then choose a new span.
            if mask_tok in span:
                curr_retries += 1
                if curr_retries >= max_retries:
                    break
                continue
            spans.append((random_idx, span, span_length))
            split_sentence[random_idx:random_idx+span_length] = [mask_tok] * span_length
            masked_so_far += len(span)
            if masked_so_far >= max_to_mask:
                break
    # Concatenate the spans to form the output. Use <s> as the separator.
    output_sentence = ""
    for _, span, _ in sorted(spans, key=lambda x: x[0]):
        output_sentence += " ".join(span) + " <s> "
    output_sentence = output_sentence.strip()
    # Concatenate the tokens to form the input.
    for pos, _, span_length in sorted(spans, key=lambda x: x[0], reverse=True):
        split_sentence[pos: pos + span_length] = [mask_tok]
    input_sentence = " ".join(split_sentence)
    
    if args.span_prediction: # mt5
        return input_sentence, output_sentence
    elif args.span_to_sentence_prediction: # reverse mt5
        return output_sentence, original_sentence
    else: ## mbart
        return input_sentence, original_sentence

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
    language_list = [lang for lang, _ in files]
    print("Training for:", language_list)
    language_file_dict = []
    probs = []
    language_indices = [i for i in range(len(language_list))]
    if args.bucketed_batching:
        print("Bucketed batching will be used.")
        args.bucket_intervals.insert(0, 0)
        all_bucket_stats = {}

    for lang, file_details in files:
        file_content = open(file_details[0]+"."+"%02d" % rank).readlines() if args.num_domains_for_domain_classifier > 1 else open(file_details+"."+"%02d" % rank).readlines()
        probs.append(len(file_content))
        if args.bucketed_batching:
            bucket_yielder, bucket_stats = get_bucket_indexed_indefinite_corpus_yielder_mono(file_content, lang, args.bucket_intervals, args.sorted_batching, tok)
            language_file_dict.append(bucket_yielder)
            all_bucket_stats[lang] = bucket_stats
        else:
            language_file_dict.append(yield_corpus_indefinitely_mono(file_content, lang, args.sorted_batching, bucketed_batching=False, bucket_id=None, tokenizer=tok))
    probs_temp = [probval/sum(probs) for probval in probs]
    probs = probs_temp
    probs_temp = [probsval**(1.0/args.data_sampling_temperature) for probsval in probs] ## Temperature sampling probabilities.
    probs = probs_temp
    probs_temp = [probsval/sum(probs) for probsval in probs]
    probs = probs_temp
    # Normalize bucket stats across languages. For each bucket, add the stats across languages and divide by the sum.
    if args.bucketed_batching:
        bucket_stats = []
        for i in range(len(args.bucket_intervals)):
            bucket_stats.append(sum([all_bucket_stats[lang][i] for lang in language_list]))
        bucket_stats = [statval/sum(bucket_stats) for statval in bucket_stats]
    print("Language probabilities:", probs)
    if args.bucketed_batching:
        print("Bucket probabilities:", bucket_stats)

    dropped_sentence = "" ## We will save the sentence to be dropped this batch and add it to the next batch.
    dropped_language = "" ## We will save the language to be dropped this batch and add it to the next batch.
    while batch_count != (args.num_batches*args.multistep_optimizer_steps):
        curr_batch_count = 0
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_label_batch = []
        batch_count += 1
        max_src_sent_len = 0
        max_tgt_sent_len = 0
        start = time.time()
        sents_in_batch = 0
        if args.ul2_denoising and args.ul2_denoising_same_objective_per_batch:
            current_ul2_denoising_style = random.choice(args.denoising_styles)
        if args.bucketed_batching:
            current_bucket = random.choices(args.bucket_intervals, bucket_stats)[0] # Its a chicken and egg problem. Since I need to choose sentences from the same bucket, I need to know the bucket first and the bucket distributions are language dependent. So I normalize the bucket distributions across languages and then sample from it. Not ideal but ill take what I can get.
        if args.num_domains_for_domain_classifier > 1:
            domain_classifier_labels = []
        while True:
            if dropped_sentence != "": ## If we have a dropped sentence from the previous batch, then we will use it in this batch. However, when doing UL2, if we ever want to use the same objective per batch, then we will end up having a different type of objective applied to the dropped sentence. This is not ideal but I dont think it will matter much.
                language = dropped_language # Reuse the previous language
                sentence = dropped_sentence # Reuse the previous sentence
                dropped_sentence = ""
                dropped_language = ""
            else:
                language_index = random.choices(language_indices, probs)[0]
                language = language_list[language_index]
                if args.bucketed_batching:
                    if current_bucket in language_file_dict[language_index]:
                        sentence = next(language_file_dict[language_index][current_bucket]).strip()
                    else:
                        continue
                else:
                    sentence = next(language_file_dict[language_index]).strip()
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
                    sentence_split = sentence_split[:args.max_length] ## For MT this makes sense but for pre-training, does this even matter? Shouldnt I just take a starting point between 0 to (len(sentence_split)-max_length)
                    sentence = " ".join(sentence_split)
                    sent_len = args.max_length
            
            if args.ul2_denoising:
                masked_sentence, sentence = generate_ul2_input_and_output(sentence_split, args, current_ul2_denoising_style=current_ul2_denoising_style)
            else:
                masked_sentence, sentence = generate_mbart_or_mt5_input_and_output(sentence_split, sentence, mask_percent, mask_tok, args) ## Simply preserving the original code's args.
            
            if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit. Not touching this for the UL2 work.
                iids = tok(masked_sentence).input_ids
                curr_src_sent_len = len(iids)
                if curr_src_sent_len - 2 > args.hard_truncate_length: ##Ignoring the BOS and EOS tokens.
                    masked_sentence = tok.decode(iids[1:args.hard_truncate_length+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_src_sent_len = args.hard_truncate_length
                iids = tok(sentence).input_ids
                curr_tgt_sent_len = len(iids)
                if curr_tgt_sent_len - 2 > args.hard_truncate_length: ##Ignoring the BOS and EOS tokens
                    sentence = tok.decode(iids[1:args.hard_truncate_length+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_tgt_sent_len = args.hard_truncate_length
            else:
                iids = tok(masked_sentence, add_special_tokens=False).input_ids
                curr_src_sent_len = len(iids)
                if curr_src_sent_len > args.hard_truncate_length:
                    masked_sentence = tok.decode(iids[:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_src_sent_len = args.hard_truncate_length
                iids = tok(sentence, add_special_tokens=False).input_ids
                curr_tgt_sent_len = len(iids)
                if curr_tgt_sent_len > args.hard_truncate_length:
                    sentence = tok.decode(iids[:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_tgt_sent_len = args.hard_truncate_length
            
                        
            if curr_src_sent_len > max_src_sent_len:
                max_src_sent_len = curr_src_sent_len
            
            if curr_tgt_sent_len > max_tgt_sent_len:
                max_tgt_sent_len = curr_tgt_sent_len
            
            if args.batch_size_indicates_lines: ## Batch a fixed number of sentences. We can safely add the current example because we assume that the user knows the max batch size.
                if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit. Not touching this for the UL2 work.
                    encoder_input_batch.append(masked_sentence)
                    decoder_input_batch.append(sentence)
                else:
                    if args.use_official_pretrained and "50" in args.pretrained_model: ## mbart-50 model has a different input representation
                        encoder_input_batch.append(lang + " " + masked_sentence + " </s>")
                    else:
                        if args.ul2_denoising: # Lets not prime the encoder with language indicators because we want to be able to prompt with in-context examples which can be multilingual.
                            encoder_input_batch.append(masked_sentence)
                        else:
                            encoder_input_batch.append(masked_sentence + " </s> " + lang)
                    if args.ul2_denoising: # Lets not prime the decoder with language indicators because we want to be able to generate multilingually. But we do need a starting token so lets use <s> as usual.
                        decoder_input_batch.append("<s> " + sentence)
                        decoder_label_batch.append(sentence + " </s>")
                    else:
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
                    elif args.bucketed_batching: ## We will drop it for now but we will use it in the next iteration. This is semi optimal but we dont want to include a sentence from one bucket into another bucket.
                        dropped_sentence = ""
                        dropped_language = ""
                    else:
                        dropped_sentence = sentence
                        dropped_language = language
                    break
                if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit. Not touching this for the UL2 work.
                    encoder_input_batch.append(masked_sentence)
                    decoder_input_batch.append(sentence)
                else:
                    if args.use_official_pretrained and "50" in args.pretrained_model: ## mbart-50 model has a different input representation
                        encoder_input_batch.append(lang + " " + masked_sentence + " </s>")
                    else:
                        if args.ul2_denoising: # Lets not prime the encoder with language indicators because we want to be able to prompt with in-context examples which can be multilingual.
                            encoder_input_batch.append(masked_sentence)
                        else:
                            encoder_input_batch.append(masked_sentence + " </s> " + lang)
                    if args.ul2_denoising: # Lets not prime the decoder with language indicators because we want to be able to generate multilingually. But we do need a starting token so lets use <s> as usual.
                        decoder_input_batch.append("<s> " + sentence)
                        decoder_label_batch.append(sentence + " </s>")
                    else:
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
        input_masks = (input_ids != tok.pad_token_id).int()
        if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit. No support for stochastic tokenizer because the roberta tokenizer which is inherited from GPT2 tokenizer does its onw weird BPE and I dont want to mess with it.
            decoder_input_ids = tok(decoder_input_batch, return_tensors="pt", padding=True).input_ids
        else:
            decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
        if (args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model) or args.tokenization_sampling: ## We have to be careful when using stochastic segmentation. Note again that there will be no stoachastic segmentation with the official bart model. IT JUST WONT WORK.
            labels = decoder_input_ids[:,1:]
            decoder_input_ids = decoder_input_ids[:,:-1]
        else:
            labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        end = time.time()
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
    while batch_count != (args.num_batches*args.multistep_optimizer_steps):
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
    model_grads = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

def grad_status(model):
    """Checks whether the parameter needs gradient or not. Part of asserting that the correct parts of the model are frozen."""
    return (par.requires_grad for par in model.parameters())


def freeze_params(model, exception="none", rank=0):
    """Set requires_grad=False for each of model.parameters() thereby freezing those parameters. We use this when we want to prevent parts of the model from being trained."""
    if exception == "none":
        for par in model.parameters():
            par.requires_grad = False
    elif exception is not None:
        exception = exception.split(",")
        if rank == 0:
            print("Freezing all parameters except those containing the following strings:", exception)
        for name, par in model.named_parameters():
            if not any(individual_exception in name for individual_exception in exception):
                if rank == 0:
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
            iids = tok(src_sent).input_ids
            sent_len = len(iids)
            if sent_len - 2 > args.hard_truncate_length: ##Ignoring the BOS and EOS tokens
                src_sent = tok.decode(iids[1:args.hard_truncate_length+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                sent_len = args.hard_truncate_length
            encoder_input_batch.append(src_sent)
        else:
            iids = tok(src_sent, add_special_tokens=False).input_ids
            sent_len = len(iids)
            if sent_len > args.hard_truncate_length:
                src_sent = tok.decode(iids[:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
                iids = tok(src_sent_parent).input_ids
                sent_len_parent = len(iids)
                if sent_len_parent - 2 > args.hard_truncate_length: ##Ignoring the BOS and EOS tokens
                    src_sent_parent = tok.decode(iids[1:args.hard_truncate_length+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    sent_len_parent = args.hard_truncate_length
                encoder_input_batch_parent.append(src_sent_parent)
            else:
                iids = tok(src_sent_parent, add_special_tokens=False).input_ids
                sent_len_parent = len(iids)
                if sent_len_parent > args.hard_truncate_length:
                    src_sent_parent = tok.decode(iids[:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    sent_len_parent = args.hard_truncate_length
                if args.use_official_pretrained and "50" in args.pretrained_model: ## mbart-50 model has a different input representation
                    encoder_input_batch_parent.append(lang_parent + " " + src_sent_parent + " </s>")
                else:
                    encoder_input_batch_parent.append(src_sent_parent + " </s> " + lang_parent)

            

        curr_batch_count += 1
        if curr_batch_count == args.dev_batch_size:
            if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit.
                input_ids = tok(encoder_input_batch, return_tensors="pt", padding=True).input_ids
            else:
                input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            input_masks = (input_ids != tok.pad_token_id).int()
            
            if args.multi_source: ## Process the batch for the additional source as well.
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
                input_masks_parent = (input_ids_parent != tok.pad_token_id).int()
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
        input_masks = (input_ids != tok.pad_token_id).int()
        
        if args.multi_source: ## Process the batch for the additional source as well.
            input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            input_masks_parent = (input_ids_parent != tok.pad_token_id).int()
            yield [input_ids, input_ids_parent], [input_masks, input_masks_parent]
        else:
            yield input_ids, input_masks


def generate_batches_bilingual(tok, args, files, rank, tgt_tok=None):
    """Generates the source, target and source attention masks for the training set. The source and target sentences are ignored if empty and are truncated if longer than a threshold. The batch size in this context is the maximum number of tokens in the batch post padding."""
    if args.tokenization_sampling:
        print("Stochastic tokenizer will be used.")
        if "mbart" in args.tokenizer_name_or_path:
            print("BPE dropout with a dropout probability of", args.tokenization_alpha_or_dropout, "will be used.")
        else:
            print("Sentencepiece regularization with an alpha value of", args.tokenization_alpha_or_dropout, "will be used.")

    if tgt_tok is None: # If the target tokenizer is not provided, use the source tokenizer which is a joint tokenizer for both source and target.
        tgt_tok = tok

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

    language_list = [lang for lang, _ in files]
    print("Training for:", language_list)
    language_file_dict = []
    probs = []
    language_indices = [i for i in range(len(language_list))]
    if args.bucketed_batching:
        print("Bucketed batching will be used.")
        args.bucket_intervals.insert(0, 0)
        all_bucket_stats = {}

    for lang, file_details in files:
        src_file_content = open(file_details[0]+"."+"%02d" % rank).readlines()
        tgt_file_content = open(file_details[1]+"."+"%02d" % rank).readlines()
        probs.append(len(src_file_content))
        file_content = list(zip(src_file_content, tgt_file_content))
        if args.bucketed_batching:
            bucket_yielder, bucket_stats = get_bucket_indexed_indefinite_corpus_yielder_bi(file_content, lang, args.bucket_intervals, args.sorted_batching, tok, tgt_tok)
            language_file_dict.append(bucket_yielder)
            all_bucket_stats[lang] = bucket_stats
        else:
            language_file_dict.append(yield_corpus_indefinitely_bi(file_content, lang, args.sorted_batching, bucketed_batching=False, bucket_id=None, tokenizer=tok, tgt_tokenizer=tgt_tok))
    print("Corpora stats:", probs)
    probs_temp = [probval/sum(probs) for probval in probs]
    probs = probs_temp
    probs_temp = [probsval**(1.0/args.data_sampling_temperature) for probsval in probs] ## Temperature sampling probabilities.
    probs = probs_temp
    probs_temp = [probsval/sum(probs) for probsval in probs]
    probs = probs_temp
    # Normalize bucket stats across languages. For each bucket, add the stats across languages and divide by the sum.
    if args.bucketed_batching:
        bucket_stats = []
        for i in range(len(args.bucket_intervals)):
            bucket_stats.append(sum([all_bucket_stats[lang][i] for lang in language_list]))
        bucket_stats = [statval/sum(bucket_stats) for statval in bucket_stats]
    print("Corpora sampling probabilities:", probs)
    if args.bucketed_batching:
        print("Bucket stats (post tokenization length):", bucket_stats)
    dropped_source_sentence = "" ## We will save the source sentence to be dropped this batch and add it to the next batch.
    dropped_target_sentence = "" ## We will save the target sentence to be dropped this batch and add it to the next batch.
    dropped_language = "" ## We will save the language indicator to be dropped this batch and add it to the next batch.
    if args.cross_distillation or args.multi_source: ## We assume an additional source language.
        dropped_source_sentence_parent = "" ## We will save the source parent sentence to be dropped this batch and add it to the next batch.
    while batch_count != (args.num_batches*args.multistep_optimizer_steps): ## Note that in case of multistep optimizer, each model step will see multistep_optimizer_steps batches per gpu before updating. Thus if I dont multiply by multistep_optimizer_steps, my data generator will end around num_batches/multistep_optimizer_steps. 
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
        if args.bucketed_batching:
            bucket_index = random.choices(args.bucket_intervals, bucket_stats)[0] # Its a chicken and egg problem. Since I need to choose sentences from the same bucket, I need to know the bucket first and the bucket distributions are language dependent. So I normalize the bucket distributions across languages and then sample from it. Not ideal but ill take what I can get.
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
                language_index = random.choices(language_indices, probs)[0]
                language = language_list[language_index]
                if args.bucketed_batching:
                    if bucket_index in language_file_dict[language_index]:
                        src_sent, tgt_sent = next(language_file_dict[language_index][bucket_index])
                    else:
                        continue
                else:
                    src_sent, tgt_sent = next(language_file_dict[language_index])
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
                if type(mp_val_or_range) is float:
                    mask_percent = mp_val_or_range
                else:
                    mask_percent = random.uniform(mp_val_or_range[0], mp_val_or_range[1])
                
                if args.source_masking_for_bilingual:
                    assert (not args.span_to_sentence_prediction)
                    assert (not args.span_prediction)
                    ## Careful not to use args.span_to_sentence_prediction or args.span_prediction when using args.source_masking_for_bilingual. If you do use the latter two flags, your tgt_sent will be wiped out. These two flags wont mix well.
                else:
                    pass
                
                if args.ul2_denoising:
                    masked_sentence, sentence = generate_ul2_input_and_output(src_sent_split, args, current_ul2_denoising_style=None) ## We dont really care about padding since its out of wack for mixed MT and denoising objectives. TODO: See if theres a possibility to do something interesting here.
                else:
                    src_sent, tgt_sent = generate_mbart_or_mt5_input_and_output(src_sent_split, tgt_sent, mask_percent, mask_tok, args) ## Simply preserving the original code's args.
                # src_sent, tgt_sent = generate_mbart_or_mt5_input_and_output(src_sent_split, tgt_sent, mask_percent, mask_tok, args) ## Careful not to use args.span_to_sentence_prediction or args.span_prediction when using args.source_masking_for_bilingual. If you do use the latter two flags, your tgt_sent will be wiped out. These two flags wont mix well.

            if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                iids = tok(src_sent).input_ids
                curr_src_sent_len = len(iids)
                if curr_src_sent_len - 2 > args.hard_truncate_length: ##Ignoring the BOS and EOS tokens
                    src_sent = tok.decode(iids[1:args.hard_truncate_length+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_src_sent_len = args.hard_truncate_length
                
                iids = tgt_tok(tgt_sent).input_ids
                curr_tgt_sent_len = len(iids)
                if curr_tgt_sent_len - 2 > args.hard_truncate_length: ##Ignoring the BOS and EOS tokens
                    tgt_sent = tgt_tok.decode(iids[1:args.hard_truncate_length+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_tgt_sent_len = args.hard_truncate_length
            else:
                iids = tok(src_sent, add_special_tokens=False).input_ids
                curr_src_sent_len = len(iids)
                if curr_src_sent_len > args.hard_truncate_length:
                    src_sent = tok.decode(iids[:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_src_sent_len = args.hard_truncate_length
                
                iids = tgt_tok(tgt_sent, add_special_tokens=False).input_ids
                curr_tgt_sent_len = len(iids)
                if curr_tgt_sent_len > args.hard_truncate_length:
                    tgt_sent = tgt_tok.decode(iids[:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    curr_tgt_sent_len = args.hard_truncate_length
            
            if args.cross_distillation or args.multi_source:
                if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                    iids = tok(src_sent_parent, add_special_tokens=False).input_ids
                    curr_src_sent_len_parent = len(iids)
                    if curr_src_sent_len_parent - 2 > args.hard_truncate_length: ##Ignoring the BOS and EOS tokens
                        src_sent_parent = tok.decode(iids[1:args.hard_truncate_length+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        curr_src_sent_len_parent = args.hard_truncate_length
                else:
                    iids = tok(src_sent_parent, add_special_tokens=False).input_ids
                    curr_src_sent_len_parent = len(iids)
                    if curr_src_sent_len_parent > args.hard_truncate_length:
                        src_sent_parent = tok.decode(iids[:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
                    elif args.bucketed_batching: ## We will drop these for now but we will use it in the next iteration. This is semi optimal but we dont want to include a sentence from one bucket into another bucket.
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
                            dropped_source_sentence_parent = ""
                        elif args.bucketed_batching: ## We will drop these for now but we will use it in the next iteration. This is semi optimal but we dont want to include a sentence from one bucket into another bucket.
                            dropped_source_sentence_parent = ""
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
        input_masks = (input_ids != tok.pad_token_id).int()
        if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            decoder_input_ids = tgt_tok(decoder_input_batch, return_tensors="pt", padding=True).input_ids
        else:
            decoder_input_ids = tgt_tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
        if (args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model) or args.tokenization_sampling: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            labels = decoder_input_ids[:,1:]
            decoder_input_ids = decoder_input_ids[:,:-1]
        else:
            labels = tgt_tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        if args.cross_distillation or args.multi_source:
            if args.use_official_pretrained and ("bart" in args.pretrained_model or "barthez" in args.pretrained_model) and "mbart" not in args.pretrained_model: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            else:
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
            input_masks_parent = (input_ids_parent != tok.pad_token_id).int()
            end = time.time()
            yield [input_ids, input_ids_parent], [input_masks, input_masks_parent], decoder_input_ids, labels
        else:
            end = time.time()
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
        
        
        if args.use_official_pretrained and ("bart" in args.model_path or "barthez" in args.model_path) and "mbart" not in args.model_path: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
            iids = tok(src_sent).input_ids
            sent_len = len(iids)
            if sent_len - 2 > args.hard_truncate_length: ##Ignoring the BOS and EOS tokens
                src_sent = tok.decode(iids[1:args.hard_truncate_length+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                sent_len = args.hard_truncate_length
            encoder_input_batch.append(src_sent)
        else: 
            iids = tok(src_sent, add_special_tokens=False).input_ids
            sent_len = len(iids)
            if sent_len > args.hard_truncate_length:
                src_sent = tok.decode(iids[0:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
                iids = tok(src_sent_parent).input_ids
                sent_len_parent = len(iids)
                if sent_len_parent - 2 > args.hard_truncate_length:  ##Ignoring the BOS and EOS tokens
                    src_sent_parent = tok.decode(iids[1:args.hard_truncate_length+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    sent_len_parent = args.hard_truncate_length
                encoder_input_batch.append(src_sent_parent)
            else: 
                iids = tok(src_sent_parent, add_special_tokens=False).input_ids
                sent_len_parent = len(iids)
                if sent_len_parent > args.hard_truncate_length:
                    src_sent_parent = tok.decode(iids[0:args.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
            input_masks = input_ids != tok.pad_token_id
            end = time.time()
            if args.multi_source: ## Process the batch for the additional source as well.
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
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
        input_masks = input_ids != tok.pad_token_id
        if args.multi_source: ## Process the batch for the additional source as well.
            if args.use_official_pretrained and ("bart" in args.model_path or "barthez" in args.model_path) and "mbart" not in args.model_path: ## The bart tokenizer is wacky so we need to tweak the inputs a bit
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
            else:
                input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True, sample=args.tokenization_sampling, nbest=args.tokenization_nbest_list_size, alpha_or_dropout=args.tokenization_alpha_or_dropout).input_ids
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