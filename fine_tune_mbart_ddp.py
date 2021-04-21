# -*- coding: utf-8 -*-
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
from torch.optim import Adam
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
import functools
##

## Seed setting here
torch.manual_seed(621311)
##


def assert_all_frozen(model):
    """Checks if frozen parameters are all linked to each other or not. Ensures no disjoint components of graphs."""
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

def grad_status(model):
    """Checks whether the parameter needs gradient or not. Part of asserting that the correct parts of the model are frozen."""
    return (par.requires_grad for par in model.parameters())


def freeze_params(model):
    """Set requires_grad=False for each of model.parameters() thereby freezing those parameters. We use this when we want to prevent parts of the model from being trained."""
    for par in model.parameters():
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

def generate_batches_eval(tok, args, file, slang):
    """Generates the source sentences for the dev set. This ensures that long sentences are truncated and then batched. The batch size is the number of sentences and not the number of tokens."""
    src_file = file
    curr_batch_count = 0
    encoder_input_batch = []
    max_src_sent_len = 0

    for src_line in src_file:
        start = time.time()
        src_sent = src_line.strip()
        if args.multi_source: ## We assume that we use a N-way corpus of 3 languages X, Y and Z. We want to distill Y-Z behavior into X-Z where the Y-Z pair also has additional larger corpora but X-Z does not. As such the source sentence should be a tab separated sentence consisting of X[tab]Y.
            src_sent = src_sent.split("\t")
            src_sent_parent = src_sent[1].strip() ## This is the sentence for Y
            src_sent = src_sent[0] ## This is the sentence for X
            slang = slang.split("-")
            slang_parent = slang[1]
            slang = slang[0]
            lang_parent = "<2"+slang_parent+">"
        lang = "<2"+slang+">" ## proceed from here to generate a batch for the multisource development set.
        src_sent_split = src_sent.split(" ")
        sent_len = len(src_sent_split)
        if sent_len > args.max_src_length: ## Initial truncation
            src_sent_split=src_sent_split[:args.max_src_length]
            src_sent = " ".join(src_sent_split)
            sent_len = args.max_src_length
        iids = tok(src_sent + " </s> " + lang, add_special_tokens=False, return_tensors="pt").input_ids
        curr_src_sent_len = len(iids[0])
        if curr_src_sent_len > max_src_sent_len:
            max_src_sent_len = curr_src_sent_len

        encoder_input_batch.append(src_sent + " </s> " + lang)
        curr_batch_count += 1
        if curr_batch_count == args.dev_batch_size:
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
            if args.hard_truncate_length > 0 and len(input_ids[0]) > args.hard_truncate_length:
                input_ids = input_ids[:,:args.hard_truncate_length]
            input_masks = (input_ids != tok.pad_token_id).int()
            end = time.time()
            if args.is_summarization:
                print(input_ids.size(), functools.reduce(lambda x,y: x*y, input_ids.size()))
            yield input_ids, input_masks
            curr_batch_count = 0
            encoder_input_batch = []
            max_src_sent_len = 0

    if len(encoder_input_batch) != 0:
        input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
        if args.hard_truncate_length > 0 and len(input_ids[0]) > args.hard_truncate_length:
            input_ids = input_ids[:,:args.hard_truncate_length]
        input_masks = (input_ids != tok.pad_token_id).int()
        if args.is_summarization:
            print(input_ids.size(), functools.reduce(lambda x,y: x*y, input_ids.size()))
        yield input_ids, input_masks



def generate_batches(tok, args, files, rank, mp_val_or_range=0.3, lamb=3.5):
    """Generates the source, target and source attention masks for the training set. The source and target sentences are ignored if empty and are truncated if longer than a threshold. The batch size in this context is the maximum number of tokens in the batch post padding."""
    batch_count = 0
    language_list = list(files.keys())
    print("Training for:", language_list)
    language_file_dict = {}
    probs = {}
    for l in language_list:
        src_file_content = open(files[l][0]+"."+"%02d" % rank).readlines()
        tgt_file_content = open(files[l][1]+"."+"%02d" % rank).readlines()
        probs[l] = len(src_file_content)
        file_content = list(zip(src_file_content, tgt_file_content))
        language_file_dict[l] = yield_corpus_indefinitely_bi(file_content, l)
    print("Corpora stats:", probs)
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = probs_temp
    probs_temp = {lang: probs[lang]**(1.0/args.data_sampling_temperature) for lang in probs} ## Temperature sampling probabilities.
    probs = probs_temp
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = [probs_temp[lang] for lang in language_list]
    num_langs = len(language_list)
    language_indices = list(range(num_langs))
    while batch_count != args.num_batches:
        curr_batch_count = 0
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_label_batch = []
        batch_count += 1
        max_src_sent_len = 0
        max_tgt_sent_len = 0
        prev_max_src_sent_len = 0
        prev_max_tgt_sent_len = 0
        start = time.time()
        sents_in_batch = 0
        if args.cross_distillation or args.multi_source: ## We assume an additional source language.
            max_src_sent_len_parent = 0
            encoder_input_batch_parent = []
        while True:
            language_idx = random.choices(language_indices, probs)[0]
            src_sent, tgt_sent = next(language_file_dict[language_list[language_idx]])
            if args.cross_distillation or args.multi_source: ## We assume that we use a N-way corpus of 3 languages X, Y and Z. We want to distill Y-Z behavior into X-Z where the Y-Z pair also has additional larger corpora but X-Z does not. As such the source sentence should be a tab separated sentence consisting of X[tab]Y.
                src_sent = src_sent.split("\t")
                src_sent_parent = src_sent[1].strip() ## This is the sentence for Y
                src_sent = src_sent[0] ## This is the sentence for X
            src_sent = src_sent.strip()
            tgt_sent = tgt_sent.strip()
            slangtlang = language_list[language_idx].strip().split("-")
            if args.cross_distillation or args.multi_source: ## In this case only we provide a hyphen separated triplet to represent languages X, Y and Z.
                slang_parent = "<2"+slangtlang[0]+">"
                slang = "<2"+slangtlang[1]+">"
                tlang = "<2"+slangtlang[2]+">"
            else:
                slang = "<2"+slangtlang[0]+">"
                tlang = "<2"+slangtlang[1]+">"
            src_sent_split = src_sent.split(" ")
            tgt_sent_split = tgt_sent.split(" ")
            tgt_sent_len = len(tgt_sent_split)
            src_sent_len = len(src_sent_split)
            
            if src_sent_len <=1 or tgt_sent_len <=1:
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
                if src_sent_len_parent <=1:
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
                spans_to_mask = list(np.random.poisson(lamb, 1000))
                curr_sent_len = src_sent_len
                while mask_count < max_mask_count:
                    try:
                        span_to_mask = spans_to_mask[0]
                        del spans_to_mask[0]
                        if span_to_mask > (max_mask_count-mask_count): ## Cant mask more than the allowable number of tokens.
                            continue
                        idx_to_mask = random.randint(0, (curr_sent_len-1)-(span_to_mask-1))
                        if "[MASK]" not in src_sent_split[idx_to_mask:idx_to_mask+span_to_mask]:
                            src_sent_split[idx_to_mask:idx_to_mask+span_to_mask] = ["[MASK]"]
                            mask_count += span_to_mask
                            curr_sent_len -= (span_to_mask-1)
                    except:
                        break ## If we cannot get a properly masked sentence despite all our efforts then we just give up and continue with what we have so far.
                src_sent = " ".join(src_sent_split)
            iids = tok(src_sent + " </s> " + slang, add_special_tokens=False, return_tensors="pt").input_ids
            curr_src_sent_len = len(iids[0])
            
                
            iids = tok(tlang + " " + tgt_sent, add_special_tokens=False, return_tensors="pt").input_ids
            curr_tgt_sent_len = len(iids[0])

            if curr_src_sent_len > max_src_sent_len:
                prev_max_src_sent_len = max_src_sent_len
                max_src_sent_len = curr_src_sent_len
            
            if curr_tgt_sent_len > max_tgt_sent_len:
                prev_max_tgt_sent_len = max_tgt_sent_len
                max_tgt_sent_len = curr_tgt_sent_len
            
            potential_batch_count = max(max_src_sent_len, max_tgt_sent_len)*(sents_in_batch+1) ## We limit ourselves based on the maximum of either source or target.
            if potential_batch_count > args.batch_size: ## We will drop this sentence for now. It may be used in a future iteration.
                max_src_sent_len = prev_max_src_sent_len
                max_tgt_sent_len = prev_max_tgt_sent_len
                break
            
            encoder_input_batch.append(src_sent + " </s> " + slang)
            decoder_input_batch.append(tlang + " " + tgt_sent)
            decoder_label_batch.append(tgt_sent + " </s>")
            sents_in_batch += 1
            if args.cross_distillation or args.multi_source:
                iids = tok(src_sent_parent + " </s> " + slang_parent, add_special_tokens=False, return_tensors="pt").input_ids
                curr_src_sent_len_parent = len(iids[0])
                if curr_src_sent_len_parent > max_src_sent_len_parent:
                    max_src_sent_len_parent = curr_src_sent_len_parent
                encoder_input_batch_parent.append(src_sent_parent + " </s> " + slang_parent)
                
        if len(encoder_input_batch) == 0:
            print("Zero size batch due to an abnormal example. Skipping empty batch.")
            continue    

        input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
        if args.hard_truncate_length > 0 and len(input_ids[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
            input_ids = input_ids[:,:args.hard_truncate_length]
        input_masks = (input_ids != tok.pad_token_id).int()
        decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
        if args.hard_truncate_length > 0 and len(decoder_input_ids[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
            decoder_input_ids = decoder_input_ids[:,:args.hard_truncate_length]
        labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
        if args.hard_truncate_length > 0 and len(labels[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
            labels = labels[:,:args.hard_truncate_length]
        if args.is_summarization:
            print(input_ids.size(), functools.reduce(lambda x,y: x*y, input_ids.size()), decoder_input_ids.size(), functools.reduce(lambda x,y: x*y, decoder_input_ids.size()))
        if args.cross_distillation or args.multi_source:
            input_ids_parent = tok(encoder_input_batch_parent, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len_parent).input_ids
            if args.hard_truncate_length > 0 and  len(input_ids_parent[0]) > args.hard_truncate_length: ## Truncate again if we exceed the maximum sequence length.
                input_ids_parent = input_ids_parent[:,:args.hard_truncate_length]
            input_masks_parent = (input_ids_parent != tok.pad_token_id).int()
            end = time.time()
            yield [input_ids_parent, input_ids], [input_masks_parent, input_masks], decoder_input_ids, labels
        else:
            end = time.time()
            yield input_ids, input_masks, decoder_input_ids, labels

def model_create_load_run_save(gpu, args, train_files, dev_files):
    """The main function which does the overall training. Should be split into multiple parts in the future. Currently monolithc intentionally."""
    
    rank = args.nr * args.gpus + gpu ## The rank of the current process out of the total number of processes indicated by world_size.
    print("Launching process:", rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    if args.shard_files and rank == 0: ## First shard the data using process 0 aka the prime process or master process. Other processes will wait.
        shard_files_bi(train_files, args.world_size)
    
    dist.barrier() ## Stop other processes from proceeding till sharding is done.
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=False, use_fast=False, keep_accents=True) ## Fast tokenizers are not good because their behavior is weird. Accents should be kept or else the segmentation will be messed up on languages with accented characters. No lower case obviously because we want to train on the original case. Set to false if you are ok with the model not dealing with cases.
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False) ## In case we do summarization.

    print("Tokenizer is:", tok)
    
    print(f"Running DDP checkpoint example on rank {rank}.")
    
    if args.fp16: ## Although the code supports FP16/AMP training, it tends to be unstable in distributed setups so use this carefully.
        print("We will do fp16 training")
        scaler = torch.cuda.amp.GradScaler() ## Gradient scaler which will be used with torch's automatic mixed precision
    else:
        print("We will do fp32 training")
    
    if args.encoder_tying_config is not None:
        print("We will use recurrently stacked layers for the encoder with configuration:", args.encoder_tying_config)
    if args.decoder_tying_config is not None:
        print("We will use recurrently stacked layers for the decoder with configuration:", args.decoder_tying_config)
        
    config = MBartConfig(vocab_size=len(tok), encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_dim, decoder_ffn_dim=args.decoder_ffn_dim, d_model=args.d_model, add_final_layer_norm=args.add_final_layer_norm, normalize_before=args.normalize_before, normalize_embedding=args.normalize_embedding, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], static_position_embeddings=True, encoder_tying_config=args.encoder_tying_config, decoder_tying_config=args.decoder_tying_config, multilayer_softmaxing=args.multilayer_softmaxing, wait_k=args.wait_k) ## Configuration. TODO: Save this configuration somehow.
    model = MBartForConditionalGeneration(config)
    model.train()
    
    if args.distillation: ## When distilling we need a parent model. The creation of the model is in the same way as the child. This model is immediately loaded with some pretrained params and then loaded into the GPU.
        print("We will do distillation from a parent model.")
        parent_config = MBartConfig(vocab_size=len(tok), encoder_layers=args.parent_encoder_layers, decoder_layers=args.parent_decoder_layers, dropout=args.parent_dropout, attention_dropout=args.parent_attention_dropout, activation_dropout=args.parent_activation_dropout, encoder_attention_heads=args.parent_encoder_attention_heads, decoder_attention_heads=args.parent_decoder_attention_heads, encoder_ffn_dim=args.parent_encoder_ffn_dim, decoder_ffn_dim=args.parent_decoder_ffn_dim, d_model=args.parent_d_model, add_final_layer_norm=args.add_final_layer_norm, normalize_before=args.normalize_before, normalize_embedding=args.normalize_embedding, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], static_position_embeddings=True, encoder_tying_config=args.encoder_tying_config, decoder_tying_config=args.decoder_tying_config, wait_k=args.wait_k)
        parent_model = MBartForConditionalGeneration(config)
        parent_model.cuda(gpu)
        parent_model.train() ## We do this to enable dropout but we wont have an optimizer for this so we wont train this model. For now. Future implementations should ask if we want to do co-distill or not. By co-distillation I mean, the parent will learn together with the child.
        parent_model = DistributedDataParallel(parent_model, device_ids=[gpu], output_device=gpu)
        print("Loading a parent model from which distillation will be done.")
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        parent_checkpoint_dict = torch.load(args.parent_pretrained_model, map_location=map_location)
        if type(parent_checkpoint_dict) == dict:
            parent_model.load_state_dict(parent_checkpoint_dict['model'])
        else:
            parent_model.load_state_dict(parent_checkpoint_dict)
            
        parent_model.train()

    torch.cuda.set_device(gpu) ## Set the device to the current GPU. This is different from the rank so keep this in mind.
    
    if args.freeze_embeddings: ## If we wish to freeze the model embeddings. This may be useful when fine-tuning a pretrained model.
        print("Freezing embeddings")
        freeze_embeds(model)
    if args.freeze_encoder: ## If we wish to freeze the encoder itself. This may be useful when fine-tuning a pretrained model.
        print("Freezing encoder")
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    model.cuda(gpu) ## Move the model to the GPU.

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ] ## We suppose that weight decay will be used except for biases and layer norm weights.
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-09) ## Our glorious optimizer.
    
    model = DistributedDataParallel(model, device_ids=[gpu], output_device=gpu) ## This wrapper around the model will enable distributed training.
    model.train()
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.num_batches*args.world_size) ## A warmup and decay scheduler. We use the linear scheduler for now. TODO: Enable other schedulers with a flag.
    
    while scheduler.get_lr()[0] < 1e-7: ## We want to keep a minimum learning rate else for the initial batch or initial few batches barely anything will be learned which is a waste of computation. This minimum value is kept to 1e-7 by default in accordance with previous literature, other implementations and the Paris peace accords.
        scheduler.step()
    print("Initial LR is:", scheduler.get_lr()[0])
    
    if args.pretrained_bilingual_model == "" and args.pretrained_model != "": ## Here we load a pretrained NMT model or a previous checkpoint in case training crashed.
        print("Loading a pretrained mbart model")
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint_dict = torch.load(args.pretrained_model, map_location=map_location)
        if type(checkpoint_dict) == dict:
            model.load_state_dict(remap_embeddings_eliminate_components_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict['model'], 4, args), args), strict=True if (args.remap_encoder == "" and args.remap_decoder == "" and not args.eliminate_encoder_before_initialization and not args.eliminate_decoder_before_initialization and not args.eliminate_embeddings_before_initialization) else False)
        else:
            model.load_state_dict(remap_embeddings_eliminate_components_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict, 3, args), args), strict=True if (args.remap_encoder == "" and args.remap_decoder == "" and not args.eliminate_encoder_before_initialization and not args.eliminate_decoder_before_initialization and not args.eliminate_embeddings_before_initialization) else False)
        ctr = 0
    elif args.pretrained_bilingual_model != "":
        print("Loading a previous checkpoint")
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint_dict = torch.load(args.pretrained_bilingual_model, map_location=map_location)
        if type(checkpoint_dict) == dict:
            model.load_state_dict(remap_embeddings_eliminate_components_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict['model'], 4, args), args), strict=True if (args.remap_encoder == "" and args.remap_decoder == "" and not args.eliminate_encoder_before_initialization and not args.eliminate_decoder_before_initialization and not args.eliminate_embeddings_before_initialization) else False)
            if args.remap_encoder is '' and args.remap_decoder is '': ## No not load optimizers and schedulers when remapping
                optimizer.load_state_dict(checkpoint_dict['optimizer'])
                scheduler.load_state_dict(checkpoint_dict['scheduler'])
            ctr = checkpoint_dict['ctr']
        else:
            model.load_state_dict(remap_embeddings_eliminate_components_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict, 3, args), args), strict=True if (args.remap_encoder == "" and args.remap_decoder == "" and not args.eliminate_encoder_before_initialization and not args.eliminate_decoder_before_initialization and not args.eliminate_embeddings_before_initialization) else False)
            ctr = 0
    else:
        print("Training from scratch")
        ctr = 0
    model.train()
        
    print("Using label smoothing of", args.label_smoothing)
    print("Using gradient clipping norm of", args.max_gradient_clip_value)
    print("Using softmax temperature of", args.softmax_temperature)
    if args.max_ent_weight != -1:
        print("Doing entropy maximization during loss computation.")
    global_sbleu_history = [] ## To save the global evaluation metric history.
    max_global_sbleu = 0 ## Maximum global evaluation metric score.
    max_global_sbleu_step = 0 ## Step at which we achieved the maximum global evaluation metric score.
    individual_sbleu_history = {dev_pair: [] for dev_pair in dev_files} ## For multilingual NMT settings we suppose that we will keep a track of the histories for individual language pairs being evaluated and this dictionary keeps track of the history.
    max_individual_sbleu = {dev_pair: 0 for dev_pair in dev_files} ## The maximum score per pair.
    max_individual_sbleu_step = {dev_pair: 0 for dev_pair in dev_files} ## The step at which maximum score was achieved per pair.
    curr_eval_step = 0
    annealing_attempt = 0 ## We use this to limit the number of times annealing will take place. When we anneal the LR is divided by a factor. How this is achieved will be explained below.
    inps = {dev_pair: [inpline.strip() for inpline in open(dev_files[dev_pair][0])][:args.max_eval_batches*args.dev_batch_size] for dev_pair in dev_files} ## Get all inputs for each pair. Select up to args.max_eval_batches*args.dev_batch_size examples.
    if args.is_summarization: ## Slight data structure difference for summarization vs translation when computing the evaluation metric. For summarization the metric is Rouge.
        refs = {dev_pair: [[refline.strip() for refline in open(dev_files[dev_pair][1])][:args.max_eval_batches*args.dev_batch_size]] for dev_pair in dev_files} ## Get all references for each input. Select up to args.max_eval_batches*args.dev_batch_size examples.
        scores = {dev_pair: 0 for dev_pair in dev_files} ## The rouge scorer works at the sentence level so we have to add all individual scores per sentence and this dictionary keeps track of the score. This dictionary may not be needed.
    else:
        refs = {dev_pair: [[refline.strip() for refline in open(dev_files[dev_pair][1])][:args.max_eval_batches*args.dev_batch_size]] for dev_pair in dev_files} ## Get all references for each input. Select up to args.max_eval_batches*args.dev_batch_size examples.
    for input_ids, input_masks, decoder_input_ids, labels in generate_batches(tok, args, train_files, rank, (0.30, 0.40), 3.5): #Batches are generated from here. The argument (0.30, 0.40) is a range which indicates the percentage of the source sentence to be masked in case we want masking during training just like we did during BART pretraining. The argument 3.5 is the lambda to the poisson length sampler which indicates the average length of a word sequence that will be masked.
        start = time.time()
        if ctr % args.eval_every == 0: ## We have to evaluate our model every eval_every steps.
            CHECKPOINT_PATH = args.fine_tuned_model
            if rank == 0: ## Evaluation will be done only on the prime/master process which is at rank 0. Other processes will sleep.
                if not args.no_eval: ## If we dont care about early stopping and only on training for a bazillion batches then you can save time by skipping evaluation.
                    print("Running eval on dev set(s)")
                    hyp = {dev_pair: [] for dev_pair in dev_files}
                    sbleus = {}
                    model.eval() ## We go to eval mode so that there will be no dropout.
                    checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr} ## This training state will be saved.
                    for dev_pair in dev_files: ## For each evaluation pair we will decode and compute scores.
                        slangtlang =dev_pair.strip().split("-")
                        if args.multi_source: ## In case we do multisource NMT
                            slang=slangtlang[0]+"-"+slangtlang[1] ## This will be split in the generate_batches_eval function as we expect a triplet. 
                            tlang=slangtlang[2]
                        else:
                            slang=slangtlang[0]
                            tlang=slangtlang[1]
                        eval_batch_counter = 0
                        for dev_input_ids, dev_input_masks in generate_batches_eval(tok, args, inps[dev_pair], slang):
                            start = time.time()
                            dev_input_ids = dev_input_ids.to(gpu)
                            dev_input_masks = dev_input_masks.to(gpu)
                            if args.is_summarization: ## Things can be slow so best show progress
                                print("Decoding batch from a pool of", len(inps[dev_pair]), "examples")
                            with torch.no_grad(): ## torch.no_grad is apparently known to prevent the code from allocating memory for gradient computation in addition to making things faster. I have not verified this but have kept it as a safety measure to ensure that my model is not being directly tuned on the development set.
                                translations = model.module.generate(dev_input_ids.to(gpu), use_cache=True, num_beams=1, max_length=int(len(dev_input_ids[0])*args.max_decode_length_multiplier), min_length=int(len(dev_input_ids[0])*args.min_decode_length_multiplier), early_stopping=True, attention_mask=dev_input_masks.to(gpu), pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], decoder_start_token_id=tok(["<2"+tlang+">"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], length_penalty=args.length_penalty, repetition_penalty=args.repetition_penalty, encoder_no_repeat_ngram_size=args.encoder_no_repeat_ngram_size, no_repeat_ngram_size=args.no_repeat_ngram_size) ## We translate the batch.
                            dev_input_ids = dev_input_ids.to('cpu')
                            dev_input_masks = dev_input_masks.to('cpu')
                            translations=translations.to('cpu') ## Move to cpu. Not needed but its a safe step.
                            for translation in translations:
                                translation  = tok.decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False) ### Get the raw sentences.
                                hyp[dev_pair].append(translation)
                        if args.use_rouge: ## Get the evaluation metric score.
                            for curr_ref, curr_pred in zip(refs[dev_pair][0], hyp[dev_pair]):
                                score = scorer.score(curr_ref, curr_pred)
                                scores[dev_pair] += score['rougeL'].fmeasure
                            sbleu = scores[dev_pair]/len(hyp[dev_pair])
                            metric = 'Rouge'
                        else:
                            sbleu = get_sacrebleu(refs[dev_pair], hyp[dev_pair])
                            metric = 'BLEU'
                        individual_sbleu_history[dev_pair].append([sbleu, ctr]) ## Update the score history for this pair.
                        sbleus[dev_pair] = sbleu
                        print(metric, "score using sacrebleu after", ctr, "iterations is", sbleu, "for language pair", dev_pair)
                        if sbleu > max_individual_sbleu[dev_pair]: ## Update the best score and step number. If the score has improved then save a model copy for this pair. Although we will stop on the global score (average across scores over all pairs) we save these models if we want a model that performs the best on a single pair.
                            max_individual_sbleu[dev_pair] = sbleu
                            max_individual_sbleu_step[dev_pair] = curr_eval_step
                            print("New peak reached for", dev_pair,". Saving.")
                            torch.save(checkpoint_dict, CHECKPOINT_PATH+".best_dev_bleu."+dev_pair+"."+str(ctr))
                            torch.save(model.module.state_dict(), CHECKPOINT_PATH+".best_dev_bleu."+dev_pair+"."+str(ctr)+".pure_model") ## Pure model without any ddp markers or optimizer info.

                    ## Global stats
                    sbleu = sum(sbleus.values())/len(sbleus) ## The global score.
                    global_sbleu_history.append([sbleu, ctr]) ## Update the global score history.
                    print("Global", metric, "score using sacrebleu after", ctr, "iterations is:", sbleu)
                    if sbleu > max_global_sbleu: ## Update the best score and step number. If this has improved then save a copy for the model. Note that this model MAY NOT be the model that gives the best performance for all pairs.
                        max_global_sbleu = sbleu
                        max_global_sbleu_step = curr_eval_step
                        print("New peak reached. Saving.")
                        torch.save(checkpoint_dict, CHECKPOINT_PATH+".best_dev_bleu.global."+str(ctr))
                        torch.save(model.module.state_dict(), CHECKPOINT_PATH+".best_dev_bleu.global."+str(ctr)+".pure_model") ## Pure model without any ddp markers or optimizer info.
                    if curr_eval_step - max_global_sbleu_step > (args.early_stop_checkpoints + annealing_attempt*args.additional_early_stop_checkpoints_per_anneal_step): ## If the global scores have not improved for more than early_stop_checkpoints + some additional checkpoints to wait for till annealing is done then we stop training.
                        if annealing_attempt < args.max_annealing_attempts: ## We will only downscale the LR a fixed number of times. Each time we downscale the number of checkpoints to wait for declaring convergence will increase by a fixed value.
                            annealing_attempt += 1
                            curr_lr = scheduler.get_lr()[0]
                            print("LR before annealing is:", curr_lr)
                            while scheduler.get_lr()[0] > (curr_lr/args.learning_rate_scaling): ## Currently we down scale the LR by advancing the scheduler by some steps. Now this is a bad idea because the scheduler may reach maximum number of steps where the LR is 0. However the training loop will continue and nothing will be updated. The loophole I have used is to set the maximum number of steps to a large value. Thus far I have not seen a case where this has a bad effect but users who do not trust this part of the code should not use annealing.
                                scheduler.step()
                            print("LR after annealing is:", scheduler.get_lr()[0])

                        else: ## Convergence has been reached and we stop and report the final metrics.
                            print("We have seemingly converged as", metric, "failed to increase for the following number of checkpoints:", args.early_stop_checkpoints+annealing_attempt*args.additional_early_stop_checkpoints_per_anneal_step, ". You may want to consider increasing the number of tolerance steps, doing additional annealing or having a lower peak learning rate or something else.")
                            print("Terminating training")
                            print("Global dev", metric, "history:", global_sbleu_history)
                            print("Individual", metric, "history:", individual_sbleu_history )
                            break
                    curr_eval_step += 1

                    model.train() ## Put the model back in training mode where dropout will be done.

                else: ## If no evaluation will be done then I consider it prudent to save the model every 10000 checkpoints by default. Change this to whatever value you want.
                    if ctr % args.no_eval_save_every == 0:
                        print("No evaluation based early stopping so saving every", args.no_eval_save_every, "checkpoints.")
                        checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
                        torch.save(checkpoint_dict, CHECKPOINT_PATH+"."+str(ctr))
                        torch.save(model.state_dict(), CHECKPOINT_PATH+"."+str(ctr)+".pure_model")
                print("Saving the model")
                sys.stdout.flush()
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
                torch.save(checkpoint_dict, CHECKPOINT_PATH) ## Save a model by default every eval_every steps. This model will be saved with the same file name each time.
                torch.save(model.state_dict(), CHECKPOINT_PATH+".pure_model")
                

            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            dist.barrier()
            # configure map_location properly
            print("Loading from checkpoint")
            sys.stdout.flush()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            checkpoint_dict = torch.load(CHECKPOINT_PATH, map_location=map_location)
            model.load_state_dict(checkpoint_dict['model'])
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            scheduler.load_state_dict(checkpoint_dict['scheduler'])
        
        dist.barrier()
        if args.cross_distillation or args.multi_source: ## The returned input ids and input masks are actually a list of two items each. The first item is to be fed to the parent model and the second item is to be fed to the child model.
            input_ids_parent=input_ids[0]
            input_ids=input_ids[1]
            input_ids_parent = input_ids_parent.to(gpu) ## Move to gpu
            input_masks_parent=input_masks[0]
            input_masks=input_masks[1]
            input_masks_parent = input_masks_parent.to(gpu) ## Move to gpu
        input_ids=input_ids.to(gpu) ## Move to gpu
        input_masks=input_masks.to(gpu) ## Move to gpu
        decoder_input_ids=decoder_input_ids.to(gpu) ## Move to gpu
        labels=labels.to(gpu) ## Move to gpu
        optimizer.zero_grad() ## Empty the gradients before any computation.
        try:
            if args.fp16: ## The difference between AMP and FP32 is the use of the autocast. The code below is duplicated and can be shrunk. TODO.
                with torch.cuda.amp.autocast():
                    mod_compute = model(input_ids=input_ids, attention_mask=input_masks ,decoder_input_ids=decoder_input_ids, output_hidden_states=args.distillation, output_attentions=args.distillation) ## Run the model and get logits.
                    logits = mod_compute.logits
                    lprobs = torch.nn.functional.log_softmax(logits/args.softmax_temperature, dim=-1) ## Softmax tempering of logits if needed.
                    loss = label_smoothed_nll_loss(
                        lprobs, labels, args.label_smoothing, ignore_index=tok.pad_token_id
                    ) ## Label smoothed cross entropy loss.
                    loss = loss*args.softmax_temperature ## Up scale loss in case of non unitary temperatures.
                    ## We will do multilayer softmaxing without any consideration for entropy maximization or distillation.
                    for logits in mod_compute.additional_lm_logits:
                        lprobs = torch.nn.functional.log_softmax(logits/args.softmax_temperature, dim=-1) ## Softmax tempering of logits if needed.
                        loss_extra = label_smoothed_nll_loss(
                            lprobs, labels, args.label_smoothing, ignore_index=tok.pad_token_id
                        ) ## Label smoothed cross entropy loss.
                        loss += loss_extra*args.softmax_temperature ## Up scale loss in case of non unitary temperatures.
                    if args.max_ent_weight != -1: ## This deals with softmax entropy maximization. The logic is that we compute the softmax entropy of the predictions via -(P(Y/X)*log(P(Y/X))). We then add it to the cross entropy loss with a negative sign as we wish to maximize entropy. This should penalize overconfident predictions.
                        assert (args.max_ent_weight >= 0 and args.max_ent_weight <= 1)
                        lprobs = torch.nn.functional.log_softmax(logits, dim=-1) ## No tempering here
                        entropy = -(torch.exp(lprobs)*lprobs).mean()
                        loss = loss*(1-args.max_ent_weight) - entropy*args.max_ent_weight ## Maximize the entropy so a minus is needed. Weigh and add losses as required.
                    if args.distillation: ## Time to distill.
                        if args.cross_distillation: ## The input ids and masks should be replaced with those appropriate for the parent.
                            input_ids = input_ids_parent
                            input_masks = input_masks_parent
                        with torch.no_grad(): ## No gradient to avoid memory allocation.
                            parent_mod_compute = parent_model(input_ids=input_ids, attention_mask=input_masks ,decoder_input_ids=decoder_input_ids, output_hidden_states=args.distillation, output_attentions=args.distillation) ## Get the parent model's computations.
                        distillation_loss = compute_distillation_losses(mod_compute, parent_mod_compute, labels, tok.pad_token_id, args) ## Compute distillation losses.
                        loss = args.distillation_loss_weight*distillation_loss + (1.0 - args.distillation_loss_weight)*loss ## Update the main loss with weighing and adding.
            else:
                mod_compute = model(input_ids=input_ids, attention_mask=input_masks, decoder_input_ids=decoder_input_ids, output_hidden_states=args.distillation, output_attentions=args.distillation) ## Run the model and get logits.
                logits = mod_compute.logits
                lprobs = torch.nn.functional.log_softmax(logits/args.softmax_temperature, dim=-1) ## Softmax tempering of logits if needed.
                loss = label_smoothed_nll_loss(
                    lprobs, labels, args.label_smoothing, ignore_index=tok.pad_token_id
                ) ## Label smoothed cross entropy loss.
                loss = loss*args.softmax_temperature ## Up scale loss in case of non unitary temperatures.
                ## We will do multilayer softmaxing without any consideration for entropy maximization or distillation.
                for logits in mod_compute.additional_lm_logits:
                    lprobs = torch.nn.functional.log_softmax(logits/args.softmax_temperature, dim=-1) ## Softmax tempering of logits if needed.
                    loss_extra = label_smoothed_nll_loss(
                        lprobs, labels, args.label_smoothing, ignore_index=tok.pad_token_id
                    ) ## Label smoothed cross entropy loss.
                    loss += loss_extra*args.softmax_temperature ## Up scale loss in case of non unitary temperatures.
                if args.max_ent_weight != -1: ## This deals with softmax entropy maximization. The logic is that we compute the softmax entropy of the predictions via -(P(Y/X)*log(P(Y/X))). We then add it to the cross entropy loss with a negative sign as we wish to maximize entropy. This should penalize overconfident predictions.
                    assert (args.max_ent_weight >= 0 and args.max_ent_weight <= 1)
                    lprobs = torch.nn.functional.log_softmax(logits, dim=-1) ## No tempering here
                    entropy = -(torch.exp(lprobs)*lprobs).mean()
                    loss = loss*(1-args.max_ent_weight) - entropy*args.max_ent_weight ## Maximize the entropy so a minus is needed. Weigh and add losses as required.
                if args.distillation: ## Time to distill.
                    if args.cross_distillation: ## The input ids and masks should be replaced with those appropriate for the parent.
                        input_ids = input_ids_parent
                        input_masks = input_masks_parent
                    with torch.no_grad(): ## No gradient to avoid memory allocation.
                        parent_mod_compute = parent_model(input_ids=input_ids, attention_mask=input_masks ,decoder_input_ids=decoder_input_ids, output_hidden_states=args.distillation, output_attentions=args.distillation) ## Get the parent model's computations.
                        distillation_loss = compute_distillation_losses(mod_compute, parent_mod_compute, labels, tok.pad_token_id, args) ## Compute distillation losses.
                    loss = args.distillation_loss_weight*distillation_loss + (1.0 - args.distillation_loss_weight)*loss ## Update the main loss with weighing and adding.

                    
        except Exception as e: ## This is a generic net to catch an exception. Should be a bit more sophisticated in the future. TODO.
            print("NAN loss was computed or something messed up")
            print(e)
            sys.stdout.flush()
            break
        input_ids=input_ids.to('cpu') ## Move to CPU. May not be needed but its a safety net.
        input_masks=input_masks.to('cpu') ## Move to CPU. May not be needed but its a safety net.
        decoder_input_ids=decoder_input_ids.to('cpu') ## Move to CPU. May not be needed but its a safety net.
        labels=labels.to('cpu') ## Move to CPU. May not be needed but its a safety net.
        if args.cross_distillation:
            input_ids_parent=input_ids_parent.to('cpu') ## Move to CPU. May not be needed but its a safety net.
            input_masks_parent=input_masks_parent.to('cpu') ## Move to CPU. May not be needed but its a safety net.
        if args.fp16: ## The gradient scaler needs to be invoked with FP16/AMP computation.
            scaler.scale(loss).backward()
        else:
            pass
        if args.fp16: ## With FP16/AMP computation we need to unscale gradients before clipping them. We then optimize and update the scaler.
            if args.max_gradient_clip_value != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else: ## With FP32, we just do regular backpropagation, gradient clipping and then step the optimizer.
            loss.backward()
            if args.max_gradient_clip_value != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_clip_value)
            optimizer.step()
        scheduler.step() ## Advance the scheduler to get to the next value of LR.
        lv = loss.detach().cpu().numpy() ## Detach the loss in order to report it.
        if ctr % 10 == 0 and rank  % 8 == 0: ## Print the current loss every 10 batches but only for the master/prime process.
            print(ctr, lv)
            sys.stdout.flush()
        end = time.time()
        ctr += 1
    
    if rank == 0:
        CHECKPOINT_PATH = args.fine_tuned_model
        print("Saving the model after the final step")
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        print("The best bleu was:", max_global_sbleu)
        print("The corresponding step was:", max_global_sbleu_step*args.eval_every)
        checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
        torch.save(checkpoint_dict, CHECKPOINT_PATH) ## Save one last time.
        torch.save(model.module.state_dict(), CHECKPOINT_PATH+".pure_model") ## Pure model without any ddp markers or optimizer info.

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
    parser.add_argument('--freeze_embeddings', action='store_true', 
                        help='Should freeze embeddings during fine tuning?')
    parser.add_argument('--freeze_encoder', action='store_true', 
                        help='Should we freeze encoder during fine tuning?')
    parser.add_argument('--add_final_layer_norm', action='store_true', 
                        help='Should we add a final layer norm?')
    parser.add_argument('--normalize_before', action='store_true', 
                        help='Should we normalize before doing attention?')
    parser.add_argument('--normalize_embedding', action='store_true', 
                        help='Should we normalize embeddings?')
    parser.add_argument('--scale_embedding', action='store_true', 
                        help='Should we scale embeddings?')
    parser.add_argument('--mnmt', action='store_true', 
                        help='Are we training MNMT models? If so then the datagen will be slightly tweaked. We will also expect that training and development files will be comma separated when passed as arguments. The slang and tlang markers will also be comma separated and will follow the order of these files.')
    parser.add_argument('--encoder_layers', default=6, type=int, help="The value for number of encoder layers")
    parser.add_argument('--decoder_layers', default=6, type=int, help="The value for number of decoder layers")
    parser.add_argument('--label_smoothing', default=0.1, type=float, help="The value for label smoothing")
    parser.add_argument('--weight_decay', default=0.0001, type=float, help="The value for weight decay")
    parser.add_argument('--lr', default=7e-4, type=float, help="The value for the learning rate")
    parser.add_argument('--dropout', default=0.1, type=float, help="The value for embedding dropout")
    parser.add_argument('--attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--data_sampling_temperature', default=5.0, type=float, help="The value for the data sampling temperature")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, 
                        help='To prevent repetition during decoding. 1.0 means no repetition. 1.2 was supposed to be a good value for some settings according to some researchers.')
    parser.add_argument('--no_repeat_ngram_size', default=0, type=int, 
                        help='N-grams of this size will never be repeated in the decoder. Lets play with 2-grams as default.')
    parser.add_argument('--length_penalty', default=1.0, type=float, 
                        help='Set to more than 1.0 for longer sentences.')
    parser.add_argument('--encoder_no_repeat_ngram_size', default=0, type=int, 
                        help='N-gram sizes to be prevented from being copied over from encoder. Lets play with 2-grams as default.')
    parser.add_argument('--encoder_tying_config', default=None, type=str, 
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--decoder_tying_config', default=None, type=str,
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--softmax_temperature', default=1.0, type=float, help="The value for the softmax temperature")
    parser.add_argument('--encoder_attention_heads', default=8, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--decoder_attention_heads', default=8, type=int, help="The value for number of decoder attention heads")
    parser.add_argument('--wait_k', default=-1, type=int, help="The value for k in wait-k snmt. Keep as -1 for non-snmt aka vanilla NMT.")
    parser.add_argument('--decoder_ffn_dim', default=2048, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--encoder_ffn_dim', default=2048, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--d_model', default=512, type=int, help="The value for model hidden size")
    parser.add_argument('--eval_every', default=1000, type=int, help="The number of iterations after which an evaluation must be done. Also saves a checkpoint every these number of steps.")
    parser.add_argument('--no_eval_save_every', default=10000, type=int, help="The number of iterations after which a model must be force saved in case evaluation is not done.")
    parser.add_argument('--max_gradient_clip_value', default=1.0, type=float, help="The max value for gradient norm value")

    parser.add_argument('--pretrained_model', default='', type=str, 
                        help='Path to the pretrained model')
    parser.add_argument('--pretrained_bilingual_model', default='', type=str, 
                        help='Path to the pretrained bilingual model. Use this if you want to continue training a bilingual model.')
    parser.add_argument('-m', '--fine_tuned_model', default='pytorch.bin', type=str, 
                        help='Path to save the fine tuned model')
    parser.add_argument('--warmup_steps', default=16000, type=int,
                        help='Scheduler warmup steps')
    parser.add_argument('--batch_size', default=2048, type=int, 
                        help='Train batch sizes in tokens')
    parser.add_argument('--dev_batch_size', default=1024, type=int, 
                        help='Dev batch sizes in lines')
    parser.add_argument('--max_src_length', default=256, type=int, 
                        help='Maximum token length for source language')
    parser.add_argument('--max_tgt_length', default=256, type=int, 
                        help='Maximum token length for target language')
    parser.add_argument('--early_stop_checkpoints', default=10, type=int, 
                        help='Number of checkpoints to wait to see if BLEU increases.')
    parser.add_argument('--learning_rate_scaling', default=2, type=int, 
                        help='How much should the LR be divided by during annealing?. Set num_batches to a larger value or else you will see lr go to zero too soon.')
    parser.add_argument('--max_annealing_attempts', default=2, type=int, 
                        help='Number of times LR should be annealed.')
    parser.add_argument('--additional_early_stop_checkpoints_per_anneal_step', default=5, type=int, 
                        help='How many additional checkpoints should we wait till declaring convergence? This will be multiplied with the annealing step number.')
    parser.add_argument('--num_batches', default=1000000, type=int, 
                        help='Number of batches to train on')
    parser.add_argument('--max_eval_batches', default=1000, type=int, 
                        help='These many evaluation batches will be considered. Use a small value like 5 to cover a portion of the evaluation data.')
    parser.add_argument('--max_decode_length_multiplier', default=2.0, type=float, 
                        help='This multiplied by the source sentence length will be the maximum decoding length.')
    parser.add_argument('--min_decode_length_multiplier', default=0.1, type=float, 
                        help='This multiplied by the source sentence length will be the minimum decoding length.')
    parser.add_argument('--train_slang', default='en', type=str, 
                        help='Source language(s) for training')
    parser.add_argument('--tokenizer_name_or_path', default='ai4bharat/indic-bert', type=str, 
                        help='Name of or path to the tokenizer')
    parser.add_argument('--pretrained_tokenizer_name_or_path', default=None, type=str, 
                        help='Name of or path to the tokenizer of the pretrained model if its different from the current model. This tokenizer will be used for remapping embeddings so as to reuse as many pretrained embeddings as possible.')
    parser.add_argument('--train_tlang', default='hi', type=str, 
                        help='Target language(s) for training')
    parser.add_argument('--train_src', default='', type=str, 
                        help='Source language training sentences')
    parser.add_argument('--train_tgt', default='', type=str, 
                        help='Target language training sentences')
    parser.add_argument('--dev_slang', default='en', type=str, 
                        help='Source language(s) for training')
    parser.add_argument('--dev_tlang', default='hi', type=str, 
                        help='Target language(s) for training')
    parser.add_argument('--dev_src', default='', type=str, 
                        help='Source language(s) development sentences')
    parser.add_argument('--dev_tgt', default='', type=str, 
                        help='Target language(s) development sentences')
    parser.add_argument('--fp16', action='store_true', 
                        help='Should we use fp16 training?')
    parser.add_argument('--no_eval', action='store_true', 
                        help='Should we skip evaluation?')
    parser.add_argument('--source_masking_for_bilingual', action='store_true', 
                        help='Should we use masking on source sentences when training on parallel corpora?')
    parser.add_argument('--is_summarization', action='store_true', 
                        help='Should we use masking on source sentences when training on parallel corpora?')
    parser.add_argument('--hard_truncate_length', default=0, type=int, 
                        help='Should we perform a hard truncation of the batch? This will be needed to eliminate cuda caching errors for when sequence lengths exceed a particular limit. This means self attention matrices will be massive and I used to get errors. Choose this value empirically.')
    parser.add_argument('--use_rouge', action='store_true', 
                        help='Should we use ROUGE for evaluation?')
    parser.add_argument('--max_ent_weight', type=float, default=-1.0, 
                        help='Should we maximize softmax entropy? If the value is anything between 0 and 1 then yes. If its -1.0 then no maximization will be done.')
    parser.add_argument('--shard_files', action='store_true', 
                        help='Should we shard the training data? Set to true only if the data is not already pre-sharded.')
    parser.add_argument('--multi_source', action='store_true', 
                        help='Are we doing multisource NMT? In that case you should specify the train_src as a hyphen separated pair indicating the parent language and the child language. You should also ensure that the source file is a tab separated file where each line contains "the parent pair source sentence[tab]child pair source sentence".')
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
    ### Distillation flags
    parser.add_argument('--distillation', action='store_true', 
                        help='Should we perform distillation from a parent model? If so then you must specify the model using "parent_pretrained_model". There are several distillation options check the flag called "distillation_styles".')
    parser.add_argument('--cross_distillation', action='store_true', 
                        help='Should we perform cross distillation from a parent model which has been trained on another source language but the same target language? If so then you must specify the model using "parent_pretrained_model". Additionally you should specify the train_src as a hyphen separated pair indicating the parent language and the child language. You should also ensure that the source file is a tab separated file where each line contains "the parent pair source sentence[tab]child pair source sentence" There are several distillation options check the flag called "distillation_styles".')
    parser.add_argument('--parent_pretrained_model', default='', type=str, 
                        help='Path to the parent pretrained model for distillation. The pretrained_model flag will be used to initialize the child model.')
    parser.add_argument('--distillation_loss_weight', type=float, default=0.7, 
                        help='All the distillation losses will be averaged and then multiplied by this weight before adding it to the regular xentropy loss which will be weighted by (1- distillation_loss_weight).')
    parser.add_argument('--distillation_styles', default='cross_entropy', type=str, 
                        help='One or more of softmax_distillation, attention_distillation, hidden_layer_regression. For attention distillation you must make sure that the number of attention heads between the parent and child are the same and for hidden layer regression you must make sure that the hidden size (d_model) is the same for the parent and child. In both these cases, you should also specify the layer mapping. See the "distillation_layer_mapping" flag.')
    parser.add_argument('--distillation_layer_mapping', default='1-1,2-2,3-3,4-4,5-5,6-6', type=str, 
                        help='This indicates the mappings between the parent and child model. The same flag is used for the encoder and the decoder. If you want to map the 2nd parent layer to the first child layer then use 2-1. Note that the layers are not zero indexed as per the description. Ensure that your indices are correct because checking is not done at the moment. If you get weird results then first make sure that your flags are correctly set. If the parent has 6 layers and the child has 3 layers then something like 6-4 will definitely throw an error. User beware! Dokuro mark.')
    parser.add_argument('--parent_encoder_layers', default=6, type=int, help="The value for number of encoder layers")
    parser.add_argument('--parent_decoder_layers', default=6, type=int, help="The value for number of decoder layers")
    parser.add_argument('--parent_dropout', default=0.1, type=float, help="The value for embedding dropout")
    parser.add_argument('--parent_attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--parent_activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--parent_encoder_attention_heads', default=8, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--parent_decoder_attention_heads', default=8, type=int, help="The value for number of decoder attention heads")
    parser.add_argument('--parent_decoder_ffn_dim', default=2048, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--parent_encoder_ffn_dim', default=2048, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--parent_d_model', default=512, type=int, help="The value for model hidden size")
    ###
    args = parser.parse_args()
    print("IP address is", args.ipaddr)
    
    args.world_size = args.gpus * args.nodes                #
    
    train_files = {}
    if args.mnmt:
        slangs = args.train_slang.strip().split(",")
        tlangs = args.train_tlang.strip().split(",")
        train_srcs = args.train_src.strip().split(",")
        train_tgts = args.train_tgt.strip().split(",")
        train_files = {slang+"-"+tlang: (train_src, train_tgt) for slang, tlang, train_src, train_tgt in zip(slangs, tlangs, train_srcs, train_tgts)}
    else:
        train_files = {args.train_slang+"-"+args.train_tlang : (args.train_src, args.train_tgt)}
    print("Training files are:", train_files)
    
    dev_files = {}
    if args.mnmt:
        slangs = args.dev_slang.strip().split(",")
        tlangs = args.dev_tlang.strip().split(",")
        dev_srcs = args.dev_src.strip().split(",")
        dev_tgts = args.dev_tgt.strip().split(",")
        dev_files = {slang+"-"+tlang: (dev_src, dev_tgt) for slang, tlang, dev_src, dev_tgt in zip(slangs, tlangs, dev_srcs, dev_tgts)}
    else:
        dev_files = {args.dev_slang+"-"+args.dev_tlang : (args.dev_src, args.dev_tgt)}
    print("Development files are:", dev_files)
    
    os.environ['MASTER_ADDR'] = args.ipaddr              #
    os.environ['MASTER_PORT'] = args.port                      #
    mp.spawn(model_create_load_run_save, nprocs=args.gpus, args=(args,train_files, dev_files))         #
    
if __name__ == "__main__":
    run_demo()