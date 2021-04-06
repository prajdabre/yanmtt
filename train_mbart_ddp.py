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

## Other imports
import random
import numpy as np
import math
import sacrebleu
##

## Seed setting here
torch.manual_seed(621313)
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

def compute_distillation_losses(child_mod_compute, parent_mod_compute, target, ignore_index, args):
    """Implemented by me. This is based on distill bert, distill mbart etc.
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


def yield_corpus_indefinitely(corpus, lang):
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

def shard_files(files, world_size):
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
        

def generate_batches(tok, num_batches=1000, batch_size=2048, mp_val_or_range=0.3, lamb=3.5, rank=0, temperature=5.0, files=None, max_length=100):
    """Generates the source, target and source attention masks for denoising. Long sequences are truncated and short sequences are ignored."""
    
    batch_count = 0
    language_list = list(files.keys())
    print("Training for:", language_list)
    language_file_dict = {}
    probs = {}
    for l in language_list:
        file_content = open(files[l]+"."+"%02d" % rank).readlines()
        probs[l] = len(file_content)
        language_file_dict[l] = yield_corpus_indefinitely(file_content, l)
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = probs_temp
    probs_temp = {lang: probs[lang]**(1.0/temperature) for lang in probs} ## Temperature sampling probabilities.
    probs = probs_temp
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = [probs_temp[lang] for lang in language_list]
    num_langs = len(language_list)
    language_indices = list(range(num_langs))
    while batch_count != num_batches:
        curr_batch_count = 0
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_label_batch = []
        batch_count += 1
        max_src_sent_len = 0
        max_tgt_sent_len = 0
        start = time.time()
        sents_in_batch = 0
        while True:
            language_idx = random.choices(language_indices, probs)[0]
            sentence = next(language_file_dict[language_list[language_idx]]).strip()
            lang = "<2"+language_list[language_idx]+">"
            if type(mp_val_or_range) is float:
                mask_percent = mp_val_or_range
            else:
                mask_percent = random.uniform(mp_val_or_range[0], mp_val_or_range[1])
            sentence_split = sentence.split(" ")
            sent_len = len(sentence_split)
            if sent_len < 10: 
                continue
            if sent_len > max_length: ## Initial truncation
                sentence_split = sentence_split[:max_length]
                sentence = " ".join(sentence_split)
                sent_len = max_length
            mask_count = 0
            max_mask_count = int(mask_percent*sent_len)
            spans_to_mask = list(np.random.poisson(lamb, 1000))
            curr_sent_len = sent_len
            while mask_count < max_mask_count:
                try:
                    span_to_mask = spans_to_mask[0]
                    del spans_to_mask[0]
                    if span_to_mask > (max_mask_count-mask_count): ## Cant mask more than the allowable number of tokens.
                        continue
                    idx_to_mask = random.randint(0, (curr_sent_len-1)-(span_to_mask-1))
                    if "[MASK]" not in sentence_split[idx_to_mask:idx_to_mask+span_to_mask]:
                        sentence_split[idx_to_mask:idx_to_mask+span_to_mask] = ["[MASK]"]
                        mask_count += span_to_mask
                        curr_sent_len -= (span_to_mask-1)
                except:
                    break ## If we cannot get a properly masked sentence despite all our efforts then we just give up and continue with what we have so far.
            
            masked_sentence = " ".join(sentence_split)
            iids = tok(lang + " " + masked_sentence + " </s>", add_special_tokens=False, return_tensors="pt").input_ids
            curr_src_sent_len = len(iids[0])
            
            iids = tok("<s> " + sentence, add_special_tokens=False, return_tensors="pt").input_ids
            curr_tgt_sent_len = len(iids[0])
            
            if curr_src_sent_len > 256 or curr_tgt_sent_len > 256: ## Hard limit. These are too long after tokenization and we dont want them for now.
                continue
            
            if curr_src_sent_len > max_src_sent_len:
                max_src_sent_len = curr_src_sent_len
            
            if curr_tgt_sent_len > max_tgt_sent_len:
                max_tgt_sent_len = curr_tgt_sent_len
            encoder_input_batch.append(masked_sentence + " </s> " + lang)
            decoder_input_batch.append(lang + " " + sentence)
            decoder_label_batch.append(sentence + " </s>")
            sents_in_batch += 1
            curr_batch_count = max(max_src_sent_len, max_tgt_sent_len)*sents_in_batch
            if curr_batch_count > batch_size:
                break
        input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
        if len(input_ids[0]) > max_length: ## Truncate again if we exceed the maximum sequence length.
            input_ids = input_ids[:,:max_length]
        input_masks = (input_ids != tok.pad_token_id).int()
        decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
        if len(decoder_input_ids[0]) > max_length: ## Truncate again if we exceed the maximum sequence length.
            decoder_input_ids = decoder_input_ids[:,:max_length]
        labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_tgt_sent_len).input_ids
        if len(labels[0]) > max_length: ## Truncate again if we exceed the maximum sequence length.
            labels = labels[:,:max_length]
        end = time.time()
        yield input_ids, input_masks, decoder_input_ids, labels
       
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
        return model
    
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
                del model_to_load_dict[our_model_key]
    return model_to_load_dict

def model_create_load_run_save(gpu, args, files):
    """The main function which does the overall training. Should be split into multiple parts in the future. Currently monolithc intentionally."""
    rank = args.nr * args.gpus + gpu ## The rank of the current process out of the total number of processes indicated by world_size.
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    if args.shard_files and rank == 0: ## First shard the data using process 0 aka the prime process or master process. Other processes will wait.
        shard_files(files, args.world_size)
    
    dist.barrier() ## Stop other processes from proceeding till sharding is done.
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=False, use_fast=False, keep_accents=True) ## Fast tokenizers are not good because their behavior is weird. Accents should be kept or else the segmentation will be messed up on languages with accented characters. No lower case obviously because we want to train on the original case. Set to false if you are ok with the model not dealing with cases.
    print("Tokenizer is:", tok)
    
    print(f"Running DDP checkpoint example on rank {rank}.") ## Unlike the FT script this will always be distributed

    if args.fp16: ## Although the code supports FP16/AMP training, it tends to be unstable in distributed setups so use this carefully.
        print("We will do fp16 training")
        scaler = torch.cuda.amp.GradScaler()
    else:
        print("We will do fp32 training")
    
    if args.encoder_tying_config is not None:
        print("We will use recurrently stacked layers for the encoder with configuration:", args.encoder_tying_config)
    if args.decoder_tying_config is not None:
        print("We will use recurrently stacked layers for the decoder with configuration:", args.decoder_tying_config)
        
    config = MBartConfig(vocab_size=len(tok), encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_dim, decoder_ffn_dim=args.decoder_ffn_dim, d_model=args.d_model, add_final_layer_norm=args.add_final_layer_norm, normalize_before=args.normalize_before, normalize_embedding=args.normalize_embedding, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], static_position_embeddings=True, encoder_tying_config=args.encoder_tying_config, decoder_tying_config=args.decoder_tying_config, multilayer_softmaxing=args.multilayer_softmaxing) ## Configuration. TODO: Save this configuration somehow.
    model = MBartForConditionalGeneration(config)
    torch.cuda.set_device(gpu)

    model.cuda(gpu)
    model.train()
    
    if args.distillation: ## When distilling we need a parent model. The creation of the model is in the same way as the child. This model is immediately loaded with some pretrained params and then loaded into the GPU.
        print("We will do distillation from a parent model.")
        parent_config = MBartConfig(vocab_size=len(tok), encoder_layers=args.parent_encoder_layers, decoder_layers=args.parent_decoder_layers, dropout=args.parent_dropout, attention_dropout=args.parent_attention_dropout, activation_dropout=args.parent_activation_dropout, encoder_attention_heads=args.parent_encoder_attention_heads, decoder_attention_heads=args.parent_decoder_attention_heads, encoder_ffn_dim=args.parent_encoder_ffn_dim, decoder_ffn_dim=args.parent_decoder_ffn_dim, d_model=args.parent_d_model, add_final_layer_norm=args.add_final_layer_norm, normalize_before=args.normalize_before, normalize_embedding=args.normalize_embedding, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], static_position_embeddings=True, encoder_tying_config=args.encoder_tying_config, decoder_tying_config=args.decoder_tying_config)
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
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.iters*args.world_size) ## A warmup and decay scheduler. We use the linear scheduler for now. TODO: Enable other schedulers with a flag.
    while scheduler.get_lr()[0] < 1e-7: ## We want to keep a minimum learning rate else for the initial batch or initial few batches barely anything will be learned which is a waste of computation. This minimum value is kept to 1e-7 by default in accordance with previous literature, other implementations and the Paris peace accords.
        scheduler.step()
    print("Initial LR is:", scheduler.get_lr()[0])
    if args.initialization_model != "": ## Here we load a previous checkpoint in case training crashed.
        print("Loading from checkpoint")
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        sys.stdout.flush()
        checkpoint_dict = torch.load(args.initialization_model, map_location=map_location)
        if type(checkpoint_dict) == dict:
            model.load_state_dict(remap_embeddings_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict['model'], 4, args), args))
            optimizer.load_state_dict(checkpoint_dict['optimizer']) ## Dubious
            scheduler.load_state_dict(checkpoint_dict['scheduler']) ## Dubious
            ctr = checkpoint_dict['ctr']
        else:
            model.load_state_dict(remap_embeddings_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict, 3, args), args))
            ctr = 0
    else:
        ctr = 0
    model.train()
    print("Using label smoothing of", args.label_smoothing)
    print("Using gradient clipping norm of", args.max_gradient_clip_value)
    
    for input_ids, input_masks, decoder_input_ids, labels in generate_batches(tok, args.iters, args.batch_size, (0.30, 0.40), 3.5, rank, args.data_sampling_temperature, files, args.max_length): #Batches are generated from here. The argument (0.30, 0.40) is a range which indicates the percentage of the source sentence to be masked in case we want masking during training just like we did during BART pretraining. The argument 3.5 is the lambda to the poisson length sampler which indicates the average length of a word sequence that will be masked.
        start = time.time()
        optimizer.zero_grad() ## Empty the gradients before any computation.
        
        if ctr % args.eval_every == 0: ## We have to evaluate our model every eval_every steps. Since there is no evaluation data this means our model is saved every eval_every steps.
            CHECKPOINT_PATH = args.model_path
            if rank == 0:
                print("Saving the model")
                sys.stdout.flush()
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
                torch.save(checkpoint_dict, CHECKPOINT_PATH) ## Save a model by default every eval_every steps. This model will be saved with the same file name each time.
                torch.save(model.module.state_dict(), CHECKPOINT_PATH+".pure_model")
                if ctr % args.no_eval_save_every == 0: ## If no evaluation will be done then I consider it prudent to save the model every 10000 checkpoints by default. Change this to whatever value you want.
                    torch.save(checkpoint_dict, CHECKPOINT_PATH + "."+str(ctr)) 
                    torch.save(model.module.state_dict(), CHECKPOINT_PATH+ "."+str(ctr)+".pure_model")

            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            dist.barrier()
            # configure map_location properly
            print("Loading from checkpoint")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            sys.stdout.flush()
            checkpoint_dict = torch.load(CHECKPOINT_PATH, map_location=map_location)
            model.load_state_dict(checkpoint_dict['model'])
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            scheduler.load_state_dict(checkpoint_dict['scheduler'])
            
        input_ids=input_ids.to(gpu) ## Move to gpu
        input_masks=input_masks.to(gpu) ## Move to gpu
        decoder_input_ids=decoder_input_ids.to(gpu) ## Move to gpu
        labels=labels.to(gpu) ## Move to gpu
        try:
            if args.fp16: ## The difference between AMP and FP32 is the use of the autocast. The code below is duplicated and can be shrunk. TODO.
                with torch.cuda.amp.autocast():
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
                        with torch.no_grad(): ## No gradient to avoid memory allocation.
                            parent_mod_compute = parent_model(input_ids=input_ids, attention_mask=input_masks ,decoder_input_ids=decoder_input_ids, output_hidden_states=args.distillation, output_attentions=args.distillation)
                        distillation_loss = compute_distillation_losses(mod_compute, parent_mod_compute, labels, tok.pad_token_id, args) ## Get the parent model's computations.
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
        if ctr % 10 == 0 and rank % 8 == 0: ## Print the current loss every 10 batches but only for the master/prime process.
            print(ctr, lv)
            sys.stdout.flush()
        end = time.time()
        ctr += 1
    
    if rank == 0:
        checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
        torch.save(checkpoint_dict, CHECKPOINT_PATH) ## Save one last time.
        torch.save(model.module.state_dict(), CHECKPOINT_PATH+".pure_model") ## We will distribute this model and/or use it for fine tuning.

    dist.destroy_process_group()

def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=8, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-i', '--iters', default=2000000, type=int, 
                        metavar='N',
                        help='number of total iterations to run')
    parser.add_argument('-a', '--ipaddr', default='localhost', type=str, 
                        help='IP address of the main node')
    parser.add_argument('-m', '--model_path', default='ddpdefault', type=str, 
                        help='Name of the model')
    parser.add_argument('--initialization_model', default='', type=str, 
                        help='Name of the model')
    parser.add_argument('--langs', default='', type=str, 
                        help='Comma separated string of source languages')
    parser.add_argument('--file_prefixes', default='', type=str, 
                        help='Comma separated string of source language file prefixes. Make sure that these are split into N groups where N is the number of GPUs you plan to use.')
    parser.add_argument('--add_final_layer_norm', action='store_true', 
                        help='Should we add a final layer norm?')
    parser.add_argument('--normalize_before', action='store_true', 
                        help='Should we normalize before doing attention?')
    parser.add_argument('--normalize_embedding', action='store_true', 
                        help='Should we normalize embeddings?')
    parser.add_argument('--scale_embedding', action='store_true', 
                        help='Should we scale embeddings?')
    parser.add_argument('--tokenizer_name_or_path', default='ai4bharat/indic-bert', type=str, 
                        help='Name of or path to the tokenizer')
    parser.add_argument('--pretrained_tokenizer_name_or_path', default=None, type=str, 
                        help='Name of or path to the tokenizer of the pretrained model if its different from the current model. This tokenizer will be used for remapping embeddings so as to reuse as many pretrained embeddings as possible.')
    parser.add_argument('--warmup_steps', default=16000, type=int,
                        help='Scheduler warmup steps')
    parser.add_argument('--encoder_layers', default=6, type=int, help="The value for number of encoder layers")
    parser.add_argument('--decoder_layers', default=6, type=int, help="The value for number of decoder layers")
    parser.add_argument('--max_length', default=100, type=int, 
                        help='Maximum sequence length for training')
    parser.add_argument('--batch_size', default=2048, type=int, 
                        help='Maximum number of tokens in batch')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help="The value for label smoothing.")
    parser.add_argument('--lr', default=1e-3, type=float, help="The value for the learning rate")
    parser.add_argument('--weight_decay', default=0.00001, type=float, help="The value for weight decay")
    parser.add_argument('--dropout', default=0.1, type=float, help="The value for embedding dropout")
    parser.add_argument('--attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--encoder_attention_heads', default=16, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--decoder_attention_heads', default=16, type=int, help="The value for number of decoder attention heads")
    
    parser.add_argument('--decoder_ffn_dim', default=4096, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--encoder_ffn_dim', default=4096, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--d_model', default=1024, type=int, help="The value for model hidden size")
    parser.add_argument('--data_sampling_temperature', default=5.0, type=float, help="The value for data sampling temperature")
    parser.add_argument('--max_gradient_clip_value', default=1.0, type=float, help="The max value for gradient norm")
    parser.add_argument('--softmax_temperature', default=1.0, type=float, help="The value for the softmax temperature")
    parser.add_argument('--max_ent_weight', type=float, default=-1.0, 
                        help='Should we maximize softmax entropy? If the value is anything between 0 and 1 then yes. If its -1.0 then no maximization will be done.')
    parser.add_argument('--fp16', action='store_true', 
                        help='Should we use fp16 training?')
    parser.add_argument('--encoder_tying_config', default=None, type=str,
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--decoder_tying_config', default=None, type=str,
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--shard_files', action='store_true', 
                        help='Should we shard the training data? Set to true only if the data is not already pre-sharded.')
    parser.add_argument('--multilayer_softmaxing', action='store_true', 
                        help='Should we apply a softmax for each decoder layer? Unsupported for distillation. Only for vanilla training.')
    parser.add_argument('--remap_encoder', default='', type=str, 
                        help='This indicates the remappings for the layer. Example: 1-2,2-4,3-6. The plan is to use these remappings to cut down the model prior to decoding or training. Suppose we have a 6 layer model but we only want to utilize the 2nd, 4th and 6th layer then we will copy the content of the 2nd, 4th and 6th layers to the 1st, 2nd and 3rd layer and delete the former layers from the parameter dictionary. This counts as layer pruning.')
    parser.add_argument('--remap_decoder', default='', type=str, 
                        help='This indicates the remappings for the layer. Example: 1-2,2-4,3-6. The plan is to use these remappings to cut down the model prior to decoding or training. Suppose we have a 6 layer model but we only want to utilize the 2nd, 4th and 6th layer then we will copy the content of the 2nd, 4th and 6th layers to the 1st, 2nd and 3rd layer and delete the former layers from the parameter dictionary. This counts as layer pruning.')
    ### Distillation flags
    parser.add_argument('--distillation', action='store_true', 
                        help='Should we perform distillation from a parent model? If so then you must specify the model using "parent_pretrained_model". There are several distillation options check the flag called "distillation_styles".')
    parser.add_argument('--parent_pretrained_model', default='', type=str, 
                        help='Path to the parent pretrained model for distillation. The pretrained_model flag will be used to initialize the child model.')
    parser.add_argument('--distillation_loss_weight', type=float, default=0.7, 
                        help='All the distillation losses will be averaged and then multiplied by this weight before adding it to the regular xentropy loss which will be weighted by (1- distillation_loss_weight).')
    parser.add_argument('--distillation_styles', default='cross_entropy', type=str, 
                        help='One or more of softmax_distillation, attention_distillation, hidden_layer_regression. For attention distillation you must make sure that the number of attention heads between the parent and child are the same and for hidden layer regression you must make sure that the hidden size (d_model) is the same for the parent and child. In both these cases, you should also specify the layer mapping. See the "distillation_layer_mapping" flag.')
    parser.add_argument('--distillation_layer_mapping', default='1-1,2-2,3-3,4-4,5-5,6-6', type=str, 
                        help='This indicates the mappings between the parent and child model. The same flag is used for the encoder and the decoder. If you want to map the 2nd parent layer to the first child layer then use 2-1. Note that the layers are not zero indexed as per the description. Ensure that your indices are correct because checking is not done at the moment. If you get weird results then first make sure that your flags are correctly set. If the parent has 6 layers and the child has 3 layers then something like 6-4 will definitely throw an error. User beware! Dokuro mark.')
    parser.add_argument('--eval_every', default=1000, type=int, help="The number of iterations after which an evaluation must be done. Also saves a checkpoint every these number of steps.")
    parser.add_argument('--no_eval_save_every', default=10000, type=int, help="The number of iterations after which a model must be force saved in case evaluation is not done.")
    parser.add_argument('--parent_encoder_layers', default=3, type=int, help="The value for number of encoder layers")
    parser.add_argument('--parent_decoder_layers', default=3, type=int, help="The value for number of decoder layers")
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

    files = {lang: file_prefix for lang, file_prefix in zip(args.langs.strip().split(","), args.file_prefixes.strip().split(","))}
    print("All files:", files)

    os.environ['MASTER_ADDR'] = args.ipaddr              #
    os.environ['MASTER_PORT'] = '26023'                      #
    mp.spawn(model_create_load_run_save, nprocs=args.gpus, args=(args,files,))         #
    
if __name__ == "__main__":
    run_demo()