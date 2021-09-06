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
from transformers import MBartForConditionalGeneration, BartForConditionalGeneration, MBartConfig, get_linear_schedule_with_warmup
from transformers import AdamW
##


## Pytorch imports
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam
from torch.nn.functional import cosine_similarity
from torch.utils.tensorboard import SummaryWriter
##

## Our imports
from common_utils import *
##

## Other imports
import random
import numpy as np
import math
import sacrebleu
import functools
##

## Seed setting here
torch.manual_seed(621313)
##


def model_create_load_run_save(gpu, args, files, train_files):
    """The main function which does the overall training. Should be split into multiple parts in the future. Currently monolithc intentionally."""
    rank = args.nr * args.gpus + gpu ## The rank of the current process out of the total number of processes indicated by world_size.
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    if args.shard_files and rank == 0: ## First shard the data using process 0 aka the prime process or master process. Other processes will wait.
        shard_files_mono(files, args)
        shard_files_bi(train_files, args)
    
    dist.barrier() ## Stop other processes from proceeding till sharding is done.
    
    if args.use_official_pretrained:
        if "mbart" in args.model_path:
            if "50" in args.model_path:
                tok = MBart50Tokenizer.from_pretrained(args.tokenizer_name_or_path)
            else:
                tok = MBartTokenizer.from_pretrained(args.tokenizer_name_or_path)
        else:
            tok = BartTokenizer.from_pretrained(args.tokenizer_name_or_path)
    else:
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
    
    if args.unidirectional_encoder:
        print("Using unidirectional encoder.")
    
    writer = SummaryWriter(args.model_path+".tflogs")
    
    if args.use_official_pretrained:
        if "mbart" in args.pretrained_model:
            model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model) ## We may use FBs official model and fine-tune it for our purposes.
        elif "bart" in args.pretrained_model:
            model = BartForConditionalGeneration.from_pretrained(args.pretrained_model) ## We may use FBs official model and fine-tune it for our purposes.
        model.config.dropout = args.dropout ## We should set dropouts manually
        model.attention_dropout = args.attention_dropout ## We should set dropouts manually
        model.activation_dropout = args.activation_dropout ## We should set dropouts manually
    else:
        config = MBartConfig(vocab_size=len(tok), encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_dim, decoder_ffn_dim=args.decoder_ffn_dim, d_model=args.d_model, no_embed_norm=args.no_embed_norm, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"], add_special_tokens=False).input_ids[0][0], bos_token_id=tok(["<s>"], add_special_tokens=False).input_ids[0][0], encoder_tying_config=args.encoder_tying_config, decoder_tying_config=args.decoder_tying_config, multilayer_softmaxing=args.multilayer_softmaxing, wait_k=args.wait_k, unidirectional_encoder=args.unidirectional_encoder, softmax_temperature=args.softmax_temperature, temperature_calibration=args.temperature_calibration, layerdrop=args.layerdrop, no_scale_attention_embedding=args.no_scale_attention_embedding, positional_encodings=args.positional_encodings, num_domains_for_domain_classifier=args.num_domains_for_domain_classifier, gradient_reversal_for_domain_classifier=args.gradient_reversal_for_domain_classifier) ## Configuration. TODO: Save this configuration somehow.
        model = MBartForConditionalGeneration(config)
    torch.cuda.set_device(gpu)

    model.cuda(gpu)
    model.train()
    
    if args.distillation: ## When distilling we need a parent model. The creation of the model is in the same way as the child. This model is immediately loaded with some pretrained params and then loaded into the GPU.
        print("We will do distillation from a parent model.")
        if args.use_official_parent_pretrained:
            if "mbart" in args.parent_pretrained_model:
                parent_model = MBartForConditionalGeneration.from_pretrained(args.parent_pretrained_model) ## We may use FBs official model and fine-tune it for our purposes.
            elif "bart" in args.parent_pretrained_model:
                parent_model = BartForConditionalGeneration.from_pretrained(args.parent_pretrained_model) ## We may use FBs official model and fine-tune it for our purposes.
            parent_model.config.dropout = args.dropout ## We should set dropouts manually
            parent_model.attention_dropout = args.attention_dropout ## We should set dropouts manually
            parent_model.activation_dropout = args.activation_dropout ## We should set dropouts manually
        else:
            parent_config = MBartConfig(vocab_size=len(tok), encoder_layers=args.parent_encoder_layers, decoder_layers=args.parent_decoder_layers, dropout=args.parent_dropout, attention_dropout=args.parent_attention_dropout, activation_dropout=args.parent_activation_dropout, encoder_attention_heads=args.parent_encoder_attention_heads, decoder_attention_heads=args.parent_decoder_attention_heads, encoder_ffn_dim=args.parent_encoder_ffn_dim, decoder_ffn_dim=args.parent_decoder_ffn_dim, d_model=args.parent_d_model, no_embed_norm=args.no_embed_norm, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"], add_special_tokens=False).input_ids[0][0], bos_token_id=tok(["<s>"], add_special_tokens=False).input_ids[0][0], encoder_tying_config=args.encoder_tying_config, decoder_tying_config=args.decoder_tying_config, multilayer_softmaxing=args.multilayer_softmaxing, wait_k=args.wait_k, unidirectional_encoder=args.unidirectional_encoder, softmax_temperature=args.softmax_temperature, temperature_calibration=args.temperature_calibration, layerdrop=args.layerdrop, no_scale_attention_embedding=args.no_scale_attention_embedding, positional_encodings=args.positional_encodings)
            parent_model = MBartForConditionalGeneration(config)
        parent_model.cuda(gpu)
        parent_model.train() ## We do this to enable dropout but we wont have an optimizer for this so we wont train this model. For now. Future implementations should ask if we want to do co-distill or not. By co-distillation I mean, the parent will learn together with the child.
        parent_model = DistributedDataParallel(parent_model, device_ids=[gpu], output_device=gpu)
        print("Loading a parent model from which distillation will be done.")
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        if not args.use_official_parent_pretrained:
            parent_checkpoint_dict = torch.load(args.parent_pretrained_model, map_location=map_location)
            if type(parent_checkpoint_dict) == dict:
                parent_model.load_state_dict(parent_checkpoint_dict['model']) # We never do any remapping of the parent. We always reuse it as it is.
            else:
                parent_model.module.load_state_dict(parent_checkpoint_dict) # We never do any remapping of the parent. We always reuse it as it is.

    model = DistributedDataParallel(model, device_ids=[gpu], output_device=gpu) ## This wrapper around the model will enable distributed training.
    
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
    
    model.train()
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.num_batches) ## A warmup and decay scheduler. We use the linear scheduler for now. TODO: Enable other schedulers with a flag.
    while scheduler.get_lr()[0] < 1e-7: ## We want to keep a minimum learning rate else for the initial batch or initial few batches barely anything will be learned which is a waste of computation. This minimum value is kept to 1e-7 by default in accordance with previous literature, other implementations and the Paris peace accords.
        scheduler.step()
    print("Initial LR is:", scheduler.get_lr()[0])
    if args.pretrained_model != "" and not args.use_official_pretrained: ## Here we load a previous checkpoint in case training crashed.
        print("Loading from checkpoint. Strict loading by default but if there are missing or non matching keys, they will be ignored when layer remapping or component selection is done.")
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        sys.stdout.flush()
        checkpoint_dict = torch.load(args.pretrained_model, map_location=map_location)
        if type(checkpoint_dict) == dict:
            model.load_state_dict(remap_embeddings_eliminate_components_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict['model'], 4, args), args), strict=True if (args.remap_encoder == "" and args.remap_decoder == "" and not args.eliminate_encoder_before_initialization and not args.eliminate_decoder_before_initialization and not args.eliminate_embeddings_before_initialization) else False)
            if not args.no_reload_optimizer_ctr_and_scheduler and args.remap_encoder is '' and args.remap_decoder is '' and not args.eliminate_encoder_before_initialization and not args.eliminate_decoder_before_initialization and not args.eliminate_embeddings_before_initialization: ## Do not load optimizers, ctr and schedulers when remapping or resuming training.
                if 'optimizer' in checkpoint_dict:
                    print("Reloading optimizer")
                    optimizer.load_state_dict(checkpoint_dict['optimizer']) ## Dubious
                if 'scheduler' in checkpoint_dict:
                    print("Reloading scheduler")
                    scheduler.load_state_dict(checkpoint_dict['scheduler']) ## Dubious
                if 'ctr' in checkpoint_dict:
                    print("Reloading ctr. This means we resume training.")
                    ctr = checkpoint_dict['ctr']
            else:
                ctr = 0
        else:
            model.module.load_state_dict(remap_embeddings_eliminate_components_and_eliminate_mismatches(model.state_dict(), remap_layers(checkpoint_dict, 3, args), args), strict=True if (args.remap_encoder == "" and args.remap_decoder == "" and not args.eliminate_encoder_before_initialization and not args.eliminate_decoder_before_initialization and not args.eliminate_embeddings_before_initialization) else False)
            ctr = 0
    else:
        print("Training from scratch")
        CHECKPOINT_PATH = args.model_path
        if rank == 0:
            checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': 0}
            torch.save(checkpoint_dict, CHECKPOINT_PATH) ## Save a model by default every eval_every steps. This model will be saved with the same file name each time.
            torch.save(model.state_dict(), CHECKPOINT_PATH+".pure_model")
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        checkpoint_dict = torch.load(CHECKPOINT_PATH, map_location=map_location)
        model.load_state_dict(checkpoint_dict['model'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        ctr = checkpoint_dict['ctr']

    model.train()
    print("Using label smoothing of", args.label_smoothing)
    print("Using gradient clipping norm of", args.max_gradient_clip_value)
    print("Using softmax temperature of", args.softmax_temperature)
    if args.max_ent_weight != -1:
        print("Doing entropy maximization during loss computation.")
    if args.multistep_optimizer_steps > 1:
        print("Using a multistep optimizer where gradients will be accumulated over", args.multistep_optimizer_steps, "batches.")
    num_batches_this_optimizer_step = 0
    losses = 0
    
    for input_ids, input_masks, decoder_input_ids, labels in generate_batches_monolingual_masked_or_bilingual(tok, args, rank, files, train_files, ctr): #Batches are generated from here. The argument (0.30, 0.40) is a range which indicates the percentage of the source sentence to be masked in case we want masking during training just like we did during BART pretraining. The argument 3.5 is the lambda to the poisson length sampler which indicates the average length of a word sequence that will be masked. Since this is pretraining we do not do any evaluations even if we train on parallel corpora.
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
                    if args.save_intermediate_checkpoints:
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
            
        if args.num_domains_for_domain_classifier > 1: ## The label will contain the label as well as the domain indicator
            domain_classifier_labels=labels[1] ## This is not a tensor yet
#             print(domain_classifier_labels)
            domain_classifier_labels = torch.tensor(domain_classifier_labels, dtype=torch.int64).to(gpu) ## Move to gpu
            labels=labels[0]
            label_mask = labels.eq(tok.pad_token_id).unsqueeze(-1).to(gpu)
        input_ids=input_ids.to(gpu) ## Move to gpu
        input_masks=input_masks.to(gpu) ## Move to gpu
        decoder_input_ids=decoder_input_ids.to(gpu) ## Move to gpu
        labels=labels.to(gpu) ## Move to gpu
        
        if args.mixed_wait_k:
            model.module.config.wait_k = random.randint(1, args.wait_k)

        
        if args.fp16: ## The difference between AMP and FP32 is the use of the autocast. The code below is duplicated and can be shrunk. TODO.
            with torch.cuda.amp.autocast():
                if args.bilingual_train_frequency != -1 and ctr % args.bilingual_train_frequency == 0 and args.unify_encoder:
                    source_hidden_state_encoder = model.module.get_encoder()(input_ids=input_ids, attention_mask=input_masks).last_hidden_state ## Run the encoder for source sentence.
                    decoder_input_masks = (decoder_input_ids != tok.pad_token_id).int().to(gpu)
                    target_hidden_state_encoder = model.module.get_encoder()(input_ids=decoder_input_ids, attention_mask=decoder_input_masks).last_hidden_state ## Run the encoder for source sentence.
                    decoder_input_masks.to('cpu') ## Move to CPU. May not be needed but its a safety net. 
                    pad_mask = input_ids.eq(tok.pad_token_id).unsqueeze(2)
                    source_hidden_state_encoder.masked_fill_(pad_mask, 0.0)
                    source_hidden_state_encoder = source_hidden_state_encoder.mean(dim=1)
                    pad_mask = decoder_input_ids.eq(tok.pad_token_id).unsqueeze(2)
                    target_hidden_state_encoder.masked_fill_(pad_mask, 0.0)
                    target_hidden_state_encoder = target_hidden_state_encoder.mean(dim=1)
                    loss = -cosine_similarity(source_hidden_state_encoder, target_hidden_state_encoder)
                    if rank == 0:
                        writer.add_scalar("encoder unification loss", loss.detach().cpu().numpy(), ctr)
                else:
                    mod_compute = model(input_ids=input_ids, attention_mask=input_masks, decoder_input_ids=decoder_input_ids, output_hidden_states=args.distillation, output_attentions=args.distillation, label_mask=label_mask if args.num_domains_for_domain_classifier > 1 else None) ## Run the model and get logits.
                    logits = mod_compute.logits
                    lprobs = torch.nn.functional.log_softmax(logits, dim=-1) ## Softmax tempering of logits if needed.
                    loss = label_smoothed_nll_loss(
                        lprobs, labels, args.label_smoothing, ignore_index=tok.pad_token_id
                    ) ## Label smoothed cross entropy loss.
                    loss = loss*args.softmax_temperature ## Up scale loss in case of non unitary temperatures. Note that in case of self calibrating temperature, the softmax temperature must be set to 1.
                    if rank == 0:
                        writer.add_scalar("pure cross entropy loss", loss.detach().cpu().numpy(), ctr)
                    if args.temperature_calibration: 
                        loss = loss*mod_compute.softmax_temperature
                        if rank == 0:
                            writer.add_scalar("calibrated temperature", mod_compute.softmax_temperature.detach().cpu().numpy(), ctr)
                            writer.add_scalar("calibrated temperature loss", loss.detach().cpu().numpy(), ctr)
                    if args.num_domains_for_domain_classifier > 1: ## We augment the main loss with the domain classifier loss
                        domain_classifier_logits = mod_compute.domain_classifier_logits
                        domain_classifier_lprobs = torch.nn.functional.log_softmax(domain_classifier_logits, dim=-1) ## Softmax tempering of logits if needed.
                        domain_classifier_loss = label_smoothed_nll_loss(
                            domain_classifier_lprobs.view(-1,args.num_domains_for_domain_classifier), domain_classifier_labels.view(-1,1), args.label_smoothing
                        ) ## Label smoothed cross entropy loss. We are not going to do any temperature related stuff to this.
                        loss = domain_classifier_loss*args.domain_classifier_loss_weight + loss * (1.0-args.domain_classifier_loss_weight)
                        if rank == 0:
                            writer.add_scalar("domain classifier loss", domain_classifier_loss.detach().cpu().numpy(), ctr)
                            writer.add_scalar("loss with domain classifier loss", loss.detach().cpu().numpy(), ctr)
                    ## We will do multilayer softmaxing without any consideration for distillation or domain classification.
                    if mod_compute.additional_lm_logits is not None:
                        for additional_logits in mod_compute.additional_lm_logits:
                            lprobs = torch.nn.functional.log_softmax(additional_logits, dim=-1) ## Softmax tempering of logits if needed.
                            loss_extra = label_smoothed_nll_loss(
                                lprobs, labels, args.label_smoothing, ignore_index=tok.pad_token_id
                            ) ## Label smoothed cross entropy loss.
                            loss_extra = loss_extra*args.softmax_temperature ## Up scale loss in case of non unitary temperatures. Note that in case of self calibrating temperature, the softmax temperature must be set to 1. TODO: Perhaps log this too.
                            if args.temperature_calibration: 
                                loss_extra = loss_extra*mod_compute.softmax_temperature
                            loss += loss_extra ## Up scale loss in case of non unitary temperatures. TODO: Perhaps log this too.
                    if args.max_ent_weight != -1: ## This deals with softmax entropy maximization. The logic is that we compute the softmax entropy of the predictions via -(P(Y/X)*log(P(Y/X))). We then add it to the cross entropy loss with a negative sign as we wish to maximize entropy. This should penalize overconfident predictions. 
                        assert (args.max_ent_weight >= 0 and args.max_ent_weight <= 1)
                        logits = logits*args.softmax_temperature ## We have to undo the tempered logits else our entropy estimate will be wrong.
                        if args.temperature_calibration: 
                            logits = logits*mod_compute.softmax_temperature
                        lprobs = torch.nn.functional.log_softmax(logits, dim=-1) ## No tempering here
                        entropy = -(torch.exp(lprobs)*lprobs).mean()
                        if rank == 0:
                            writer.add_scalar("softmax entropy", entropy.detach().cpu().numpy(), ctr)
                        if mod_compute.additional_lm_logits is not None:
                            for additional_logits in mod_compute.additional_lm_logits: ## Compute entropy for each layer as well
                                additional_logits = additional_logits*args.softmax_temperature ## We have to undo the tempered logits else our entropy estimate will be wrong.
                                if args.temperature_calibration: 
                                    additional_logits = additional_logits*mod_compute.softmax_temperature
                                lprobs = torch.nn.functional.log_softmax(additional_logits, dim=-1) ## No tempering here
                                entropy_extra = -(torch.exp(lprobs)*lprobs).mean()
                                entropy += entropy_extra
                        loss = loss*(1-args.max_ent_weight) - entropy*args.max_ent_weight ## Maximize the entropy so a minus is needed. Weigh and add losses as required.
                        if rank == 0:
                            writer.add_scalar("loss with entropy loss", loss.detach().cpu().numpy(), ctr)
                    if args.distillation: ## Time to distill.
                        with torch.no_grad(): ## No gradient to avoid memory allocation.
                            parent_mod_compute = parent_model(input_ids=input_ids, attention_mask=input_masks ,decoder_input_ids=decoder_input_ids, output_hidden_states=args.distillation, output_attentions=args.distillation)
                        distillation_loss = compute_distillation_losses(mod_compute, parent_mod_compute, labels, tok.pad_token_id, args) ## Get the parent model's computations.
                        loss = args.distillation_loss_weight*distillation_loss + (1.0 - args.distillation_loss_weight)*loss ## Update the main loss with weighing and adding.
                        if rank == 0:
                            writer.add_scalar("distillation loss", distillation_loss.detach().cpu().numpy(), ctr)
                            writer.add_scalar("final loss", loss.detach().cpu().numpy(), ctr)
        else:
            if args.bilingual_train_frequency != -1 and ctr % args.bilingual_train_frequency == 0 and args.unify_encoder:
                source_hidden_state_encoder = model.module.get_encoder()(input_ids=input_ids, attention_mask=input_masks).last_hidden_state ## Run the encoder for source sentence.
                decoder_input_masks = (decoder_input_ids != tok.pad_token_id).int().to(gpu)
                target_hidden_state_encoder = model.module.get_encoder()(input_ids=decoder_input_ids, attention_mask=decoder_input_masks).last_hidden_state ## Run the encoder for source sentence.
                decoder_input_masks.to('cpu') ## Move to CPU. May not be needed but its a safety net. 
                pad_mask = input_ids.eq(tok.pad_token_id).unsqueeze(2)
                source_hidden_state_encoder.masked_fill_(pad_mask, 0.0)
                source_hidden_state_encoder = source_hidden_state_encoder.mean(dim=1)
                pad_mask = decoder_input_ids.eq(tok.pad_token_id).unsqueeze(2)
                target_hidden_state_encoder.masked_fill_(pad_mask, 0.0)
                target_hidden_state_encoder = target_hidden_state_encoder.mean(dim=1)
                loss = -cosine_similarity(source_hidden_state_encoder, target_hidden_state_encoder)
                if rank == 0:
                    writer.add_scalar("encoder unification loss", loss.detach().cpu().numpy(), ctr)
            else:
                mod_compute = model(input_ids=input_ids, attention_mask=input_masks, decoder_input_ids=decoder_input_ids, output_hidden_states=args.distillation, output_attentions=args.distillation, label_mask=label_mask if args.num_domains_for_domain_classifier > 1 else None) ## Run the model and get logits.
                logits = mod_compute.logits
                lprobs = torch.nn.functional.log_softmax(logits, dim=-1) ## Softmax tempering of logits if needed.
                loss = label_smoothed_nll_loss(
                    lprobs, labels, args.label_smoothing, ignore_index=tok.pad_token_id
                ) ## Label smoothed cross entropy loss.
                loss = loss*args.softmax_temperature ## Up scale loss in case of non unitary temperatures.
                if rank == 0:
                    writer.add_scalar("pure cross entropy loss", loss.detach().cpu().numpy(), ctr)
                if args.temperature_calibration: 
                    loss = loss*mod_compute.softmax_temperature
                    if rank == 0:
                        writer.add_scalar("calibrated temperature", mod_compute.softmax_temperature.detach().cpu().numpy(), ctr)
                        writer.add_scalar("calibrated temperature loss", loss.detach().cpu().numpy(), ctr)
                if args.num_domains_for_domain_classifier > 1: ## We augment the main loss with the domain classifier loss
                    domain_classifier_logits = mod_compute.domain_classifier_logits
                    domain_classifier_lprobs = torch.nn.functional.log_softmax(domain_classifier_logits, dim=-1) ## Softmax tempering of logits if needed.
#                     print(domain_classifier_labels, domain_classifier_labels.size(), domain_classifier_lprobs, domain_classifier_lprobs.size(), labels, labels.size())
                    domain_classifier_loss = label_smoothed_nll_loss(
                        domain_classifier_lprobs.view(-1,args.num_domains_for_domain_classifier), domain_classifier_labels.view(-1,1), args.label_smoothing
                    ) ## Label smoothed cross entropy loss. We are not going to do any temperature related stuff to this.
                    loss = domain_classifier_loss*args.domain_classifier_loss_weight + loss * (1.0-args.domain_classifier_loss_weight)
                    if rank == 0:
                        writer.add_scalar("domain classifier loss", domain_classifier_loss.detach().cpu().numpy(), ctr)
                        writer.add_scalar("loss with domain classifier loss", loss.detach().cpu().numpy(), ctr)
                ## We will do multilayer softmaxing without any consideration for entropy maximization or distillation.
                if mod_compute.additional_lm_logits is not None:
                    for additional_logits in mod_compute.additional_lm_logits:
                        lprobs = torch.nn.functional.log_softmax(additional_logits, dim=-1) ## Softmax tempering of logits if needed.
                        loss_extra = label_smoothed_nll_loss(
                            lprobs, labels, args.label_smoothing, ignore_index=tok.pad_token_id
                        ) ## Label smoothed cross entropy loss.
                        loss_extra = loss_extra*args.softmax_temperature ## Up scale loss in case of non unitary temperatures. Note that in case of self calibrating temperature, the softmax temperature must be set to 1. TODO: Perhaps log this too.
                        if args.temperature_calibration: 
                            loss_extra = loss_extra*mod_compute.softmax_temperature
                        loss += loss_extra*args.softmax_temperature ## Up scale loss in case of non unitary temperatures. Note that in case of self calibrating temperature, the softmax temperature must be set to 1. TODO: Perhaps log this too.
                if args.max_ent_weight != -1: ## This deals with softmax entropy maximization. The logic is that we compute the softmax entropy of the predictions via -(P(Y/X)*log(P(Y/X))). We then add it to the cross entropy loss with a negative sign as we wish to maximize entropy. This should penalize overconfident predictions. 
                    assert (args.max_ent_weight >= 0 and args.max_ent_weight <= 1)
                    logits = logits*args.softmax_temperature ## We have to undo the tempered logits else our entropy estimate will be wrong.
                    if args.temperature_calibration: 
                        logits = logits*mod_compute.softmax_temperature
                    lprobs = torch.nn.functional.log_softmax(logits, dim=-1) ## No tempering here
                    entropy = -(torch.exp(lprobs)*lprobs).mean()
                    if rank == 0:
                        writer.add_scalar("softmax entropy", entropy.detach().cpu().numpy(), ctr)
                    if mod_compute.additional_lm_logits is not None:
                        for additional_logits in mod_compute.additional_lm_logits: ## Compute entropy for each layer as well
                            additional_logits = additional_logits*args.softmax_temperature ## We have to undo the tempered logits else our entropy estimate will be wrong.
                            if args.temperature_calibration: 
                                additional_logits = additional_logits*mod_compute.softmax_temperature
                            lprobs = torch.nn.functional.log_softmax(additional_logits, dim=-1) ## No tempering here
                            entropy_extra = -(torch.exp(lprobs)*lprobs).mean()
                            entropy += entropy_extra
                    loss = loss*(1-args.max_ent_weight) - entropy*args.max_ent_weight ## Maximize the entropy so a minus is needed. Weigh and add losses as required.
                    if rank == 0:
                        writer.add_scalar("loss with entropy loss", loss.detach().cpu().numpy(), ctr)
                if args.distillation: ## Time to distill.
                    with torch.no_grad(): ## No gradient to avoid memory allocation.
                        parent_mod_compute = parent_model(input_ids=input_ids, attention_mask=input_masks, decoder_input_ids=decoder_input_ids, output_hidden_states=args.distillation, output_attentions=args.distillation) ## Get the parent model's computations.
                    distillation_loss = compute_distillation_losses(mod_compute, parent_mod_compute, labels, tok.pad_token_id, args) ## Compute distillation losses.
                    loss = args.distillation_loss_weight*distillation_loss + (1.0 - args.distillation_loss_weight)*loss ## Update the main loss with weighing and adding.
                    if rank == 0:
                        writer.add_scalar("distillation loss", distillation_loss.detach().cpu().numpy(), ctr)
                        writer.add_scalar("final loss", loss.detach().cpu().numpy(), ctr)

        input_ids=input_ids.to('cpu') ## Move to CPU. May not be needed but its a safety net. 
        input_masks=input_masks.to('cpu') ## Move to CPU. May not be needed but its a safety net.
        decoder_input_ids=decoder_input_ids.to('cpu') ## Move to CPU. May not be needed but its a safety net.
        labels=labels.to('cpu') ## Move to CPU. May not be needed but its a safety net.
        
        if args.fp16: ## The gradient scaler needs to be invoked with FP16/AMP computation.
            loss = loss/args.multistep_optimizer_steps
            scaler.scale(loss).backward()
            num_batches_this_optimizer_step += 1
            losses += loss
            if num_batches_this_optimizer_step < args.multistep_optimizer_steps:
                continue
        else:
            pass
        if args.fp16: ## With FP16/AMP computation we need to unscale gradients before clipping them. We then optimize and update the scaler.
            if args.max_gradient_clip_value != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else: ## With FP32, we just do regular backpropagation, gradient clipping and then step the optimizer.
            loss = loss/args.multistep_optimizer_steps
            loss.backward()
            num_batches_this_optimizer_step += 1
            losses += loss
            if num_batches_this_optimizer_step < args.multistep_optimizer_steps:
                continue
            if args.max_gradient_clip_value != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_clip_value)
            optimizer.step()
        scheduler.step() ## Advance the scheduler to get to the next value of LR.
        lv = losses.detach().cpu().numpy() ## Detach the loss in order to report it.
        losses = 0
        num_batches_this_optimizer_step = 0
        if ctr % 10 == 0 and rank % 8 == 0: ## Print the current loss every 10 batches but only for the master/prime process.
            print(ctr, lv)
            sys.stdout.flush()
        if ctr % 1000 == 0 and rank == 0 and args.save_weights_and_gradeint_info: ## Save the model weight and gradient info every time this condition is triggered.
            for param_name, param_value in model.named_parameters():
                if not ("embed_positions" in param_name and args.positional_encodings):
                    writer.add_histogram("weights."+param_name, param_value.detach().cpu().numpy(), ctr)
                    writer.add_histogram("gradients."+param_name, param_value.grad.detach().cpu().numpy(), ctr)
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
    parser.add_argument('--num_batches', default=2000000, type=int, 
                        help='Number of batches to train on')
    parser.add_argument('-a', '--ipaddr', default='localhost', type=str, 
                        help='IP address of the main node')
    parser.add_argument('-m', '--model_path', default='ddpdefault', type=str, 
                        help='Name of the model')
    parser.add_argument('--save_intermediate_checkpoints', action='store_true', 
                        help='Use this flag if you want intermediate checkpoints to be saved. If so then numbers will be attached to the checkpoints.')
    parser.add_argument('--use_official_pretrained', action='store_true', 
                        help='Use this flag if you want the argument "pretrained_model" to specify a pretrained model created by someone else.')
    parser.add_argument('--pretrained_model', default='', type=str, 
                        help='Name of the model')
    parser.add_argument('--no_reload_optimizer_ctr_and_scheduler', action='store_true',
                        help='Should we reload the optimizer, counter and secheduler? By default we always reload these. Set this to False if we only want to reload the model params and optimize from scratch.')
    parser.add_argument('--langs', default='', type=str, 
                        help='Comma separated string of source languages')
    parser.add_argument('--monolingual_domains', default='', type=str, 
                        help='In case we have multiple domains for monolingual corpora then domain indicator tokens should be provided as a comma separated list of tokens which be used to index the domain indicator. We will convert this into an index later. You have to provide values for this argument if you have more than one domains for the parallel or monolingual corpora.')
    parser.add_argument('--train_slang', default='en', type=str, 
                        help='Source language(s) for training')
    parser.add_argument('--train_tlang', default='hi', type=str, 
                            help='Target language(s) for training')
    parser.add_argument('--train_domains', default='', type=str, 
                        help='In case we have multiple domains for parallel corpora then domain indicator tokens should be provided as a comma separated list of tokens which be used to index the domain indicator. We will convert this into an index later. You have to provide values for this argument if you have more than one domains for the parallel or monolingual corpora.')
    parser.add_argument('--num_domains_for_domain_classifier', type=int, default=1, 
                        help='If we have multiple domains then we should set this to a value higher than one.')
    parser.add_argument('--gradient_reversal_for_domain_classifier', action='store_true', 
                        help='Should we do gradient reversal for the domain classifier? If true then all gradients below the softmax layer (meaning linear projection plus softmax activation) for the classifier will be reversed. Essentially, the representations for two domains will be forced to become more similar. This may in turn be used for style transfer.')
    parser.add_argument('--domain_classifier_loss_weight', type=float, default=0.1, 
                        help='What weight should we give to the domain classifier? 1 minus this weight will be given to the main loss.')
    parser.add_argument('--train_src', default='', type=str, 
                            help='Source language training sentences')
    parser.add_argument('--train_tgt', default='', type=str, 
                            help='Target language training sentences')
    parser.add_argument('--bilingual_train_frequency', default=-1, type=int, 
                        help='If this is -1 then we assume no bilingual corpora. If this is set to a value say 5 then every 5th batch will be a bilingual batch and all others will be monolingual batches.')
    parser.add_argument('--unify_encoder', action='store_true', 
                        help='Should we minimize the encoder representation distances instead of regular cross entropy minimization on the parallel corpus?')
    parser.add_argument('--mono_src', default='', type=str, 
                        help='Comma separated string of source language file prefixes. Make sure that these are split into N groups where N is the number of GPUs you plan to use.')
    parser.add_argument('--positional_encodings', action='store_true', 
                        help='If true then we will use positional encodings instead of learned positional embeddings.')
    parser.add_argument('--no_embed_norm', action='store_true', 
                        help='If true then we wont normalize embeddings.')
    parser.add_argument('--scale_embedding', action='store_true', 
                        help='Should we scale embeddings?')
    parser.add_argument('--no_scale_attention_embedding', action='store_true', 
                        help='Should we scale attention embeddings?')
    parser.add_argument('--is_document', action='store_true', 
                        help='This assumes that the input corpus is a document level corpus and each line is in fact a document. Each line also contains a token such as "[SEP]" (controlled by the "document_level_sentence_delimiter" flag) to mark the boundaries of sentences. When generating training data we will use this flag to select arbitrary sequences of sentences in case of long documents.')
    parser.add_argument('--document_level_sentence_delimiter', default='</s>', type=str, 
                        help='If the corpus is document level then we assume that sentences are separated via this token. Please change this in case you have a different type of delimiter.')
    parser.add_argument('--future_prediction', action='store_true', 
                        help='This assumes that we dont mask token sequences randomly but only after the latter half of the sentence. We do this to make the model more robust towards missing future information. Granted we can achieve this using wait-k but methinks this may be a better way of training.')
    parser.add_argument('--unidirectional_encoder', action='store_true', 
                        help='This assumes that we use a unidirectional encoder. This is simulated via a lower-triangular matrix mask in the encoder. Easy peasy lemon squeazy.')
    parser.add_argument('--tokenizer_name_or_path', default='ai4bharat/indic-bert', type=str, 
                        help='Name of or path to the tokenizer')
    parser.add_argument('--pretrained_tokenizer_name_or_path', default=None, type=str, 
                        help='Name of or path to the tokenizer of the pretrained model if its different from the current model. This tokenizer will be used for remapping embeddings so as to reuse as many pretrained embeddings as possible.')
    parser.add_argument('--tokenization_sampling', action='store_true', 
                        help='Should we use stoachastic tokenization aka BPE dropout or Subword regularization?')
    parser.add_argument('--tokenization_nbest_list_size', type=int, default=64, 
                        help='The size of the nbest list when doing stochastic tokenization.')
    parser.add_argument('--tokenization_alpha_or_dropout', type=float, default=0.1, 
                        help='The value of sentence piece regularization amount controlled via alpha or the amount of BPE dropout controlled by dropout.')
    parser.add_argument('--warmup_steps', default=16000, type=int,
                        help='Scheduler warmup steps')
    parser.add_argument('--multistep_optimizer_steps', default=1, type=int, help="In case you want to simulate a larger batch you should set this to a higher value.")
    parser.add_argument('--encoder_layers', default=6, type=int, help="The value for number of encoder layers")
    parser.add_argument('--decoder_layers', default=6, type=int, help="The value for number of decoder layers")
    parser.add_argument('--max_length', default=128, type=int, 
                        help='Maximum sequence length for training')
    parser.add_argument('--max_src_length', default=256, type=int, 
                        help='Maximum token length for source language')
    parser.add_argument('--max_tgt_length', default=256, type=int, 
                        help='Maximum token length for target language')
    parser.add_argument('--hard_truncate_length', default=0, type=int, 
                        help='Should we perform a hard truncation of the batch? This will be needed to eliminate cuda caching errors for when sequence lengths exceed a particular limit. This means self attention matrices will be massive and I used to get errors. Choose this value empirically.')
    parser.add_argument('--batch_size', default=4096, type=int, 
                        help='Maximum number of tokens in batch')
    parser.add_argument('--batch_size_indicates_lines', action='store_true', 
                        help='Should we batch as a fixed number of lines?')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help="The value for label smoothing.")
    parser.add_argument('--lr', default=1e-3, type=float, help="The value for the learning rate")
    parser.add_argument('--weight_decay', default=0.00001, type=float, help="The value for weight decay")
    parser.add_argument('--layerdrop', default=0.0, type=float, help="The value for layerdrop which indicates the probability that a whole layer will be bypassed via an identity transformation.")
    parser.add_argument('--dropout', default=0.1, type=float, help="The value for embedding dropout")
    parser.add_argument('--attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--encoder_attention_heads', default=16, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--decoder_attention_heads', default=16, type=int, help="The value for number of decoder attention heads")
    parser.add_argument('--wait_k', default=-1, type=int, help="The value for k in wait-k snmt. Keep as -1 for non-snmt aka vanilla NMT.")
    parser.add_argument('--mixed_wait_k', action='store_true', 
                        help='Should we train using up to wait_k? This can help simulate multiple wait_k')
    parser.add_argument('--decoder_ffn_dim', default=4096, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--encoder_ffn_dim', default=4096, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--d_model', default=1024, type=int, help="The value for model hidden size")
    parser.add_argument('--data_sampling_temperature', default=5.0, type=float, help="The value for data sampling temperature")
    parser.add_argument('--token_masking_lambda', default=3.5, type=float, help="The value for the poisson sampling lambda value")
    parser.add_argument('--token_masking_probs_range', nargs='+', type=float, default=[0.3], help="The range of probabilities with which the token will be masked. If you want a fixed probability then specify one argument else specify ONLY 2.")
    parser.add_argument('--max_gradient_clip_value', default=1.0, type=float, help="The max value for gradient norm")
    parser.add_argument('--softmax_temperature', default=1.0, type=float, help="The value for the softmax temperature")
    parser.add_argument('--distillation_temperature', default=1.0, type=float, help="The value for the softmax temperature during distillation")
    parser.add_argument('--temperature_calibration', action='store_true', 
                        help='Are we calibrating the temperature automatically during training? If yes then the softmax_temperature parameter should have a value of 1.0 furthermore the returned temperature will be used to scale the loss.')
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
    parser.add_argument('--use_official_parent_pretrained', action='store_true', 
                        help='Use this flag if you want the argument "pretrained_model" to specify a pretrained model created by someone else for the purposes of distillation. Use this carefully because if the parent is created by someone else then you have to have your own model with different configurations for fine-tuning. Essentially you must make sure that use_official_parent_pretrained and use_official_pretrained are not true simultaneously.')
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
    parser.add_argument('--save_weights_and_gradeint_info', action='store_true', 
                        help='Saving gradient information is time consuming. We should make this optional.')
    ###
    ### Placeholder flags to prevent code from breaking. These flags are not intended to be used for pretraining. These flags are here because the common_utils.py methods assume the existence of these args for when joint mbart training and regular NMT training is done. TODO: Modify code to avoid the need for these flags in this script.
    parser.add_argument('--multi_source', action='store_true', 
                        help='Are we doing multisource NMT? In that case you should specify the train_src as a hyphen separated pair indicating the parent language and the child language. You should also ensure that the source file is a tab separated file where each line contains "the parent pair source sentence[tab]child pair source sentence".')
    parser.add_argument('--cross_distillation', action='store_true', 
                        help='Should we perform cross distillation from a parent model which has been trained on another source language but the same target language? If so then you must specify the model using "parent_pretrained_model". Additionally you should specify the train_src as a hyphen separated pair indicating the parent language and the child language. You should also ensure that the source file is a tab separated file where each line contains "the parent pair source sentence[tab]child pair source sentence" There are several distillation options check the flag called "distillation_styles".')
    ###
    args = parser.parse_args()
    assert len(args.token_masking_probs_range) <= 2
    print("IP address is", args.ipaddr)

    args.world_size = args.gpus * args.nodes                #
    
    langs = args.langs.strip().split(",")
    mono_src = args.mono_src.strip().split(",")
    if args.num_domains_for_domain_classifier > 1: ## In case we have to do domain classification
        monolingual_domains = args.monolingual_domains.strip().split(",") ## Should not be empty
        args.train_domains = {} ## We can index the domain indicator this way
        domain_idx = 0    
        for monolingual_domain in monolingual_domains:
            if monolingual_domain not in args.train_domains:
                args.train_domains[monolingual_domain] = domain_idx
                domain_idx += 1
        files = {lang+"-"+monolingual_domain: [mono_file, args.train_domains[monolingual_domain]] for lang, mono_file, monolingual_domain in zip(langs, mono_src, monolingual_domains)}
        
    else:
        files = {lang: mono_file for lang, mono_file in zip(langs, mono_src)}
    print("Monolingual training files are:", files)
    
    
    train_files = {}
    if args.bilingual_train_frequency != -1:
        slangs = args.train_slang.strip().split(",")
        tlangs = args.train_tlang.strip().split(",")
        train_srcs = args.train_src.strip().split(",")
        train_tgts = args.train_tgt.strip().split(",")
        if args.num_domains_for_domain_classifier > 1: ## In case we have to do domain classification
            train_domains = args.train_domains.strip().split(",") ## Should not be empty
            for train_domain in train_domains:
                if train_domain not in args.train_domains:
                    args.train_domains[train_domain] = domain_idx
                    domain_idx += 1
            train_files = {slang+"-"+tlang+"-"+train_domain: (train_src, train_tgt, args.train_domains[train_domain]) for slang, tlang, train_src, train_tgt, train_domain in zip(slangs, tlangs, train_srcs, train_tgts, train_domains)}
        else:
            train_files = {slang+"-"+tlang: (train_src, train_tgt) for slang, tlang, train_src, train_tgt in zip(slangs, tlangs, train_srcs, train_tgts)}
        print("Parallel training files are:", train_files)
    
    if args.num_domains_for_domain_classifier > 1: ## In case we have to do domain classification
        print("Number of unique domains are ", len(args.train_domains))
    os.environ['MASTER_ADDR'] = args.ipaddr              #
    os.environ['MASTER_PORT'] = '26023'                      #
    mp.spawn(model_create_load_run_save, nprocs=args.gpus, args=(args,files,train_files,))         #
    
if __name__ == "__main__":
    run_demo()