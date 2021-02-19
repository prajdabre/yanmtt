from transformers import AutoTokenizer
import time

from transformers import MBartForConditionalGeneration, MBartConfig, get_linear_schedule_with_warmup
from transformers import AdamW

import os

import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import sys
import numpy as np
import torch.distributed as dist

torch.manual_seed(621311)

import random

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None):
    """From fairseq"""
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

    nll_loss = nll_loss.mean()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def generate_batches(tok, num_batches=1000, batch_size=2048, mp_val_or_range=0.3, lamb=3.5, rank=0, temperature=5.0, languages=""):
    files = {"as": "data/as/as.txt", "bn": "data/bn/bn.txt", "en": "data/en/en.txt", "gu": "data/gu/gu.txt", "hi": "data/hi/hi.txt", "kn": "data/kn/kn.txt", "ml": "data/ml/ml.txt", "mr": "data/mr/mr.txt", "or": "data/or/or.txt", "pa": "data/pa/pa.txt", "ta": "data/ta/ta.txt", "te": "data/te/te.txt"} ## Get this from command line

    probs = {"as": 1388109, "or": 6942483, "en": 54250995, "mr": 33976000, "pa": 29194279, "gu": 41129078, "ta": 31542481, "te": 47877462, "bn": 39877942, "kn": 53266064, "ml": 56061611, "hi": 63057909} ## Get this by automatic calculation
    batch_count = 0
    language_list = list(files.keys()) if languages == "" else languages.strip().split("-")
    print("Training for:", language_list)
    probs = {lang: probs[lang] for lang in language_list} ## Narrow it down
    files = {lang: files[lang] for lang in language_list} ## Narrow it down
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = probs_temp
    probs_temp = {lang: probs[lang]**(1.0/temperature) for lang in probs}
    probs = probs_temp
    probs_temp = {lang: probs[lang]/sum(probs.values()) for lang in probs}
    probs = [probs_temp[lang] for lang in language_list] ## NARROW IT DOWN
    num_langs = len(language_list)
    language_indices = list(range(num_langs))
    language_file_dict = {}
    for l in language_list:
        language_file_dict[l] = open(files[l]+"."+"%02d" % rank)
    while batch_count != num_batches:
        curr_batch_count = 0
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_label_batch = []
        batch_count += 1
        max_src_sent_len = 0
        max_tgt_sent_len = 0
        start = time.time()
        while curr_batch_count <= batch_size:
            language_idx = random.choices(language_indices, probs)[0]
            sentence = language_file_dict[language_list[language_idx]].readline().strip()
            if sentence == "":
                should_reset = True
                for _ in range(100):
                    sentence = language_file_dict[language_list[language_idx]].readline().strip()
                    if sentence != "":
                        should_reset = False
                        break
                if should_reset:
                    language_file_dict[language_list[language_idx]] = open(files[language_list[language_idx]]+"."+"%02d" % rank)
                    print("Reset", language_list[language_idx], files[language_list[language_idx]]+".""%02d" % rank)
                    sentence = language_file_dict[language_list[language_idx]].readline().strip()
            #curr_batch.append(["<2"+language_list[language_idx]+">", inline])
            #curr_batch_count += 1
            lang = "<2"+language_list[language_idx]+">"
            if type(mp_val_or_range) is float:
                mask_percent = mp_val_or_range
            else:
                mask_percent = random.uniform(mp_val_or_range[0], mp_val_or_range[1])
            sentence_split = sentence.split(" ")
            sent_len = len(sentence_split)
            if sent_len < 5 or sent_len > 80:
                continue
            mask_count = 0.0
            mask_percent_curr = 1.0*mask_count/sent_len
            spans_to_mask = list(np.random.poisson(lamb, 1000))
            curr_sent_len = sent_len
            while mask_percent_curr < mask_percent:
                span_to_mask = spans_to_mask[0]
                del spans_to_mask[0]
                if span_to_mask > curr_sent_len:
                    continue
                idx_to_mask = random.randint(0, (curr_sent_len-1)-(span_to_mask-1))
                if "[MASK]" not in sentence_split[idx_to_mask:idx_to_mask+span_to_mask]:
                    sentence_split[idx_to_mask:idx_to_mask+span_to_mask] = ["[MASK]"]
                    mask_count += span_to_mask
                    curr_sent_len -= (span_to_mask-1)
                    mask_percent_curr = 1.0*mask_count/sent_len
            masked_sentence = " ".join(sentence_split)
            iids = tok(lang + " " + masked_sentence + " </s>", add_special_tokens=False, return_tensors="pt").input_ids
            curr_src_sent_len = len(iids[0])
            
            iids = tok("<s> " + sentence, add_special_tokens=False, return_tensors="pt").input_ids
            curr_tgt_sent_len = len(iids[0])
            if curr_src_sent_len < 5 or curr_src_sent_len > 100 or curr_tgt_sent_len < 5 or curr_tgt_sent_len > 100:
                continue
            if curr_src_sent_len > max_src_sent_len:
                max_src_sent_len = curr_src_sent_len
            
            if curr_tgt_sent_len > max_tgt_sent_len:
                max_tgt_sent_len = curr_tgt_sent_len
            encoder_input_batch.append(masked_sentence + " </s> " + lang)
            decoder_input_batch.append(lang + " " + sentence)
            decoder_label_batch.append(sentence + " </s>")
            curr_batch_count += curr_tgt_sent_len
        #print("Max source and target lengths are:", max_src_sent_len, "and", max_tgt_sent_len)
        #print("Number of sentences in batch:", len(encoder_input_batch))
        input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
        input_masks = input_ids != tok.pad_token_id
        decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=curr_tgt_sent_len).input_ids
        labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=curr_tgt_sent_len).input_ids
        end = time.time()
        #print("Batch generation time:", end-start, "seconds")
        yield input_ids, input_masks, decoder_input_ids, labels
       
def init_weights(module, in_features, out_features):
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

def model_create_load_run_save(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=False)
    files = {"as": "data/as/as.txt", "bn": "data/bn/bn.txt", "en": "data/en/en.txt", "gu": "data/gu/gu.txt", "hi": "data/hi/hi.txt", "kn": "data/kn/kn.txt", "ml": "data/ml/ml.txt", "mr": "data/mr/mr.txt", "or": "data/or/or.txt", "pa": "data/pa/pa.txt", "ta": "data/ta/ta.txt", "te": "data/te/te.txt"}  ## Get this from command line
    
    special_tokens_dict = {'additional_special_tokens': ["<s>", "</s>"] + ["<2"+lang+">" for lang in files.keys()]}
    num_added_toks = tok.add_special_tokens(special_tokens_dict)

    #print(tok)
    

    #print(tok.vocab_size) ## Should be 20k

    #print(len(tok)) ## Should be 20k + number of special tokens we added earlier

    print(f"Running DDP checkpoint example on rank {rank}.")
    #setup(rank, world_size)
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.fp16:
        print("We will do fp16 training")
        scaler = torch.cuda.amp.GradScaler()
    else:
        print("We will do fp32 training")
    model = MBartForConditionalGeneration(MBartConfig(vocab_size=len(tok), encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_dim, decoder_ffn_dim=args.decoder_ffn_dim, d_model=args.d_model, add_final_layer_norm=args.add_final_layer_norm, normalize_before=args.normalize_before, normalize_embedding=args.normalize_embedding, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1], static_position_embeddings=True))
    torch.cuda.set_device(gpu)

#    model = MBartForConditionalGeneration.from_pretrained("/share03/draj/data/monolingual_corpora/indic/trial_model/")
    model.cuda(gpu)
    model.train()
    if args.initialization_model == "":
        for module in model.modules():
            print("Manual initialization")
            if isinstance(module, nn.Linear):
                init_weights(module, module.in_features, module.out_features)
            if isinstance(module, torch.nn.Embedding):
                init_weights(module, len(tok), args.d_model) ## Might need modification
    #    print(device)

    #print(model.config)
    #print(model.parameters)

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
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-05, eps=1e-09)
    
    model = DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)
    #model.load_state_dict(torch.load("/share03/draj/data/monolingual_corpora/indic/trial_model/dpmodel"))
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.iters*args.world_size)
    if args.initialization_model != "":
        print("Loading from checkpoint")
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        sys.stdout.flush()
        checkpoint_dict = torch.load(args.initialization_model, map_location=map_location)
        if type(checkpoint_dict) == dict:
            model.load_state_dict(checkpoint_dict['model'])
            optimizer.load_state_dict(checkpoint_dict['optimizer']) ## Dubious
            scheduler.load_state_dict(checkpoint_dict['scheduler']) ## Dubious
            ctr = checkpoint_dict['ctr']
        else:
            model.load_state_dict(checkpoint_dict)
            ctr = 0
    else:
        ctr = 0
    print("Using label smoothing of", args.label_smoothing)

    #model.cuda()

    ## Print model config

    ## Compute dry run loss

#     loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)[0]
#     print(loss)

    #import pytorch_lightning as pl
    # model.cpu()
    # del model
    # torch.cuda.empty_cache()
    


    
    
    
    for input_ids, input_masks, decoder_input_ids, labels in generate_batches(tok, args.iters, 512, (0.30, 0.40), 3.5, rank, args.temperature, args.languages): #infinite_same_sentence(10000):
        start = time.time()
        
        if ctr % 1000 == 0:
            #model.save_pretrained("/share03/draj/data/monolingual_corpora/indic/trial_model/")
            #torch.save(model.state_dict(), "/share03/draj/data/monolingual_corpora/indic/trial_model/dpmodel")
            CHECKPOINT_PATH = args.model_path
            if rank == 0:
                print("Saving the model")
                sys.stdout.flush()
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
                torch.save(checkpoint_dict, CHECKPOINT_PATH)

            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            dist.barrier()
            # configure map_location properly
            print("Loading from checkpoint")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            sys.stdout.flush()
            checkpoint_dict = torch.load(CHECKPOINT_PATH, map_location=map_location)
            model.load_state_dict(checkpoint_dict['model'])
            optimizer.load_state_dict(checkpoint_dict['optimizer']) ## Dubious
            scheduler.load_state_dict(checkpoint_dict['scheduler']) ## Dubious

        try:
            if args.fp16:
                with torch.cuda.amp.autocast():
                    mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu), labels=labels.to(gpu))
                    logits = mod_compute[1]
                    if args.label_smoothing == 0.0:
                        loss = mod_compute[0]
                    else:
                        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                        loss, nll_loss = label_smoothed_nll_loss(
                            lprobs, labels.to(gpu), args.label_smoothing, ignore_index=tok.pad_token_id
                        )
            else:
                mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu), decoder_input_ids=decoder_input_ids.to(gpu), labels=labels.to(gpu))
                logits = mod_compute[1]
                if args.label_smoothing == 0.0:
                    loss = mod_compute[0]
                else:
                    lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                    loss, nll_loss = label_smoothed_nll_loss(
                        lprobs, labels.to(gpu), args.label_smoothing, ignore_index=tok.pad_token_id
                    )
        except:
            print("NAN loss was computed or something messed up")
            sys.stdout.flush()
            #print(input_ids)
            #for elem in input_ids:
            #    print(tok.convert_ids_to_tokens(elem))
            continue
        #loss = torch.mean(loss)
        #print(len(mod_compute))
        #loss = loss/(labels.to(gpu) != tok.pad_token_id).sum()
        optimizer.zero_grad()
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            pass
        lv = loss.detach().cpu().numpy()
        if ctr % 10 == 0 and rank == 0:
            print(ctr, lv)
            sys.stdout.flush()
        #loss.backward()
        #optimizer.step()
        if args.fp16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_clip_value)
            optimizer.step()
        scheduler.step()
        end = time.time()
        #print("Batch processing time:", end-start, "seconds")
        ctr += 1
    
    checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
    torch.save(checkpoint_dict, CHECKPOINT_PATH)
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
    parser.add_argument('--add_final_layer_norm', action='store_true', 
                        help='Should we add a final layer norm?')
    parser.add_argument('--normalize_before', action='store_true', 
                        help='Should we normalize before doing attention?')
    parser.add_argument('--normalize_embedding', action='store_true', 
                        help='Should we normalize embeddings?')
    parser.add_argument('--scale_embedding', action='store_true', 
                        help='Should we scale embeddings?')
    parser.add_argument('-l', '--languages', default="", type=str, 
                        help='Hyphen separated list of the language or languages to pre-train on.')
    parser.add_argument('--tokenizer_name_or_path', default='ai4bharat/indic-bert', type=str, 
                        help='Name of or path to the pre-trained indic language tokenizer')
    parser.add_argument('--warmup_steps', default=16000, type=int,
                        help='Scheduler warmup steps')
    parser.add_argument('--encoder_layers', default=12, type=int, help="The value for number of encoder layers")
    parser.add_argument('--decoder_layers', default=12, type=int, help="The value for number of decoder layers")
    parser.add_argument('--label_smoothing', default=0.1, type=float, help="The value for label smoothing.")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="The value for weight decay")
    parser.add_argument('--dropout', default=0.1, type=float, help="The value for embedding dropout")
    parser.add_argument('--attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--encoder_attention_heads', default=16, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--decoder_attention_heads', default=16, type=int, help="The value for number of decoder attention heads")
    parser.add_argument('--decoder_ffn_dim', default=4096, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--encoder_ffn_dim', default=4096, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--d_model', default=1024, type=int, help="The value for model hidden size")
    parser.add_argument('--temperature', default=5.0, type=float, help="The value for model hidden size")
    parser.add_argument('--max_gradient_clip_value', default=1.0, type=float, help="The max value for gradient norm")
    parser.add_argument('--fp16', action='store_true', 
                        help='Should we use fp16 training?')
    args = parser.parse_args()
    print("IP address is", args.ipaddr)
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = args.ipaddr              #
    os.environ['MASTER_PORT'] = '26023'                      #
    mp.spawn(model_create_load_run_save, nprocs=args.gpus, args=(args,))         #
    #########################################################
#     mp.spawn(demo_fn,
#              args=(args,),
#              nprocs=args.gpus,
#              join=True)
    
if __name__ == "__main__":
    run_demo()