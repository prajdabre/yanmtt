from transformers import AutoTokenizer, AlbertTokenizer
import time

from transformers import MBartForConditionalGeneration, MBartConfig, get_cosine_schedule_with_warmup
from transformers import AdamW

import os

import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import sys
import torch.distributed as dist

import random
import sacrebleu

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=0):
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


def get_sacrebleu(refs, hyp):
    bleu = sacrebleu.corpus_bleu(hyp, refs)
    return bleu.score

def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

def grad_status(model):
    return (par.requires_grad for par in model.parameters())


def freeze_params(model):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    try:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    except AttributeError:
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)

def generate_batches_eval(tok, args):
    src_file = open(args.dev_src)
    curr_batch_count = 0
    encoder_input_batch = []
    max_src_sent_len = 0

    for src_line in src_file:
        start = time.time()
        src_sent = src_line
        lang = "<2"+args.tlang+">"
        src_sent_split = src_sent.split(" ")
        sent_len = len(src_sent_split)
        if sent_len <1 or sent_len > 256:
            src_sent = " ".join(src_sent_split[:256])
        iids = tok(lang + " " + src_sent + " </s>", add_special_tokens=False, return_tensors="pt").input_ids
        curr_src_sent_len = len(iids[0])

        if curr_src_sent_len > max_src_sent_len:
            max_src_sent_len = curr_src_sent_len

        encoder_input_batch.append(lang + " " + src_sent + " </s>")
        curr_batch_count += 1
        if curr_batch_count == args.dev_batch_size:
            input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
            input_masks = input_ids != tok.pad_token_id
            end = time.time()
            #print("Batch generation time:", end-start, "seconds")
            yield input_ids, input_masks
            curr_batch_count = 0
            encoder_input_batch = []
            max_src_sent_len = 0

    if len(encoder_input_batch) != 0:
        input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True, max_length=max_src_sent_len).input_ids
        input_masks = input_ids != tok.pad_token_id
        yield input_ids, input_masks



def generate_batches(tok, args):
    batch_count = 0
    src_file = open(args.train_src)
    tgt_file = open(args.train_tgt)
    epoch_counter = 0
    while batch_count != args.num_batches:
        curr_batch_count = 0
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_label_batch = []
        batch_count += 1
        max_src_sent_len = 0
        max_tgt_sent_len = 0
        start = time.time()
        while curr_batch_count <= args.batch_size:
            src_sent, tgt_sent = src_file.readline().strip(), tgt_file.readline().strip()
            if src_sent == "" and tgt_sent == "":
                should_reset = True
                print("Checking for EOF!")
                for _ in range(100):
                    src_sent, tgt_sent = src_file.readline().strip(), tgt_file.readline().strip()
                    if src_sent != "" or tgt_sent != "":
                        should_reset = False
                        break
                if should_reset:
                    epoch_counter += 1
                    print("Finished epoch:", epoch_counter)
                    print("Going to beginning of training files.")
                    src_file.close()
                    tgt_file.close()
                    src_file = open(args.train_src)
                    tgt_file = open(args.train_tgt)
                    src_sent, tgt_sent = src_file.readline().strip(), tgt_file.readline().strip()
                    
    
            #curr_batch.append(["<2"+language_list[language_idx]+">", inline])
            #curr_batch_count += 1
            lang = "<2"+args.tlang+">"
            src_sent_split = src_sent.split(" ")
            tgt_sent_split = tgt_sent.split(" ")
            sent_len = len(tgt_sent_split)
            if sent_len <1 or sent_len > 100:
                continue
            iids = tok(lang + " " + src_sent + " </s>", add_special_tokens=False, return_tensors="pt").input_ids
            curr_src_sent_len = len(iids[0])
            
            iids = tok("<s> " + tgt_sent, add_special_tokens=False, return_tensors="pt").input_ids
            curr_tgt_sent_len = len(iids[0])
            if curr_src_sent_len < 1 or curr_src_sent_len > 100 or curr_tgt_sent_len < 1 or curr_tgt_sent_len > 100:
                continue
            if curr_src_sent_len > max_src_sent_len:
                max_src_sent_len = curr_src_sent_len
            
            if curr_tgt_sent_len > max_tgt_sent_len:
                max_tgt_sent_len = curr_tgt_sent_len
            
            encoder_input_batch.append(lang + " " + src_sent + " </s>")
            decoder_input_batch.append("<s> " + tgt_sent)
            decoder_label_batch.append(tgt_sent + " </s>")
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
            

def model_create_load_run_save(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    files = {"as": "data/as/as.txt", "bn": "data/bn/bn.txt", "en": "data/en/en.txt", "gu": "data/gu/gu.txt", "hi": "data/hi/hi.txt", "kn": "data/kn/kn.txt", "ml": "data/ml/ml.txt", "mr": "data/mr/mr.txt", "or": "data/or/or.txt", "pa": "data/pa/pa.txt", "ta": "data/ta/ta.txt", "te": "data/te/te.txt"}  ## Get this from command line
    
    special_tokens_dict = {'additional_special_tokens': ["<s>", "</s>"] + ["<2"+lang+">" for lang in files.keys()]}
    num_added_toks = tok.add_special_tokens(special_tokens_dict)

    print(tok)
    print(tok.pad_token_id)

    print(tok.vocab_size) ## Should be 20k

    print(len(tok)) ## Should be 20k + number of special tokens we added earlier

    print(f"Running DDP checkpoint example on rank {rank}.")
    #setup(rank, world_size)
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.fp16:
        print("We will do fp16 training")
        scaler = torch.cuda.amp.GradScaler()
    else:
        print("We will do fp32 training")
        # , add_final_layer_norm=True, normalize_before=True,
    model = MBartForConditionalGeneration(MBartConfig(vocab_size=len(tok), encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_dim, decoder_ffn_dim=args.decoder_ffn_dim, d_model=args.d_model, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1])) ## LS is actually not being used
    #model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model)
    model.train()
    torch.cuda.set_device(gpu)
    
    if args.freeze_embeddings:
        print("Freezing embeddings")
        freeze_embeds(model)
    if args.freeze_encoder:
        print("Freezing encoder")
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    model.cuda(gpu)
#    print(device)

    #print(model.config)
    #print(model.parameters)

    #model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7], dim=0)
    optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-06)
    model = DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_batches * args.world_size, 0.5)

    if args.pretrained_bilingual_model == "" and args.pretrained_model != "":
        print("Loading a pretrained mbart model")
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint_dict = torch.load(args.pretrained_model, map_location=map_location)
        if type(checkpoint_dict) == dict:
            model.load_state_dict(checkpoint_dict['model'])
        else:
            model.load_state_dict(checkpoint_dict)
    elif args.pretrained_bilingual_model != "":
        print("Loading a previous checkpoint")
        dist.barrier()
            # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint_dict = torch.load(CHECKPOINT_PATH, map_location=map_location)
        if type(checkpoint_dict) == dict:
            model.load_state_dict(checkpoint_dict['model'])
            optimizer.load_state_dict(checkpoint_dict['optimizer']) ## Dubious
            scheduler.load_state_dict(checkpoint_dict['scheduler']) ## Dubious
            ctr = checkpoint_dict['ctr']
        else:
            model.load_state_dict(checkpoint_dict)
            ctr = 0
    else:
        print("Training from scratch")
        ctr = 0
    print("Using label smoothing of", args.label_smoothing)
    #model.load_state_dict(torch.load("/share03/draj/data/monolingual_corpora/indic/trial_model/dpmodel"))
    #model.cuda()

    ## Print model config

    ## Compute dry run loss

#     loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)[0]
#     print(loss)

    #import pytorch_lightning as pl
    # model.cpu()
    # del model
    # torch.cuda.empty_cache()
    


    
    
    
    ctr = 0
    bleu_history = []
    max_sbleu = 0
    max_sbleu_step = 0
    curr_eval_step = 0
    for input_ids, input_masks, decoder_input_ids, labels in generate_batches(tok, args): #infinite_same_sentence(10000):
        start = time.time()
        if ctr % 1000 == 0:
            #model.save_pretrained("/share03/draj/data/monolingual_corpora/indic/trial_model/")
            #torch.save(model.state_dict(), "/share03/draj/data/monolingual_corpora/indic/trial_model/dpmodel")
            CHECKPOINT_PATH = args.fine_tuned_model
            if rank == 0:
                print("Running eval on dev set")
                refs = [[refline.strip() for refline in open(args.dev_tgt)]]
                hyp = []
                model.module.eval()
                for dev_input_ids, dev_input_masks in generate_batches_eval(tok, args): #infinite_same_sentence(10000):
                    start = time.time()
                    #print(input_ids)
                    translations = model.module.generate(dev_input_ids.to(gpu), num_beams=1, max_length=int(len(input_ids[0])*1.5), early_stopping=True, attention_mask=dev_input_masks.to(gpu), pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"]).input_ids[0][1], decoder_start_token_id=tok(["<s>"]).input_ids[0][1], bos_token_id=tok(["<s>"]).input_ids[0][1])
                    for translation in translations:
                        translation  = tok.decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False) 
                        hyp.append(translation)
                sbleu = get_sacrebleu(refs, hyp)
                print("BLEU score using sacrebleu after", ctr, "iterations is:", sbleu)
                if sbleu > max_sbleu:
                    max_sbleu = sbleu
                    max_sbleu_step = curr_eval_step
                    print("New peak reached. Saving.")
                    checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
                    torch.save(checkpoint_dict, CHECKPOINT_PATH+".best_dev_bleu."+str(ctr))
                    torch.save(model.module.state_dict(), CHECKPOINT_PATH+".best_dev_bleu."+str(ctr)+".pure_model") ## Pure model without any ddp markers or optimizer info.
                    
                if curr_eval_step - max_sbleu_step > args.early_stop_checkpoints:
                    print("We have seemingly converged as BLEU failed to increase for the following number of checkpoints:", args.early_stop_checkpoints, ". You may want to consider increasing the number of tolerance steps.")
                    print("Terminating training")
                    break
                bleu_history.append(sbleu)
                curr_eval_step += 1
                model.module.train()
                print("Saving the model")
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
                torch.save(checkpoint_dict, CHECKPOINT_PATH)
                

            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            dist.barrier()
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint_dict = torch.load(CHECKPOINT_PATH, map_location=map_location)
            model.load_state_dict(checkpoint_dict['model'])
            optimizer.load_state_dict(checkpoint_dict['optimizer']) ## Dubious
            scheduler.load_state_dict(checkpoint_dict['scheduler']) ## Dubious
            

        try:
            if args.fp16:
                with torch.cuda.amp.autocast():
                    mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu) ,decoder_input_ids=decoder_input_ids.to(gpu), labels=labels.to(gpu))
                    logits = mod_compute[1]
                    if args.label_smoothing == 0.0:
                        loss = mod_compute[0]
                    else:
                        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                        loss, nll_loss = label_smoothed_nll_loss(
                            lprobs, labels.to(gpu), args.label_smoothing, ignore_index=tok.pad_token_id
                        )
            else:
                mod_compute = model(input_ids=input_ids.to(gpu), attention_mask=input_masks.to(gpu) ,decoder_input_ids=decoder_input_ids.to(gpu), labels=labels.to(gpu))
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
#             print(input_ids)
#             for elem in input_ids:
#                 print(tok.convert_ids_to_tokens(elem))
            sys.stdout.flush()
            #sys.exit()
        #loss = torch.mean(loss)
        #print(loss)
        #loss = loss/((labels != tok.pad_token_id).to(gpu)).sum()
        optimizer.zero_grad()
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            pass
        lv = loss.detach().cpu().numpy()
        if ctr % 10 == 0 and rank == 0:
            print(ctr, lv)
#             for elem in input_ids:
#                 print(tok.convert_ids_to_tokens(elem))
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
    
    CHECKPOINT_PATH = args.fine_tuned_model
    print("Saving the model after the final step")
    # All processes should see same parameters as they all start from same
    # random parameters and gradients are synchronized in backward passes.
    # Therefore, saving it in one process is sufficient.
    checkpoint_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'ctr': ctr}
    torch.save(checkpoint_dict, CHECKPOINT_PATH)
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
    parser.add_argument('--encoder_layers', default=6, type=int, help="The value for number of encoder layers")
    parser.add_argument('--decoder_layers', default=6, type=int, help="The value for number of decoder layers")
    parser.add_argument('--label_smoothing', default=0.1, type=float, help="The value for label smoothing")
    parser.add_argument('--dropout', default=0.1, type=float, help="The value for embedding dropout")
    parser.add_argument('--attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--encoder_attention_heads', default=8, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--decoder_attention_heads', default=8, type=int, help="The value for number of decoder attention heads")
    parser.add_argument('--decoder_ffn_dim', default=2048, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--encoder_ffn_dim', default=2048, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--d_model', default=512, type=int, help="The value for model hidden size")
    parser.add_argument('--max_gradient_clip_value', default=1.0, type=float, help="The max value for gradient norm value")

    parser.add_argument('--pretrained_model', default='', type=str, 
                        help='Path to the pretrained model')
    parser.add_argument('--pretrained_bilingual_model', default='', type=str, 
                        help='Path to the pretrained bilingual model. Use this if you want to continue training a bilingual model.')
    parser.add_argument('-m', '--fine_tuned_model', default='pytorch.bin', type=str, 
                        help='Path to save the fine tuned model')
    parser.add_argument('--warmup_steps', default=16000, type=int,
                        help='Scheduler warmup steps')
    parser.add_argument('--batch_size', default=1024, type=int, 
                        help='Train batch sizes in tokens')
    parser.add_argument('--dev_batch_size', default=1024, type=int, 
                        help='Dev batch sizes in lines')
    parser.add_argument('--early_stop_checkpoints', default=10, type=int, 
                        help='Number of checkpoints to wait to see if BLEU increases.')
    parser.add_argument('--num_batches', default=100000, type=int, 
                        help='Number of batches to train on')
    parser.add_argument('--slang', default='en', type=str, 
                        help='Source language')
    parser.add_argument('--tokenizer_name_or_path', default='ai4bharat/indic-bert', type=str, 
                        help='Name of or path to the pre-trained indic language tokenizer')
    parser.add_argument('--tlang', default='hi', type=str, 
                        help='Target language')
    parser.add_argument('--train_src', default='', type=str, 
                        help='Source language training sentences')
    parser.add_argument('--train_tgt', default='', type=str, 
                        help='Target language training sentences')
    parser.add_argument('--dev_src', default='', type=str, 
                        help='Source language development sentences')
    parser.add_argument('--dev_tgt', default='', type=str, 
                        help='Target language development sentences')
    parser.add_argument('--fp16', action='store_true', 
                        help='Should we use fp16 training?')
    args = parser.parse_args()
    print("IP address is", args.ipaddr)
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = args.ipaddr              #
    os.environ['MASTER_PORT'] = args.port                      #
    mp.spawn(model_create_load_run_save, nprocs=args.gpus, args=(args,))         #
    #########################################################
#     mp.spawn(demo_fn,
#              args=(args,),
#              nprocs=args.gpus,
#              join=True)
    
if __name__ == "__main__":
    run_demo()