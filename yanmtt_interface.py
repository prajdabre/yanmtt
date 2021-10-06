import web
from web import form

import transformers
from transformers import AutoTokenizer, MBartTokenizer, MBart50Tokenizer, BartTokenizer
from transformers import MBartForConditionalGeneration, BartForConditionalGeneration, MBartConfig
import torch
import torch.nn as nn

import sys, os


os.environ['MASTER_ADDR'] = "localhost"              #
os.environ['MASTER_PORT'] = "34567"                     #

tok = AutoTokenizer.from_pretrained("/share03/draj/data/monolingual_corpora/indic/albert-indic64k", do_lower_case=False, use_fast=False, keep_accents=True)

config = MBartConfig(vocab_size=len(tok), encoder_layers=6, decoder_layers=6, dropout=0.1, attention_dropout=0.1, activation_dropout=0.1, encoder_attention_heads=16, decoder_attention_heads=16, encoder_ffn_dim=4096, decoder_ffn_dim=4096, d_model=1024, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"], add_special_tokens=False).input_ids[0][0], bos_token_id=tok(["<s>"], add_special_tokens=False).input_ids[0][0]) ## Configuration.

model = MBartForConditionalGeneration(config)
model = model.eval()
checkpoint_dict = torch.load("/share03/draj/data/monolingual_corpora/indic/fixed_vocab_model/ddpmodel.all.transformer_big.6-layer.64k.ls-0.1.drop-0.1.warmup-16k.gradclip-1.0.lr-1em3.wd-0.00001.750000.pure_model")
if type(checkpoint_dict) == dict:
    model.load_state_dict(checkpoint_dict["model"])
else:
    model.load_state_dict(checkpoint_dict)

render = web.template.render('templates/')

urls = ('/', 'index')
app = web.application(urls, globals())

myform = form.Form( 
    form.Textbox('sourcesent', value=""),
    form.Dropdown(name='source', args=["English"], value="English"),
    form.Dropdown(name='target', args=["English"], value="English"),
    ) 

class index: 

    def GET(self): 
        form = myform()
        # make sure you create a copy of the form by calling it (line above)
        # Otherwise changes will appear globally
        return render.yanmtt_interface(form, "")

    def POST(self): 
        form = myform()
        if not form.validates(): 
            return render.register(form, "")
        else:
            print(form.sourcesent.value)
            input_sentence_tmp=tok(form.sourcesent.value + " </s> <2en>", add_special_tokens=False, return_tensors="pt", padding=True)
            input_sentence=input_sentence_tmp.input_ids
            input_mask=input_sentence_tmp.attention_mask
            print(input_sentence, input_mask)
            output = model.generate(input_sentence, use_cache=True, num_beams=2, max_length=len(input_sentence[0])+100, min_length=1, early_stopping=True, attention_mask=input_mask, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"], add_special_tokens=False).input_ids[0][0], decoder_start_token_id=tok(["<2en>"], add_special_tokens=False).input_ids[0][0], bos_token_id=tok(["<s>"], add_special_tokens=False).input_ids[0][0], length_penalty=1.0) ## We translate the batch.
            output = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return render.yanmtt_interface(form, output)

if __name__=="__main__":
    web.internalerror = web.debugerror
    app.run()