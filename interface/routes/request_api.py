"""The Endpoints to manage the BOOK_REQUESTS"""
import uuid
from datetime import datetime, timedelta
from flask import jsonify, abort, request, Blueprint, render_template
from bertviz.bertviz import model_view
from config import MODELS_PATH
from transformers import  MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
import json
import torch
import random
import os
import numpy as np
from validate_email import validate_email
REQUEST_API = Blueprint('request_api', __name__)


tokenizer = ''
model = ''
device = "cuda:0" if torch.cuda.is_available() else "cpu"
langidLangs = ["af", "am", "ar", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "ga", "gl", "gu", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "ne", "nl", "no", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "si", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "th", "tl", "tr", "uk", "ur", "vi", "xh", "zh", "zu"]

languages = ["Afrikaans", "Amharic", "Arabic", "Asturian", "Azerbaijani", "Bashkir", "Belarusian", "Bulgarian", "Bengali", "Breton", "Bosnian", "Valencian", "Cebuano", "Czech", "Welsh", "Danish", "German", "Greeek", "English", "Spanish", "Estonian", "Persian", "Fulah", "Finnish", "French", "Irish", "Scottish Gaelic", "Galician", "Gujarati", "Hausa", "Hebrew", "Hindi", "Croatian", "Haitian Creole", "Hungarian", "Armenian", "Indonesian", "Igbo", "Iloko", "Icelandic", "Italian", "Japanese", "Javanese", "Georgian", "Kazakh", "Central Khmer", "Kannada", "Korean", "Letzeburgesch", "Ganda", "Lingala", "Lao", "Lithuanian", "Latvian", "Malagasy", "Macedonian", "Malayalam", "Mongolian", "Marathi", "Malay", "Burmese", "Nepali", "Flemish", "Norwegian", "Northern Sotho", "Occitan", "Oriya", "Punjabi", "Polish", "Pashto", "Portuguese", "Moldovan", "Russian", "Sindhi", "Sinhalese", "Slovak", "Slovenian", "Somali", "Albanian", "Serbian", "Swati", "Sundanese", "Swedish", "Swahili", "Tamil", "Thai", "Tagalog", "Tswana", "Turkish", "Ukrainian", "Urdu", "Uzbek", "Vietnamese", "Wolof", "Xhosa", "Yiddish", "Yoruba", "Chinese", "Zulu"]
langslow = (map(lambda x: x.lower(), languages))
langCodes = ["af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"]
langDict = dict(zip(langslow,langCodes))

def mask_spans(sentence):
    """Mask the spans in the text"""
    mask_count = 0
    mask_percent = 0.35
    mask_tok = "[MASK]"
    token_masking_lambda = 3.5
    sentence_split = sentence.split(" ")
    sent_len = len(sentence_split)
    max_mask_count = int(mask_percent*sent_len)
    spans_to_mask = list(np.random.poisson(token_masking_lambda, 1000))
    curr_sent_len = sent_len
    while mask_count < max_mask_count:
        try:
            span_to_mask = spans_to_mask[0]
            del spans_to_mask[0]
            if span_to_mask > (max_mask_count-mask_count): ## Cant mask more than the allowable number of tokens.
                continue
            idx_to_mask = random.randint(0, (curr_sent_len-1)-(span_to_mask-1)) ## We mask only the remaining half of the sentence to encourage the model to learn representations that can make do without most of the future tokens.
            if mask_tok not in sentence_split[idx_to_mask:idx_to_mask+span_to_mask]:
                actually_masked_length = len(sentence_split[idx_to_mask:idx_to_mask+span_to_mask]) ## If at the end of the sentence then we have likely masked fewer tokens.
                sentence_split[idx_to_mask:idx_to_mask+span_to_mask] = [mask_tok]
                mask_count += actually_masked_length # We assume that with a low probability there are mask insertions when span lengths are 0 which may cause more mask tokens than planned. I have decided not to count these insersions towards the maximum maskable limit. This means that the total number of mask tokens will be a bit higher than what it should be. 
                curr_sent_len -= (actually_masked_length-1)
        except:
            break ## If we cannot get a properly masked sentence despite all our efforts then we just give up and continue with what we have so far.
    
    return " ".join(sentence_split)

def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


@REQUEST_API.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@REQUEST_API.route('/models', methods=['GET'])
def models():
    list_of_models = [ name for name in os.listdir(MODELS_PATH) if os.path.isdir(os.path.join(MODELS_PATH, name)) ] + ["ai4bharat/IndicBART", "ai4bharat/IndicBARTSS"]
    json_response = {"models": list_of_models}
    return jsonify(json_response)


@REQUEST_API.route('/load_model', methods=['POST'])
def load_model():
    global tokenizer, model
    if not request.form:
        abort(400)
    model_name = request.form['model_name']
    try:
        ## Hindi <2hi> English <2en> format parsing
        sourceLangDict = {}
        targetLangDict = {}
        if model_name == "ai4bharat/indicbart":
            path = "ai4bharat/IndicBART"
            Lines = ["English <2en> English <2en>", "Hindi <2hi> Hindi <2hi>", "Bengali <2bn> Bengali <2bn>", "Gujarati <2gu> Gujarati <2gu>", "Kannada <2kn> Kannada <2kn>", "Malayalam <2ml> Malayalam <2ml>", "Marathi <2mr> Marathi <2mr>", "Odia <2or> Odia <2or>", "Punjabi <2pa> Punjabi <2pa>", "Tamil <2ta> Tamil <2ta>", "Telugu <2te> Telugu <2te>", "Assamese <2as> Assamese <2as>"]
        elif  model_name == "ai4bharat/indicbartss":
            path = "ai4bharat/IndicBARTSS"
            Lines = ["English <2en> English <2en>", "Hindi <2hi> Hindi <2hi>", "Bengali <2bn> Bengali <2bn>", "Gujarati <2gu> Gujarati <2gu>", "Kannada <2kn> Kannada <2kn>", "Malayalam <2ml> Malayalam <2ml>", "Marathi <2mr> Marathi <2mr>", "Odia <2or> Odia <2or>", "Punjabi <2pa> Punjabi <2pa>", "Tamil <2ta> Tamil <2ta>", "Telugu <2te> Telugu <2te>", "Assamese <2as> Assamese <2as>"]
        else:
            path = MODELS_PATH + "/" + model_name
            lang_path = MODELS_PATH + "/" + model_name + "/supported_languages.txt"
            file1 = open(lang_path, 'r')
            Lines = file1.readlines()
        
        for line in Lines:
            lineSplit = line.split()
            sourceLangDict[lineSplit[0]] = lineSplit[1].replace('2', '')
            targetLangDict[lineSplit[2]] = lineSplit[3].replace('2', '')
        
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=False, do_lower_case=False, use_fast=False, keep_accents=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=False).to(device)
        return jsonify({"message": "success", "sourceLangDict": sourceLangDict, "targetLangDict": targetLangDict})
    except:
        return jsonify({"message": "fail"})

@REQUEST_API.route('/translate', methods=['POST'])
def translate():
    global tokenizer, model, device
    if not request.form:
        abort(400)
    
    source_text = request.form['rawtext']
    source_l = langDict[request.form['sourcelang'].lower()]
    if(source_l == ''):
        tokenizer.src_lang = "en"
    else:
        tokenizer.src_lang = source_l
    
    target_l = langDict[request.form['targetlang'].lower()]
    if(target_l == ''):
        target_l = "de"
    else:
        tokenizer.tgt_lang = target_l
    model_name = request.form['model_name']
    if model_name == "ai4bharat/indicbart" or model_name == "ai4bharat/indicbartss":
        source_text = mask_spans(source_text)
        print(source_text)

    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    input_suffix = " </s> <2"+source_l+">"
    input_sentence = source_text + input_suffix
    output_prefix = "<2"+target_l+"> "
    print(len(input_sentence.split(" ")))
    inp = tokenizer(input_sentence, add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(device)
    model_output=model.generate(inp, use_cache=False, num_beams=4, max_length=len(input_sentence.split(" "))*2, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc(output_prefix)).to(device)
    decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = {
        "raw_text": source_text,
        "translated_text": decoded_output
    }
    return jsonify(result), 200
    
    

@REQUEST_API.route('/visualize', methods=['POST'])
def visualize():

    if not request.form:
        abort(400)
    
    source_text = request.form['rawtext']
    source_l = langDict[request.form['sourcelang'].lower()]
    if(source_l == ''):
        tokenizer.src_lang = "en"
    else:
        tokenizer.src_lang = source_l
    
    target_l = langDict[request.form['targetlang'].lower()]
    if(target_l == ''):
        target_l = "de"
    else:
        tokenizer.tgt_lang = target_l
    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    input_suffix = " </s> <2"+source_l+">"
    input_sentence = source_text + input_suffix
    output_prefix = "<2"+target_l+"> "
    
    inp = tokenizer(input_sentence, add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(device)
    model_output=model.generate(inp, use_cache=False, num_beams=4, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc(output_prefix.strip())).to(device)
    decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    outputs = model(input_ids=inp, decoder_input_ids=model_output, output_attentions=True)
    pyparams, vishtml=model_view(
        encoder_attention=outputs.encoder_attentions,
        decoder_attention=outputs.decoder_attentions,
        cross_attention=outputs.cross_attentions,
        encoder_tokens= tokenizer.convert_ids_to_tokens(inp[0]),
        decoder_tokens= tokenizer.convert_ids_to_tokens(model_output[0]),
    )
    result = {
        "pyparams": pyparams,
        "vishtml": vishtml.data
    }
    return jsonify(result), 200