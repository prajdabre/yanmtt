"""The Endpoints to manage the BOOK_REQUESTS"""
import uuid
from datetime import datetime, timedelta
from flask import jsonify, abort, request, Blueprint, render_template
from bertviz.bertviz import model_view
from config import MODELS_PATH
from transformers import  MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AlbertTokenizer, AutoTokenizer
import json
import torch
import os
from validate_email import validate_email
REQUEST_API = Blueprint('request_api', __name__)


tokenizer = ''
model = ''
device = "cuda:0" if torch.cuda.is_available() else "cpu"

languages = ["Afrikaans", "Amharic", "Arabic", "Asturian", "Azerbaijani", "Bashkir", "Belarusian", "Bulgarian", "Bengali", "Breton", "Bosnian", "Valencian", "Cebuano", "Czech", "Welsh", "Danish", "German", "Greeek", "English", "Spanish", "Estonian", "Persian", "Fulah", "Finnish", "French", "Irish", "Scottish Gaelic", "Galician", "Gujarati", "Hausa", "Hebrew", "Hindi", "Croatian", "Haitian Creole", "Hungarian", "Armenian", "Indonesian", "Igbo", "Iloko", "Icelandic", "Italian", "Japanese", "Javanese", "Georgian", "Kazakh", "Central Khmer", "Kannada", "Korean", "Letzeburgesch", "Ganda", "Lingala", "Lao", "Lithuanian", "Latvian", "Malagasy", "Macedonian", "Malayalam", "Mongolian", "Marathi", "Malay", "Burmese", "Nepali", "Flemish", "Norwegian", "Northern Sotho", "Occitan", "Oriya", "Punjabi", "Polish", "Pashto", "Portuguese", "Moldovan", "Russian", "Sindhi", "Sinhalese", "Slovak", "Slovenian", "Somali", "Albanian", "Serbian", "Swati", "Sundanese", "Swedish", "Swahili", "Tamil", "Thai", "Tagalog", "Tswana", "Turkish", "Ukrainian", "Urdu", "Uzbek", "Vietnamese", "Wolof", "Xhosa", "Yiddish", "Yoruba", "Chinese", "Zulu"]
langslow = (map(lambda x: x.lower(), languages))
langCodes = ["af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"]
langDict = dict(zip(langslow,langCodes))

def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


@REQUEST_API.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@REQUEST_API.route('/models', methods=['GET'])
def models():
    # return render_template("index.html")
    list_of_models = os.listdir(MODELS_PATH) 
    json_response = {"models": list_of_models}
    return jsonify(json_response)


@REQUEST_API.route('/load_model', methods=['POST'])
def load_model():
    global tokenizer, model
    if not request.form:
        abort(400)
    model_name = request.form['model_name']
    available_models = os.listdir(MODELS_PATH)

    if model_name not in available_models:
        try:
            model_name = model_name[:-1] + model_name[-1].upper()
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
            return jsonify({"message": "success"})
        except:
            return jsonify({"message": "fail"})
    else:
        try:
            path = MODELS_PATH + "/" + model_name
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, do_lower_case=False, use_fast=False, keep_accents=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to(device)
            # model.eval()
            lang_path = MODELS_PATH + "/" + model_name + "/lang.json"
            data = json.load(open(lang_path))
            return jsonify({"message": "success",
                            "source_lang": data["source_lang"],
                            "target_lang": data["target_lang"]})
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
    available_models = os.listdir(MODELS_PATH)
    if request.form['model'] not in available_models:
        encoded_hi = tokenizer([source_text], return_tensors="pt", padding=True).to(device)
        generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(target_l))
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        result = {
            "translated_text": translation[0]
        }
        return jsonify(result), 200
    elif request.form['model'] in available_models:
        bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
        eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
        pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
        input_suffix = " </s> <2"+source_l+">"
        input_sentence = source_text + input_suffix
        output_prefix = "<2"+target_l+"> "
        
        inp = tokenizer(input_sentence, add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(device)
        # print(next(model.parameters()).device)
        model_output=model.generate(inp, use_cache=False, num_beams=4, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc(output_prefix)).to(device)
        decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = {
            "translated_text": decoded_output
        }
        return jsonify(result), 200
        # print(model_output)
        # print(decoded_output)
        # out = tokenizer("<2hi> मैं  एक लड़का हूँ </s>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids # tensor([[64006,   942,    43, 32720,  8384, 64001]])
    else:
        abort(400)
    
    

@REQUEST_API.route('/visualize', methods=['POST'])
def visualize():

    if not request.form:
        abort(400)
    
    source_text = request.form['rawtext']
    # print(langDict)
    source_l = langDict[request.form['sourcelang'].lower()]
    if(source_l == ''):
        tokenizer.src_lang = "en"
    else:
        tokenizer.src_lang = source_l
    
    target_l = langDict[request.form['targetlang'].lower()]
    # print(target_l)
    if(target_l == ''):
        target_l = "de"
    else:
        tokenizer.tgt_lang = target_l
    available_models = os.listdir(MODELS_PATH)
    if request.form['model'] not in available_models:
        encoded_hi = tokenizer([source_text], return_tensors="pt", padding=True).to(device)
        generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(target_l))
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        outputs = model(input_ids=encoded_hi.input_ids, decoder_input_ids=generated_tokens, output_attentions=True)
        pyparams, vishtml=model_view(
            encoder_attention=outputs.encoder_attentions,
            decoder_attention=outputs.decoder_attentions,
            cross_attention=outputs.cross_attentions,
            encoder_tokens= tokenizer.convert_ids_to_tokens(encoded_hi.input_ids[0]),
            decoder_tokens= tokenizer.convert_ids_to_tokens(generated_tokens[0]),
        )
        result = {
            "translated_text": translation[0],
            "pyparams": pyparams,
            "vishtml": vishtml.data
        }
        return jsonify(result), 200
    elif request.form['model'] in available_models:
        bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
        eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
        pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
        input_suffix = " </s> <2"+source_l+">"
        input_sentence = source_text + input_suffix
        output_prefix = "<2"+target_l+"> "
        
        inp = tokenizer(input_sentence, add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(device)
        # print(next(model.parameters()).device)
        model_output=model.generate(inp, use_cache=False, num_beams=4, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>")).to(device)
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
            "translated_text": decoded_output,
            "pyparams": pyparams,
            "vishtml": vishtml.data
        }
        return jsonify(result), 200
    else:
        abort(400)
