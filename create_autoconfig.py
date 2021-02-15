from transformers import AutoConfig, AlbertTokenizer, AutoTokenizer
import sys
import os


tokenizer = AlbertTokenizer.from_pretrained(sys.argv[1])
tokenizer.save_pretrained(sys.argv[1])

os.rename(sys.argv[1]+"/tokenizer_config.json",sys.argv[1]+"/config.json")

config = AutoConfig.from_pretrained(sys.argv[1])
config.save_pretrained(sys.argv[1])

print("Testing tokenizer")

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])

print(tokenizer("I am Gone"))

print("Success")