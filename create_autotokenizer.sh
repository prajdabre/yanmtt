src_files=$1 # Has to be a comma separated list
vocab_size=$2
tgt_folder=$3

mkdir $tgt_folder

/share03/draj/softwares_and_scripts/sentencepiece/build/src/spm_train   --input $src_files   --model_prefix=$tgt_folder/spiece --vocab_size=$vocab_size   --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1   --user_defined_symbols="[CLS],[SEP],[MASK]"   --shuffle_input_sentence=true   --character_coverage=0.99995 --model_type=unigram

python create_autoconfig.py $tgt_folder