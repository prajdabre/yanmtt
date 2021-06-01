src_files=$1 # Has to be a comma separated list
vocab_size=$2
tgt_folder=$3
type=$4
user_tokens=$5
mkdir -p $tgt_folder

SPM_TRAIN=/share03/draj/softwares_and_scripts/sentencepiece/build/src/spm_train ## Change this to your spm_train path

if [[ $type == "albert" ]]
then
echo "ALBERT tokenizer"
$SPM_TRAIN  --max_sentence_length 20000 --input $src_files   --model_prefix=$tgt_folder/spiece --vocab_size=$vocab_size   --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1   --user_defined_symbols="[CLS],[SEP],[MASK]"   --shuffle_input_sentence=true   --character_coverage=1.0 --model_type=unigram --input_sentence_size=1000000
elif [[ $type == "mbart" ]]
then
echo "MBART tokenizer."
$SPM_TRAIN  --max_sentence_length 20000 --input $src_files   --model_prefix=$tgt_folder/sentencepiece.bpe --vocab_size=$vocab_size   --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1   --user_defined_symbols="[CLS],[SEP],[MASK]"   --shuffle_input_sentence=true   --character_coverage=1.0 --model_type=bpe --input_sentence_size=1000000
else
echo "Unknown tokenizer."
exit
fi

## names of the bpe and spm models are set to sentencepiece.bpe and spiece as those are the names that the tokenizer implementations expect.

python /share03/draj/data/monolingual_corpora/indic/indic-mbart/create_autoconfig.py $tgt_folder $type $src_files $user_tokens
