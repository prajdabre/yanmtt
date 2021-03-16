src_files=$1 # Has to be a comma separated list
vocab_size=$2
tgt_folder=$3
type=$4
mkdir $tgt_folder

SPM_TRAIN=/share03/draj/softwares_and_scripts/sentencepiece/build/src/spm_train ## Change this to your spm_train path

if [[ $type == "albert" ]]
then
echo "ALBERT tokenizer"
$SPM_TRAIN   --input $src_files   --model_prefix=$tgt_folder/spiece --vocab_size=$vocab_size   --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1   --user_defined_symbols="[CLS],[SEP],[MASK]"   --shuffle_input_sentence=true   --character_coverage=0.99995 --model_type=unigram
elif [[ $type == "mbart" ]]
then
echo "MBART tokenizer."
$SPM_TRAIN   --input $src_files   --model_prefix=$tgt_folder/sentencepiece.bpe --vocab_size=$vocab_size   --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1   --user_defined_symbols="[CLS],[SEP],[MASK]"   --shuffle_input_sentence=true   --character_coverage=0.99995 --model_type=bpe
else
echo "Unknown tokenizer."
exit
fi

python create_autoconfig.py $tgt_folder $type $src_files
