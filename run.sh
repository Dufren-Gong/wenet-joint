#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
# export CUDA_VISIBLE_DEVICES="1"
stage=4 # start from 0 if you need to start from data preparation
stop_stage=4

. tools/parse_options.sh || exit 1;

# data directory
swbd1_dir=datasets/LDC97S62
train_set=train_nodup
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  for x in train test dev; do
    local/swbd1_data_prep.sh ${swbd1_dir}/ ${x}
    # process sets by
    # 1) convert lower to upper
    # 2) remove ._._ -1 symbols from text
    # 3) subset training set and dev set
    # 4) remove duplicated utterances
    cp data/${x}/text data/${x}/text.org
    paste -d" " <(cut -f 1 -d" " data/${x}/text.org) \
      <(cut -f 2- -d" " data/${x}/text.org | tr "[:lower:]" "[:upper:]") > data/${x}/text
    sed -i 's/\._/ /g; s/\.//g; s/THEM_1/THEM/g' data/${x}/text
  done
  tools/data/remove_dup_utts.sh 300 data/train data/train_nodup
fi

cmvn=true
feat_dir=raw
train_config=conf/train_u2++_efficonformer_v2_cs_16.yaml
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # For wav feature, just copy the data. Fbank extraction is done in training
  # mkdir -p ${feat_dir}
  # for x in ${train_set} test dev; do
  #   cp -r data/${x} ${feat_dir}
  # done
  if ${cmvm}; then
    tools/compute_cmvn_stats.py --num_workers 16 --train_config ${train_config} \
      --in_scp ${feat_dir}/${train_set}/wav.scp \
      --out_cmvn ${feat_dir}/${train_set}/global_cmvn
  fi
fi


# bpemode (unigram or bpe)
nbpe=2000
bpemode=bpe
dict=data/concat_lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/concat_lang_char/${train_set}_${bpemode}${nbpe}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "dictionary: ${dict}"
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p data/concat_lang_char/

  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " $feat_dir/${train_set}/text > data/concat_lang_char/input.txt

  tools/spm_train \
    --input=data/concat_lang_char/input.txt \
    --vocab_size=${nbpe} \
    --character_coverage=1.0 \
    --model_type=${bpemode} \
    --model_prefix=${bpemodel} \
    --input_sentence_size=100000000 \
    --user_defined_symbols="[LAUGHTER],[NOISE],[VOCALIZED-NOISE]"
  tools/spm_encode --model=${bpemodel}.model \
    --output_format=piece < data/concat_lang_char/input.txt | \
    tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}

  num_token=$(cat ${dict} | wc -l)
  echo "<sos/eos> ${num_token}" >> ${dict} # <eos>
  wc -l ${dict}
fi


nj=16
data_type=raw # raw or shard
num_utts_per_shard=1000
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare data, prepare required format"
  for x in ${train_set} test dev; do
    if [ ${data_type} == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard ${num_utts_per_shard} \
        --num_threads ${nj} ${feat_dir}/${x}/wav.scp ${feat_dir}/${x}/text \
        $(realpath ${feat_dir}/${x}/shards) ${feat_dir}/${x}/data.list
    else
      tools/make_raw_list.py ${feat_dir}/${x}/wav.scp ${feat_dir}/${x}/text \
        ${feat_dir}/${x}/data.list
    fi
  done
fi


# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
train_type=split1
save_dir=exp
dir=${save_dir}/${train_type}
shift=8037
train_set=swbd
checkpoint=
# checkpoint=${dir}/epoch_21.pt
train_config=conf/train_u2++_efficonformer_v2_cs_16.yaml
bpe_base=data/${train_set}_lang_char/${train_set}_${bpemode}${nbpe}
symbol_table_path=${bpe_base}_units.txt
bpe_path=${bpe_base}.model
HOST_NODE_ADDR="localhost:$shift"
rdzv_id=$((1024+shift))
num_nodes=1
prefetch=100
train_engine=torch_ddp
deepspeed_config=conf/ds_stage1.json
deepspeed_save_states="model_only"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p ${dir}
  # The number of gpus runing on each node/machine
  num_gpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
           --rdzv_id=${rdzv_id} --rdzv_backend="c10d" \
    local/me/bin/train.py \
      --save_model_num 20 \
      --train_engine ${train_engine} \
      --config ${train_config} \
      --data_type ${data_type} \
      --prefetch ${prefetch} \
      --train_data ./${feat_dir}/${train_set}/data.list \
      --cv_data ./${feat_dir}/test/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir ${dir} \
      --ddp.dist_backend ${dist_backend} \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states} \
      --dysf_log_temp_dir temp/$train_type \
      --symbol_table_path ${symbol_table_path}\
      --bpe_path ${bpe_path}\
      --cmvn_file ${feat_dir}/${train_set}/global_cmvn \
      --num_workers 1 \
      --pin_memory \
      --print_model
fi

# use average_checkpoint will get better result
average_checkpoint=true
average_num=20
#评估标准
evaluation_criteria=(wer f_score acc)
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  if [ ${average_checkpoint} == true ]; then
    for evaluation_mode in ${evaluation_criteria[@]}; do
      python local/me/bin/average_model.py \
        --evaluation_mode ${evaluation_mode} \
        --src_path $dir/best_model_info \
        --num ${average_num}
    done
  fi
fi


decode_checkpoint=$dir/epoch_200.pt
# decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
decode_modes="attention_rescoring"
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=${dir}/avg_${average_num}.pt
    echo "do model average and final checkpoint is ${decode_checkpoint}"
    python wenet/bin/average_model.py \
      --dst_model ${decode_checkpoint} \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.3
  reverse_weight=0.3
  for mode in ${decode_modes}; do
  {
    test_dir=${dir}/test_${mode}
    mkdir -p ${test_dir}
    python local/me/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data $feat_dir/${train_set}/data.list_small \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --blank_penalty 0.0 \
      --dict $dict \
      --bpe_model $bpemodel.model \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    sed -i.bak -r 's/<blank> //g' ${test_dir}/text
    mv ${test_dir}/text ${test_dir}/text.bak2
    tools/spm_decode --model=${bpemodel}.model --input_format=piece \
        < ${test_dir}/text.bak2 | sed -e "s/▁/ /g" > ${test_dir}/text
    python tools/compute-wer.py --char=1 --v=1 \
      $feat_dir/eval2000/text $test_dir/text > $test_dir/wer
  }
  done
  wait
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi
