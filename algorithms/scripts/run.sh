#!/bin/bash
# Usage: bash scripts/run.sh


task=video_classification_ucf101_basenet
tag=uc101
mode=train # train or validate
retrain=True # TODO Caution

if [[ $retrain == "True" ]] && [[ -d models/$tag ]] && [[ $mode == "train" ]];then
  rm -rf models/$tag
fi

python -u main.py \
  --task $task \
  --mode $mode \
  --multiprocessing_distributed \
  --tag $tag \

