#!/usr/bin/env bash
### This script produces the 5 metrics for every database and video belonging in the test set.
### The .sh scripts produce all 5 metrics for each database, split and video.
### The six last Matlab scripts compute the average for all splits and videos per database for each metric.

### Run the script passing two arguments :
###     1) The FULL path of the base directory
###     2) The path where network predictions are saved
### example: sh compute_all_databases /home/test/stavis ./experiments/audiovisual_train_test

RESPATH=$1"/"$2"/results_per_video"
ANNOT_BASE_PATH=$1"/data"
PREDICTIONS_PATH=$1"/"$2


matlab -r "addpath(genpath('./eval_code')); create_qsub_avsal('$RESPATH', '$ANNOT_BASE_PATH', '$PREDICTIONS_PATH');"

sh eval_code/AVAD_eval.sh
sh eval_code/DIEM_eval.sh
sh eval_code/Coutrot1_eval.sh
sh eval_code/Coutrot2_eval.sh
sh eval_code/ETMD_eval.sh
sh eval_code/SumMe_eval.sh

matlab -r "addpath(genpath('./eval_code')); compute_all_diem('$RESPATH', '$ANNOT_BASE_PATH');"
matlab -r "addpath(genpath('./eval_code')); compute_all_coutrot1('$RESPATH', '$ANNOT_BASE_PATH');"
matlab -r "addpath(genpath('./eval_code')); compute_all_coutrot2('$RESPATH', '$ANNOT_BASE_PATH');"
matlab -r "addpath(genpath('./eval_code')); compute_all_summe('$RESPATH', '$ANNOT_BASE_PATH');"
matlab -r "addpath(genpath('./eval_code')); compute_all_etmd('$RESPATH', '$ANNOT_BASE_PATH');"
matlab -r "addpath(genpath('./eval_code')); compute_all_avad('$RESPATH', '$ANNOT_BASE_PATH');"

cat $RESPATH'/'final_results_DIEM.txt
cat $RESPATH'/'final_results_AVAD.txt
cat $RESPATH'/'final_results_Coutrot1.txt
cat $RESPATH'/'final_results_Coutrot2.txt
cat $RESPATH'/'final_results_SumMe.txt
cat $RESPATH'/'final_results_ETMD.txt
