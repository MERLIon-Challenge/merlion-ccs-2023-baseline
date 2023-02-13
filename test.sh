#!/bin/bash
#$ -l ram_free=100G,mem_free=100G,hostname=b*
#$ -e /home/xzhan233/language_id/l.log
#$ -o /home/xzhan233/language_id/d.txt

predicting_file_folder='/home/language_id/pred'
ground_truth_folder='/home/language_id/ref'
result_output='/home/language_id'

python3 /home/xzhan233/language_id/scoring_diar.py --predicting_file ${predicting_file_folder} --ground_truth ${ground_truth_folder} --result_output ${result_output}
