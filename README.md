# MERLIon CCS Baseline System

Result and description of baseline system for MERLIon CCS Challenge [here](https://github.com/MERLIon-Challenge/merlion-ccs-2023/blob/master/readme.md#baseline-system).

Example command to run the training script:
>python train_conformer.py --dim 39 --train /home/challenge_feat_all_train.txt --test /home/devset_feats.txt --warmup 5000 --epochs 

The challenge_feat_all_train.txt is formatted as:
>chunk_1_feature.npy 0  
>chunk_2_feature.npy 1  

where 0 and 1 are language label indexes denoting English and Mandarin respectively.  

Example command to compute Equal Error Rate for Task 1 (Language Identification), i.e., compute_eer_bac.py:
>python compute_eer_bac.py --valid /your_utterance_to_language_index.txt --score /your_utterance_to_prediction.txt --trial /path_to_save_trial.txt

The your_utterance_to_language_index.txt is formatted as:
>chunk_1 0  
>chunk_2 1  
>
where 0 and 1 are ground-truth language label indexes denoting English and Mandarin respectively.

The your_utterance_to_language_index.txt is the prediction labels and can have two formats.

First format:
>chunk_1 0 5.12316  
>chunk_1 1 -12.66789
>
where 0 and 1 are the predicted language label indexes denoting English and Mandarin respectively, followed by the language prediction scores. 

Second format:
>chunk_1 5.12316 -12.66789
>
where the first language prediction score is for English followed by the language prediction score for Mandarin.  


Example to run diarization_validation.py (this is for our baseline system)  
>python diarization_validation.py --model /home/merlion/model.ckpt --audio /home/MERLIon-CCS-Challenge_Development-Set_v001/_CONFIDENTIAL/_audio/ --save /home/devset_diar/


If you have already computed the RTTMs for Task 2 (Language Diarization), the language diarization error rate and individual English and Mandarin error rates across the entire dataset can be computed by uncommenting the code in scoring_diar.py and running the following command: 
>python scoring_diar.py --predicting_file /your_folder_saved_prediction_rttm_files --ground_truth /your_folder_saved_ground_truth_rttm --excel_file /path_to_evaluated_regions_file --result_output /expected_path_to_save_result

where:
* --predicting_file is the folder path containing all the predicted RTTM files named according to the audio filenames (e.g., predicted RTTM file of 123.wav should be 123.txt in the prediction folder)
* --ground_truth is the folder path containing all ground truth RTTM files with the same audio filename (e.g., ground truth RTTM file of 123.wav should be 123.txt in the prediction folder
* --excel_file is the path that points to the Excel file that contains the timestamps (in milliseconds) of the evaluated regions for each audio recording.
* --result_output is the folder to save the results to. 

Note that the Excel file that contains the timestamps of evaluated regions is only released for the MERLIon CCS Development set. It is made available to the registered participants of the MERLIon CCS challenge. For the evaluation set in Task 2 (Language Diarization), the timestamps of evaluated regions will not be made available, so as to delink the information between Task 1 (Language Identification) and Task 2 (Language Diarization). 

We have also provided preprocess_train.py for training data processing (just in case you need), the dev_process.py for task 1 and dev_process_diar.py for task 2 to help develop your model.
