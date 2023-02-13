import argparse
import os
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument("--predicting_file", type=str, default=None, required=False, help = "Please input predicting file path")
# parser.add_argument("--ground_truth", type=str, default=None, required=False, help = "please input ground truth path")
# parser.add_argument("--result_output", type=str, default=None, required=False, help = "please input output file path")
# args = vars(parser.parse_args())

class language_diarization_error_rate:
    def __init__(self, predicting_file, ground_truth, result_output):
        self.predicting_file = predicting_file  # predicting file path, txt file 
        self.ground_truth = ground_truth        # ground truth file path, txt file
        self.result_output = result_output      # output truth file path

    
    def load_file(self, file):
        no_nonspeech_index_list = []

        with open(file) as f:
            for line in f.readlines():
                temp = line.split()
                start_time, end_time, language = temp[0], temp[1], temp[2] 
                
                factor = 1 
                start_time = round(float(start_time))
                end_time = round(float(end_time))

                if language == 'English':
                    no_nonspeech_index_list.append((start_time, end_time, 0))

                if language == 'Mandarin':
                    no_nonspeech_index_list.append((start_time + 1, end_time - 1, 1))

        return no_nonspeech_index_list

    def merge(self, times):
        if not times:
            return []
        times = sorted(times, key=lambda x: (x[0], x[1]))
        merged = []
        current_start, current_end, current_type = times[0]
        for start, end, interval_type in times[1:]:
            if start <= current_end:
                if current_type == 0 and interval_type == 1:
                    current_end = max(current_end, end)
                    current_type = 1
                elif (current_type == 0 or current_type == 1) and interval_type == 2:
                    merged.append((current_start, start, current_type))
                    current_start, current_end, current_type = start, end, interval_type
                else:
                    current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end, current_type))
                current_start, current_end, current_type = start, end, interval_type
        merged.append((current_start, current_end, current_type))
        return merged


    def build_language_list(self, draft_list, max_time):
        
        result = []

        if draft_list[0][0] > 0: # if first start time greater than 0
            draft_list.insert(0, (0, draft_list[0][0] - 1, 2)) #insert 0 to start time at begining with nonspeech label
        
        if draft_list[-1][1] < max_time: # if final end time smllar than max time
            draft_list.append((draft_list[-1][1] + 1, max_time, 2))
        
        draft_list = self.merge(draft_list)
    
        for index in range(len(draft_list) - 1):
            time_difference = draft_list[index + 1][0] - draft_list[index][1] #calculate time difference between previous end to start
            if time_difference > 1:
                draft_list.append((draft_list[index][1] + 1, draft_list[index + 1][0] - 1, 2)) # append non-speech symbol from end_i + 1 to start_{i+1} -1
        
        draft_list = sorted(draft_list)
    
        return draft_list
    
    def calculate_score(self, language_id):

        pred_file = self.load_file(self.predicting_file)
        ref_file  = self.load_file(self.ground_truth)
        max_time_1 = ref_file[-1][1]
        max_time_2 = pred_file[-1][1]
        max_time = max(max_time_1, max_time_2)
    

        pred_list = self.build_language_list(pred_file, max_time)
        ref_list = self.build_language_list(ref_file, max_time)

        pred_index_list = []
        ref_index_list = []

        for item in pred_list:
            if item[-1] == language_id:
                pred_index_list += [item[-1]] * (item[1] - item[0] + 1)
            else:
                pred_index_list += [2] * (item[1] - item[0] + 1)

        for item in ref_list:
            if item[-1] == language_id:
                ref_index_list += [item[-1]] * (item[1] - item[0] + 1)
            else:
                ref_index_list += [2] * (item[1] - item[0] + 1)

        total_time = 0
        for item in ref_list:
            if item[-1] == language_id:
                total_time += (item[1] - item[0])
        
        total_error = np.sum(np.array(pred_index_list) != np.array(ref_index_list))

        #language_diarization_error_rate = np.sum(np.array(pred_index_list) != np.array(ref_index_list)) / total_time

        return total_error, total_time

    def save_result(self):
        English_id = 0
        Mandarin_id = 1

        English_error, English_time = self.calculate_score(English_id)
        Mandarin_error, Mandarin_time = self.calculate_score(Mandarin_id)
    
        return English_error, English_time, Mandarin_error, Mandarin_time

# def main(args):
#     predicting_folder_path = args['predicting_file'] # should input the predict file folder
#     ground_truth_folder_path = args['ground_truth'] # should input reference file folder
#     result_output_path = args['result_output']
#
#     pred_file_set = os.listdir(predicting_folder_path) # same pair of file should have same file name
#     #true_file_set = os.listdir(ground_truth_folder_path)
#     total_english_error = 0
#     total_english_time = 0
#     total_mandarin_error = 0
#     total_mandarin_time = 0
#
#     for file in pred_file_set:
#         pred_path = os.path.join(predicting_folder_path, file)
#         true_path = os.path.join(ground_truth_folder_path, file)
#
#         LDER = language_diarization_error_rate(pred_path, true_path, result_output_path)
#         English_error, English_time, Mandarin_error, Mandarin_time = LDER.save_result()
#
#         total_english_error += English_error
#         total_english_time += English_time
#         total_mandarin_error += Mandarin_error
#         total_mandarin_time += Mandarin_time
#
#     Total_English_LDER = total_english_error / total_english_time
#     Total_Mandarin_LDER = total_mandarin_error / total_mandarin_time
#     Total_all_LDER = (total_mandarin_error+total_english_error) / (total_english_time+total_mandarin_time)
#
#     with open(os.path.join(result_output_path,'total_result'), 'a+') as f:
#         f.write(f'Total English LDER : {Total_English_LDER}\n')
#         f.write(f'Total Mandarin LDER : {Total_Mandarin_LDER}\n')
#         f.write(f'Total LDER : {Total_all_LDER}\n')
#
# if __name__=='__main__':
#     main(args)
