import os
import torch
import glob
import argparse
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from model import *
from data_load import *
from scoring_diar import language_diarization_error_rate

def main():
    parser = argparse.ArgumentParser(description='paras for diar validation')
    parser.add_argument('--model', type=str, help='model path',
                        default='conformer.ckpt')
    parser.add_argument('--save', type=str, help='path to the folder of ground truth and predition rttm')
    parser.add_argument('--audio', type=str, help='path to test audio folder')
    parser.add_argument('--device', type=int, help='Device name', default=0)

    args = parser.parse_args()


    model = Conformer(input_dim=39,
                      feat_dim=64,
                      d_k=64,
                      d_v=64,
                      n_heads=8,
                      d_ff=2048,
                      max_len=100000,
                      dropout=0.1,
                      device=0,
                      n_lang=2)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pretrained_dict = torch.load(args.model, map_location=device)
    new_state_dict = OrderedDict()
    model_dict = model.state_dict()
    dict_list = []
    for k, v in model_dict.items():
        dict_list.append(k)
    for k, v in pretrained_dict.items():
        if k.startswith('module.') and k[7:] in dict_list:
            new_state_dict[k[7:]] = v
        elif k in dict_list:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    save_path = args.save
    audio_dir = args.audio
    audio_list = glob.glob(audio_dir + "*wav")
    output_path = 'diarization_error_rate.txt'

    total_english_error = 0
    total_english_time = 0
    total_mandarin_error = 0
    total_mandarin_time = 0

    for audio_ in audio_list:
        print(f"Audio name: {os.path.split(audio_)[-1]}")
        audio_rttm_path = os.path.join(save_path, os.path.split(audio_)[-1].replace('.wav', '_ground_truth_rttm.txt'))
        predict_rttm_path = os.path.join(save_path, os.path.split(audio_)[-1].replace('.wav', '_predict_rttm.txt'))
        save_feat_txt = os.path.join(save_path, os.path.split(audio_)[-1].replace('.wav', '_feats.txt'))
        valid_set = RawFeatures(save_feat_txt)
        valid_data = DataLoader(dataset=valid_set,
                                batch_size=1,
                                pin_memory=True,
                                shuffle=False,
                                collate_fn=collate_fn_atten)
        model.eval()
        with open(predict_rttm_path, 'r') as f:
            lines = f.readlines()
        with open(predict_rttm_path, 'w') as f:
            with torch.no_grad():
                for step, (utt, labels, seq_len) in enumerate(valid_data):
                    utt = utt.to(device=device, dtype=torch.float)
                    labels = labels.to(device)
                    atten_mask = get_atten_mask(seq_len, utt.size(0))
                    atten_mask = atten_mask.to(device=device)
                    # Forward pass
                    outputs = model(utt, atten_mask)
                    predicted = int(torch.argmax(outputs, -1).squeeze().cpu().numpy())
                    if predicted == 0:
                        lang_out = "English"
                    else:
                        lang_out = "Mandarin"
                    f.write(lines[step].replace("unknown", f'{lang_out}'))
        LDER = language_diarization_error_rate(predict_rttm_path, audio_rttm_path, output_path)
        English_error, English_time, Mandarin_error, Mandarin_time = LDER.save_result()

        total_english_error += English_error
        total_english_time += English_time
        total_mandarin_error += Mandarin_error
        total_mandarin_time += Mandarin_time

    Total_English_LDER = total_english_error / total_english_time
    Total_Mandarin_LDER = total_mandarin_error / total_mandarin_time

    with open(output_path, 'a+') as f:
        f.write('Total English LDER :' + " " + str(Total_English_LDER) + '\n')
        f.write('Total Mandarin LDER :' + " " + str(Total_Mandarin_LDER) + '\n')
    print(f"Eng LDER: {Total_English_LDER} Man LDER: {Total_Mandarin_LDER}")



if __name__ == "__main__":
    main()




