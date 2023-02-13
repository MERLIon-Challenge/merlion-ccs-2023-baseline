import csv
import glob
import os
import soundfile as sf
import librosa
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from vad import VoiceActivityDetector
import argparse

def resample(root, target_sr=16000):
    data, sr = librosa.load(root, sr=None)
    if sr == target_sr:
        print('Do not need resampling for {}'.format(root))
    y_resample = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
    sf.write(root, y_resample, target_sr, subtype='PCM_16')


def mfcc_feat_extraction(audio_list, order_1=True, order_2=True, mfccdim=13, save_txt=None):
    """
    :param audio_list: a python dictionary {audio_seg: seg_path}
    :param order_1: path of pre-trained XLSR-53 model, default model/xlsr_53_56k.pt
    :param order_2: from which layer you'd like to extract the wav2vec 2.0 features, default 14
    :param mfccdim: dimension of mfcc dim, default 13 (39 after being stacked with its 1st and 2nd orders)
    :return: a python dictionary {audio_seg: features}
    """
    with open(save_txt, 'w') as fwrite:
        for i in tqdm(range(len(audio_list))):
            audio = audio_list[i]
            save_name = audio.replace(".wav",".npy").replace(".flac",".npy")
            audioarray, sr_ = librosa.load(path=audio, sr=None)
            preemphasis = 0.97
            preemphasized = np.append(audioarray[0], audioarray[1:] - preemphasis * audioarray[:-1])
            mfcc = librosa.feature.mfcc(y = preemphasized, sr = sr_, n_mfcc=mfccdim,
                                        hop_length=int(sr_ / 100), n_fft=int(sr_ / 40))
            if order_1 and order_2:
                delta1 = librosa.feature.delta(mfcc, order=1)
                delta2 = librosa.feature.delta(mfcc, order=2)
                mfcc_features = np.vstack((mfcc, delta1, delta2))
            else:
                mfcc_features = mfcc
            np.save(save_name, mfcc_features)
            fwrite.write(f"{save_name} 0\n")

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--groundtruth', type=str, help="path to _MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv")
    parser.add_argument('--save_g', type=str, help='path to the folder to save ground truth rttm')
    parser.add_argument('--save_p', type=str, help='path to the folder to save predition rttm')
    parser.add_argument('--save_f', type=str, help='path to the folder to save chunks & feats')
    parser.add_argument('--audio', type=str, help='path to the folder to save chunks & feats')
    args = parser.parse_args()
    
    file_path = args.groundtruth
    save_g_path = args.save_g
    save_p_path = args.save_p
    save_f_path = args.save_f
    audio_dir = args.audio
    
    audio_list = glob.glob(audio_dir + "*wav")
    print("Resampling dev set to 16KHz")
    for i in tqdm(range(len(audio_list))):
        audio_ = audio_list[i]
        resample(audio_)
    print("Completed resampling")
    if not os.path.exists(save_g_path):
        os.mkdir(save_g_path)
    if not os.path.exists(save_p_path):
        os.mkdir(save_p_path)
    for audio_ in audio_list:
        audio_rttm_path = os.path.join(save_g_path, os.path.split(audio_)[-1].replace('.wav', '_ground_truth_rttm.txt') )
        with open(audio_rttm_path, 'w') as fw:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header_info = next(reader)
                for row in reader:
                    audio_name = row[0]
                    if os.path.split(audio_)[-1] == audio_name:
                        start = float(row[2])
                        end = float(row[3])
                        fw.write(f"{start} {end} {row[5]}\n")
    
            #VAD + Segmentation
            predict_rttm_path = os.path.join(save_p_path, os.path.split(audio_)[-1].replace('.wav', '_predict_rttm.txt'))
            save_feat_txt = os.path.join(save_f_path, os.path.split(audio_)[-1].replace('.wav', '_feats.txt'))
            v = VoiceActivityDetector(audio_)
            raw_dectection = v.detect_speech()
            speech_labels = v.convert_windows_to_readible_labels(raw_dectection)
            data_ = AudioSegment.from_file(audio_)
            audio_seg_list = []
            with open(predict_rttm_path, 'w') as fff:
                for ind_ in tqdm(range(len(speech_labels))):
                    start = float(speech_labels[ind_]['speech_begin'])*1000
                    end = float(speech_labels[ind_]['speech_end'])*1000
                    if end - start >= 500:
                        data_seg = data_[start:end]
                        save_name = f"{save_f_path}/{audio_name.strip('.wav')}_{ind_}.wav"
                        data_seg.export(save_name, format='wav')
                        fff.write(f"{start} {end} unknown\n")
                        audio_seg_list.append(save_name)
            mfcc_feat_extraction(audio_seg_list, save_txt=save_feat_txt)


if __name__ == "__main__":
    main()
