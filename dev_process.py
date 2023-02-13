import csv
import glob
import os
import soundfile as sf
import librosa
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import argparse

def resample(root, target_sr=16000):
    data, sr = librosa.load(root, sr=None)
    if sr == target_sr:
        print('Do not need resampling for {}'.format(root))
    y_resample = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
    sf.write(root, y_resample, target_sr, subtype='PCM_16')


def mfcc_feat_extraction(audio_list, order_1=True, order_2=True, mfccdim=13, save_txt=None, lang=None):
    """
    :param audio_list: a python dictionary {audio_seg: seg_path}
    :param order_1: path of pre-trained XLSR-53 model, default model/xlsr_53_56k.pt
    :param order_2: from which layer you'd like to extract the wav2vec 2.0 features, default 14
    :param mfccdim: dimension of mfcc dim, default 13 (39 after being stacked with its 1st and 2nd orders)
    :return: a python dictionary {audio_seg: features}
    """
    audio_list = audio_list
    with open(save_txt, 'w') as f:
        for i in tqdm(range(len(audio_list))):
            audio = audio_list[i]
            save_name = audio.replace(".wav",".npy").replace(".flac",".npy")
            try:
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
                f.write(f"{save_name} {lang}\n")
            except:
                print(f"{audio} is too short to extract mfcc feats")

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--groundtruth', type=str, help="path to _MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv")
    parser.add_argument('--save', type=str, help='path to the folder of ground truth and predition rttm')
    parser.add_argument('--audio', type=str, help='path to test audio folder')
    parser.add_argument('--feattxt', type=str, help='path to feats.txt')

    args = parser.parse_args()
    file_path = args.groundtruth
    save_path = args.save
    audio_dir = args.audio
    save_feat_txt = args.feattxt
    audio_list = glob.glob(audio_dir+"*wav")
    print("Resampling dev set to 16KHz")
    for i in tqdm(range(len(audio_list))):
        audio_ = audio_list[i]
        resample(audio_)
    print("Completed resampling")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    audio_seg_list = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header_info = next(reader)
        for row in reader:
            audio_name = row[0]
            chunk_name = row[1]
            start = float(row[2])
            end = float(row[3])
            data_ = AudioSegment.from_file(os.path.join(audio_dir, audio_name))
            data_seg = data_[start:end]
            save_name = f"{save_path}/{audio_name.strip('.wav')}_{chunk_name}.wav"
            data_seg.export(save_name, format='wav')
            if row[5] == "English":
                audio_seg_list[save_name] = 0
            elif row[5] == "Mandarin":
                audio_seg_list[save_name] = 1
    print("Completed segmentation, starting feats extraction")
    with open(save_feat_txt, 'w') as ff:
        for k, v in audio_seg_list.items():
            ff.write(f"{k.replace('.wav','npy')} {v}\n")
    
    mfcc_feat_extraction(list(audio_seg_list.keys()))
    print("Completed processing")


if __name__ == "__main__":
    main()
