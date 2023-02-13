import glob
import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import argparse

def write_pcm16(root):
    data, sr = librosa.load(root, sr=None)
    sf.write(root, data, sr, subtype='PCM_16')

def audio_segment(audio, save_dir, segment_len=3.0, overlap=0.5, min_len=1.0):
    data, sr = librosa.load(audio, sr=16000)
    audio_len = len(data) / sr
    file_type = audio.split(".")[-1]

    num_segs = int((audio_len - overlap) // (segment_len - overlap))
    if num_segs >= 1:
        start = 0
        speech_data = AudioSegment.from_file(audio)
        for ind in range(num_segs):
            end = start + segment_len
            speech_seg = speech_data[int(start) * 1000:int(end) * 1000]
            save_name = save_dir + os.path.split(audio)[-1].replace(f'.{file_type}', f'_{ind}.{file_type}')
            speech_seg.export(save_name)
            write_pcm16(save_name)
            start = end - overlap
        rest_len = audio_len - overlap - num_segs * (segment_len - overlap)
        if rest_len >= min_len:
            speech_seg = speech_data[(start + overlap - (segment_len - rest_len)) * 1000:]
            save_name = save_dir + os.path.split(audio)[-1].replace(f'.{file_type}', f'_final.{file_type}')
            speech_seg.export(save_name)
            write_pcm16(save_name)
#
#
#
def mfcc_feat_extraction(audio_txt, order_1=True, order_2=True, mfccdim=13, save_txt=None):
    """
    :param audio_list: a python dictionary {audio_seg: seg_path}
    :param order_1: path of pre-trained XLSR-53 model, default model/xlsr_53_56k.pt
    :param order_2: from which layer you'd like to extract the wav2vec 2.0 features, default 14
    :param mfccdim: dimension of mfcc dim, default 13 (39 after being stacked with its 1st and 2nd orders)
    :return: a python dictionary {audio_seg: features}
    """
    with open(audio_txt, 'r') as fr:
        lines = fr.readlines()
    audio_list = [x.split()[0] for x in lines]
    lang_list = [x.split()[1].strip() for x in lines]
    lang = None
    with open(save_txt, 'w') as f:
        for i in tqdm(range(len(audio_list))):
            audio = audio_list[i]
            if lang_list[i] == 'English':
                lang = 0
            else:
                lang = 1
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
    parser.add_argument('--librispeech', type=str, help="path to librispeech")
    parser.add_argument('--aishell', type=str, help='path to aishell')
    parser.add_argument('--nsc', type=str, help='path to nsc')
    parser.add_argument('--save_audio', type=str, help='path to save audio chunks')
    parser.add_argument('--save_feat', type=str, help='path to save feats')
    parser.add_argument('--audio_txt', type=str, help='audio chunk txt')
    parser.add_argument('--feat_txt', type=str, help='feat txt')
    args = parser.parse_args()
    
    librispeech_path = args.librispeech
    aishell_path = args.aishell
    nsc_path = args.nsc
    save_dir = args.save_audio
    save_feat_dir = args.save_feat
    save_txt = "/home/hexin/Desktop/hexin/dataset/challenge_train.txt"
    save_feat_txt = "/home/hexin/Desktop/hexin/dataset/challenge_feat_train.txt"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    librispeech_list = glob.glob(librispeech_path+"/**/**/*.flac")
    aishell_list = glob.glob(aishell_path+"/**/*.wav")
    nsc_list = glob.glob(nsc_path+"/*.wav")
    segment_len = 3.0
    overlap = 0.5
    min_len = 1.0
    with open(save_txt, "w") as f:
        for i in tqdm(range(len(nsc_list))):
                audio = nsc_list[i]
                data, sr = librosa.load(audio, sr=16000)
                audio_len = len(data) / sr
                file_type = audio.split(".")[-1]
                num_segs = int((audio_len - overlap) // (segment_len - overlap))
                if num_segs >= 1:
                    start = 0
                    speech_data = AudioSegment.from_file(audio)
                    for ind in range(num_segs):
                        end = start + segment_len
                        speech_seg = speech_data[int(start) * 1000:int(end) * 1000]
                        save_name = save_dir + os.path.split(audio)[-1].replace(f'.{file_type}', f'_{ind}.{file_type}')
                        speech_seg.export(save_name)
                        write_pcm16(save_name)
                        start = end - overlap
                        f.write(f"{save_name} English\n")
                    rest_len = audio_len - overlap - num_segs * (segment_len - overlap)
                    if rest_len >= min_len:
                        speech_seg = speech_data[(start + overlap - (segment_len - rest_len)) * 1000:]
                        save_name = save_dir + os.path.split(audio)[-1].replace(f'.{file_type}', f'_final.{file_type}')
                        speech_seg.export(save_name)
                        write_pcm16(save_name)
                        f.write(f"{save_name} English\n")

        for i in tqdm(range(len(librispeech_list))):
            audio = librispeech_list[i]
            data, sr = librosa.load(audio, sr=16000)
            audio_len = len(data) / sr
            file_type = audio.split(".")[-1]
            num_segs = int((audio_len - overlap) // (segment_len - overlap))
            if num_segs >= 1:
                start = 0
                speech_data = AudioSegment.from_file(audio)
                for ind in range(num_segs):
                    end = start + segment_len
                    speech_seg = speech_data[int(start) * 1000:int(end) * 1000]
                    save_name = save_dir + os.path.split(audio)[-1].replace(f'.{file_type}', f'_{ind}.{file_type}')
                    speech_seg.export(save_name)
                    write_pcm16(save_name)
                    start = end - overlap
                    f.write(f"{save_name} English\n")
                rest_len = audio_len - overlap - num_segs * (segment_len - overlap)
                if rest_len >= min_len:
                    speech_seg = speech_data[(start + overlap - (segment_len - rest_len)) * 1000:]
                    save_name = save_dir + os.path.split(audio)[-1].replace(f'.{file_type}', f'_final.{file_type}')
                    speech_seg.export(save_name)
                    write_pcm16(save_name)
                    f.write(f"{save_name} English\n")

        for i in tqdm(range(len(aishell_list))):
            audio = aishell_list[i]
            data, sr = librosa.load(audio, sr=16000)
            audio_len = len(data) / sr
            file_type = audio.split(".")[-1]
            num_segs = int((audio_len - overlap) // (segment_len - overlap))
            if num_segs >= 1:
                start = 0
                speech_data = AudioSegment.from_file(audio)
                for ind in range(num_segs):
                    end = start + segment_len
                    speech_seg = speech_data[int(start) * 1000:int(end) * 1000]
                    save_name = save_dir + os.path.split(audio)[-1].replace(f'.{file_type}', f'_{ind}.{file_type}')
                    speech_seg.export(save_name)
                    write_pcm16(save_name)
                    start = end - overlap
                    f.write(f"{save_name} Mandarin\n")
                rest_len = audio_len - overlap - num_segs * (segment_len - overlap)
                if rest_len >= min_len:
                    speech_seg = speech_data[(start + overlap - (segment_len - rest_len)) * 1000:]
                    save_name = save_dir + os.path.split(audio)[-1].replace(f'.{file_type}', f'_final.{file_type}')
                    speech_seg.export(save_name)
                    write_pcm16(save_name)
                    f.write(f"{save_name} Mandarin\n")

    mfcc_feat_extraction(save_txt, save_txt=save_feat_txt)



if __name__ == "__main__":
    main()
