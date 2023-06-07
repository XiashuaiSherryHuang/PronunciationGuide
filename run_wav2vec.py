'''This file is to use wav2vec2 to get ARPAbet from audio file'''
# !pip install g2p_en
# #https://github.com/Kyubyong/g2p
# !pip install torch
# !pip install transformers
# #https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self
# #https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20
# !pip install librosa

from g2p_en import G2p
# For managing audio file
import librosa
## Grapheme To Phoneme Conversion
import torch
#Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
import os


def get_arpabet_from_audio(file):
    # Importing Wav2Vec pretrained model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h") #, "facebook/wav2vec2-large-960h-lv60-self"
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") #"facebook/wav2vec2-large-960h-lv60-self"

    sr = 16000
    audio, rate = librosa.load(file, sr = sr)
    # Taking an input value
    input_values = processor(audio, sampling_rate=rate, return_tensors = "pt").input_values

    # INFERENCE
    # Storing logits (non-normalized prediction values)
    logits = model(input_values).logits
    pred_text = processor.batch_decode(torch.argmax(logits, dim=-1), clean_up_tokenization_spaces=True)[0]
    print('Predicted text from audio is {}'.format(pred_text))

    g2p = G2p()
    pred_p = ''.join(g2p(pred_text))
    print('Processed arpabet is {}'.format(pred_p))
    return pred_p

def write_text_to_file(filename, text):
    try:
        with open(filename, 'w') as file:
            file.write(text)
    except Exception:
        print(f'An error occurred while writing to {filename}.')


if __name__ == '__main__':
    wav_directory = 'wav/'
    output_directory = 'wav2arpa/'
    print(os.listdir(wav_directory)[:10])
    ###3 files failed to do inference with wav2vec2
    error_wav_files = ['arabic42.wav', 'somali3.wav', 'quechua2.wav']
    existing_files = os.listdir(output_directory)
    for file_name in os.listdir(wav_directory):
        if file_name.endswith('.wav'):
            try:
                print(file_name)
                if file_name in error_wav_files:
                    continue
                output_file = file_name.split('.')[0] + '.txt'
                if output_file in existing_files:
                    continue
                wav_file_path = os.path.join(wav_directory, file_name)
                pred_p = get_arpabet_from_audio(wav_file_path)
                output_path = os.path.join(output_directory, output_file)
                print(output_path)
                write_text_to_file(output_path, pred_p)
            except Exception as e:
                print(f'An error occurred while processing {file_name} {str(e)}.')