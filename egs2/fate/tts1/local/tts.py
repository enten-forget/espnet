from tkinter import N
from espnet2.bin.tts_inference import Text2Speech
import torch
import time

# Load the model

text2speech = Text2Speech(
    train_config="exp/22k/tts_train_vits_raw_phn_jaconv_pyopenjtalk_use_wandbtrue_wandb_projectfate_wandb_namevits_train_saber_use_tensorboardFalse/config.yaml",
    model_file="exp/22k/tts_train_vits_raw_phn_jaconv_pyopenjtalk_use_wandbtrue_wandb_projectfate_wandb_namevits_train_saber_use_tensorboardFalse/latest.pth",
    device='cuda'
)

texts = [
   '問おう。あなたがわたしのマスターか？'
]

wavs = None

# synthesis
start = time.time()
with torch.no_grad():
    for text in texts:
        wav = text2speech(text)['wav']
        # print(wav.shape)
        if wavs is None:
            wavs = wav
        else:
            wavs = torch.cat((wavs, wav))
        silence = torch.zeros(int(text2speech.fs*0.1)).to(wavs.device)
        wavs = torch.cat((wavs, silence))
print(f"synthesis time: {time.time()-start}s")
print(f'wav time: {wavs.shape[0]/text2speech.fs}s')
# save
import soundfile as sf
sf.write(f'saber.wav',wavs.cpu().numpy(), text2speech.fs,"PCM_16")