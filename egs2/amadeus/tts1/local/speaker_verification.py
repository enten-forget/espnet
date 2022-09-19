import torch
import pydub
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EFFECTS = [
    ["remix", "-"],
    ["channels", "1"],
    ["rate", "16000"],
    ["gain", "-1.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
    ["trim", "0", "10"],
]

def load_audio(file_name):
    audio = pydub.AudioSegment.from_file(file_name)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
    arr = arr / (1 << (8 * audio.sample_width - 1))
    return arr.astype(np.float32), audio.frame_rate

THRESHOLD = 0.85

model_name = "microsoft/wavlm-base-plus-sv"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioXVector.from_pretrained(model_name).to(device)
cosine_sim = torch.nn.CosineSimilarity(dim=-1)
# audio files are decoded on the fly
# change your wav file path to path1 and path2
path1 = ''
path2 = ''
wav1, sr1 = load_audio(path1)

wav1, _ = apply_effects_tensor(torch.tensor(wav1).unsqueeze(0), sr1, EFFECTS)
wav2, sr2 = load_audio(path2)
wav2, _ = apply_effects_tensor(torch.tensor(wav2).unsqueeze(0), sr2, EFFECTS)


input1 = feature_extractor(wav1.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)
input2 = feature_extractor(wav2.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values.to(device)

with torch.no_grad():
    emb1 = model(input1).embeddings
    emb2 = model(input2).embeddings
emb1 = torch.nn.functional.normalize(emb1, dim=-1).cpu()
emb2 = torch.nn.functional.normalize(emb2, dim=-1).cpu()
similarity = cosine_sim(emb1, emb2).numpy()[0]
print(similarity)
