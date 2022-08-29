#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=22050
n_fft=1024
n_shift=256
win_length=null

opts=
if [ "${fs}" -eq 22050 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

wav_filelist=/gds/jcao/code/Amadeus/data/Amadeus/CRS.txt
train_set=train
valid_set=dev
test_sets=test

train_config=conf/tuning/finetune_vits.yaml
inference_config=conf/decode.yaml


# Input example: こ、こんにちは

# 1. Phoneme + Pause
# (e.g. k o pau k o N n i ch i w a)
# g2p=pyopenjtalk

# 2. Kana + Symbol
# (e.g. コ 、 コ ン ニ チ ワ)
# g2p=pyopenjtalk_kana

# 3. Phoneme + Accent
# (e.g. k 1 0 o 1 0 k 5 -4 o 5 -4 N 5 -3 n 5 -2 i 5 -2 ch 5 -1 i 5 -1 w 5 0 a 5 0)
# g2p=pyopenjtalk_accent

# 4. Phoneme + Accent + Pause
# (e.g. k 1 0 o 1 0 pau k 5 -4 o 5 -4 N 5 -3 n 5 -2 i 5 -2 ch 5 -1 i 5 -1 w 5 0 a 5 0)
g2p=pyopenjtalk_accent_with_pause

# 5. Phoneme + Prosody symbols
# (e.g. ^, k, #, o, _, k, o, [, N, n, i, ch, i, w, a, $)
# g2p=pyopenjtalk_prosody

./tts.sh \
    --ngpu 1 \
    --stage 6 \
    --min_wav_duration 0.38 \
    --dumpdir dump/22k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_args "--init_param downloads/f3698edf589206588f58f5ec837fa516/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.total_count.ave_10best.pth:tts:tts --use_wandb true --wandb_project amadeus" \
    --tag amadeus_vits_finetune_from_jsut_32_sentence \
    --lang jp \
    --local_data_opts "${wav_filelist}" \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner jaconv \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
