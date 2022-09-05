#!/usr/bin/env bash

# set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=1

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 1 ]; then
    log "Error: No positional arguments are passed."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${Fate}" ]; then
   log "Fill the value of 'Fate' of db.sh"
   exit 1
fi
db_root=${Fate}

train_set=train
train_dev=dev
recog_set=test


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Prepare Data"
    python local/data_preprocess.py --filelist $1 --output_dir data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: validate and fix data"
    utils/validate_data_dir.sh --no-feats data/train
    utils/validate_data_dir.sh --no-feats data/dev
    utils/validate_data_dir.sh --no-feats data/test
    utils/fix_data_dir.sh data/train
    utils/fix_data_dir.sh data/dev
    utils/fix_data_dir.sh data/test
    rm -rf dump
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
