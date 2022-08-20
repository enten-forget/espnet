import argparse

def generate_espnet_dataset_from_filelists(filelist, output_dir):
    """Generate ESPnet dataset from filelists.

    Args:
        filelist (str): filelist path
        output_dir (str): Output
    """
    import os
    from pathlib import Path
    from shutil import copyfile
    from tqdm import tqdm
    import random

    root_output_dir = Path(output_dir)

    # Read filelist
    with open(filelist, "r") as f:
        wav_map_texts = []
        for line in f:
            wav_map_texts.append(line.strip().split("|"))
    
    # replace  \u3000 to \u0020 in text in wav_map_texts
    for wav_map_text in wav_map_texts:
        wav_map_text[1] = wav_map_text[1].replace("\u3000", "\u0020")


    # Create train split output_dir directory
    for split in ['train', 'dev', 'test']:
        output_dir = root_output_dir / split
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if split == 'train':
            wav_map_texts_t = wav_map_texts
        else:
            wav_map_texts_t = random.choices(wav_map_texts, k=100)

        # generate wav.scp 
        wav_scp = os.path.join(output_dir, "wav.scp")
        with open(wav_scp, "w") as f:
            for wav_map_text in tqdm(wav_map_texts_t):
                uttid = wav_map_text[0].split("/")[-1].split(".")[0]
                f.write("{} {}\n".format(uttid, wav_map_text[0]))
        
        # generate text
        text = os.path.join(output_dir, "text")
        with open(text, "w") as f:
            for wav_map_text in tqdm(wav_map_texts_t):
                uttid = wav_map_text[0].split("/")[-1].split(".")[0]
                f.write("{} {}\n".format(uttid, wav_map_text[1]))
        
        # generate utt2spk
        utt2spk = os.path.join(output_dir, "utt2spk")
        with open(utt2spk, "w") as f:
            for wav_map_text in tqdm(wav_map_texts_t):
                uttid = wav_map_text[0].split("/")[-1].split(".")[0]
                f.write("{} {}\n".format(uttid, uttid[:3]))
        
        # generate spk2utt
        spk2utt = os.path.join(output_dir, "spk2utt")
        with open(spk2utt, "w") as f:
            for wav_map_text in tqdm(wav_map_texts_t):
                uttid = wav_map_text[0].split("/")[-1].split(".")[0]
                f.write("{} {}\n".format(uttid[:3], uttid))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    generate_espnet_dataset_from_filelists(args.filelist, args.output_dir)