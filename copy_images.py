import os
import json
import pandas as pd
from tqdm.auto import tqdm
import argparse
import shutil

def read_json_file(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d

def copy_images(args):

    pathtext = read_json_file(args.pathtext_json_path)
    pbar = tqdm(pathtext, total=len(pathtext))
    mags = set()
    for i, p in enumerate(pbar):
        mag = p['paths']['DX']['mag']
        if mag not in mags:
            os.makedirs(f'{args.target_dir}/{mag}', exist_ok=True)
        destination_path = os.path.join(args.target_dir, f'{mag}/{p["id"]}.svs')
        shutil.copy(p['paths']['DX']['file_path'], destination_path)
    #

    print(mags)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pathtext_json_path', type=str,
                        help='pathtext_json_path')
    parser.add_argument('--target_dir', type=str,
                        help='target_dir')
    args = parser.parse_args()
    copy_images(args)
    print('Finished!')