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

def get_file_paths(patient_id, tcga_dir):
    slide_paths={}
    for root, _, files in os.walk(tcga_dir):
        for file in files:
            if file.endswith(".svs") and patient_id in file:
                full_path = os.path.join(root, file)
                slide_type = file.split('.')[0].split('-')[-1][:2]
                try:
                    slide = openslide.OpenSlide(full_path)
                    magnification = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "Unknown")
                    slide_paths[slide_type] = {'file_name': file, 'file_path': full_path, 'mag': magnification}
                    slide.close()
                except Exception as e:
                    print(f'full_path: {full_path} e: {e}')
                    slide_paths[slide_type] = {'file_name': file, 'file_path': full_path, 'mag': 'Unknown'}
    return slide_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pathtext_json_path', type=str,
                        help='pathtext_json_path')
    parser.add_argument('--target_dir', type=str,
                        help='target_dir')
    args = parser.parse_args()
    reports_data = []

    pathtext = read_json_file(args.pathtext_json_path)
    pbar = tqdm(pathtext, total=len(pathtext))
    mags = set()
    for i, p in enumerate(pbar):
        mag = p['paths']['DX']['mag']
        mags.add(mag)
        # os.makedirs(f'{args.ckpt_path}/results', exist_ok=True)


    print(mags)
    print('Finished!')