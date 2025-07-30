import json
import os
import h5py
import torch
from torch.utils.data import Dataset

from utils.utils import read_json_file

class EmbeddingDataset(Dataset):

    def __init__(self, embeddings_path, reports_json_path, tokenizer, max_seq_length, embeddings_path_2):
        reports = read_json_file(reports_json_path)
        self.__reports = {report['id'].split('.')[0]: report['report'] for report in reports}
        self.__tokenizer = tokenizer
        self.__embeddings_path = embeddings_path
        self.__max_seq_length = max_seq_length
        self.__embeddings_path_2 = embeddings_path_2

        files = os.listdir(embeddings_path)
        files_1 = os.listdir(embeddings_path)
        self.__slides = [file.split('.')[0] for file in files if file in files_1]

        print(f'files: {len(files)}, files_1: {len(files_1)}')

    def __len__(self):
        return len(self.__slides)

    def __getitem__(self, idx):
        slide_id = self.__slides[idx]
        with h5py.File(f'{self.__embeddings_path}/{slide_id}.h5', "r") as h5_file:
            # coords_np = h5_file["coords"][:]
            bag_feats_deep_np = h5_file["bag_feats_deep"][:]
            bag_feats_np = h5_file["bag_feats"][:]


            # coords = torch.tensor(coords_np).float()
            embedding2 = torch.tensor(bag_feats_deep_np)
            embedding1 = torch.tensor(bag_feats_np)
            report_text = self.__reports[slide_id]
            report_ids = self.__tokenizer(report_text)

            if len(report_ids) < self.__max_seq_length:
                padding = [0] * (self.__max_seq_length-len(report_ids))
                report_ids.extend(padding)

            report_masks = [1] * len(report_ids)
            seq_length = len(report_ids)




        return slide_id, embedding1.unsqueeze(0), embedding2.unsqueeze(0), report_ids, report_masks, seq_length










