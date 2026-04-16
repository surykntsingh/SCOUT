import json
import os
import h5py
import torch
from torch.utils.data import Dataset

from utils.utils import read_json_file

class EmbeddingDataset(Dataset):

    def __init__(self, embeddings_path, reports_json_path, tokenizer, max_seq_length,
                 embeddings_path_2, gecko_emb_path, dataset_type):
        reports = read_json_file(reports_json_path)[dataset_type]
        self.__reports = {report['id'].split('.')[0] : report['report'] for report in reports}
        self.__tokenizer = tokenizer
        self.__embeddings_path = embeddings_path
        self.__max_seq_length = max_seq_length
        self.__embeddings_path_2 = embeddings_path_2
        self.__gecko_emb_path = gecko_emb_path

        files = os.listdir(embeddings_path)
        files_1 = os.listdir(embeddings_path_2)
        files_2 = os.listdir(gecko_emb_path)
        # slides = self.__reports.keys()

        # self.__slides = [file.split('.')[0] for file in files if file in files_1 and file in files_2 and file.split('.')[0] in slides]
        slides = list(self.__reports.keys())
        # slides = [slide for slide in self.__reports.keys() if slide != 'TCGA-A2-A1G0-01Z-00-DX1.9ECB0B8A-EF4E-45A9-82AC-EF36375DEF65']
        # print(f'slides: {slides}')

        # self.__slides = [file.split('.')[0] for file in self.__reports.keys()]
        print(f'dataset_type: {dataset_type} self.__slides: {len(slides)}')
        self.__slides = [file.split('.')[0] for file in files if
                         file in files_1 and f'{file[:12]}.h5' in files_2 and file.split('.')[0] in self.__slides]
        # self.__slides = [slide for slide in slides if f'{slide}.h5' in files and f'{slide}.h5' in files_1 and f'{slide[:12]}.h5' in files_2]

        # self.__slides = ['.'.join(file.split('.')[:-1]) for file in files if
        #                  file in files_1 and f'{file[:12]}.h5' in files_2 and '.'.join(file.split('.')[:-1]) in slides]
        print(f'dataset_type: {dataset_type}, files: {len(files)}, files_1: {len(files_1)}, files_2: {len(files_2)} slides: {len(self.__slides)}')

        print(f'dataset_type: {dataset_type}, files: {files[:4]}, files_1: {files_1[:4]}, files_2: {files_2[:4]} slides: {self.__slides[:4]}')

        # print(f'self.__reports: {self.__reports}')

    def __len__(self):
        return len(self.__slides)

    def __getitem__(self, idx):
        slide_id = self.__slides[idx]
        with h5py.File(f'{self.__embeddings_path}/{slide_id}.h5', "r") as h5_file:
            # coords_np = h5_file["coords"][:]
            embeddings_np = h5_file["features"][:]

            # coords = torch.tensor(coords_np).float()
            slide_embedding = torch.tensor(embeddings_np)
            report_text = self.__reports[slide_id]
            # print(f'report_text: {report_text}')
            report_ids = self.__tokenizer(report_text)

            if len(report_ids) < self.__max_seq_length:
                padding = [0] * (self.__max_seq_length-len(report_ids))
                report_ids.extend(padding)

            report_masks = [1] * len(report_ids)
            # seq_length = len(report_ids)

        with h5py.File(f'{self.__embeddings_path_2}/{slide_id}.h5', "r") as h5_file:
            embeddings_np = h5_file["features"][:]
            patch_embedding = torch.tensor(embeddings_np)

        with h5py.File(f'{self.__gecko_emb_path}/{slide_id[:12]}.h5', "r") as h5_file:
            # coords_np = h5_file["coords"][:]
            bag_feats_deep_np = h5_file["bag_feats_deep"][:]
            bag_feats_np = h5_file["bag_feats"][:]
            # bag_feat_attn_np = h5_file["attention_bag_feats"][:]

            gecko_deep_embedding = torch.tensor(bag_feats_deep_np)
            gecko_concept_embedding = torch.tensor(bag_feats_np)
            # attn_gc = torch.tensor(bag_feat_attn_np)

        return slide_id, slide_embedding, patch_embedding, gecko_deep_embedding, gecko_concept_embedding, report_ids, report_masks


class EmbeddingPredictDataset(Dataset):

    def __init__(self, embeddings_path, tokenizer, max_seq_length, embeddings_path_2, gecko_emb_path, slide_ids=None):
        # reports = read_json_file(reports_json_path)
        # self.__reports = {report['id'].split('.')[0]: report['report'] for report in reports}
        self.__tokenizer = tokenizer
        self.__embeddings_path = embeddings_path
        self.__max_seq_length = max_seq_length
        self.__embeddings_path_2 = embeddings_path_2
        self.__gecko_emb_path = gecko_emb_path

        files = os.listdir(embeddings_path)
        files_1 = os.listdir(embeddings_path)

        self.__slides = [file.split('.')[0] for file in files if file in files_1]
        print(f'slide_ids:: {slide_ids} self.__slides: {self.__slides}')
        if slide_ids:
            self.__slides = [slide_id for slide_id in slide_ids if slide_id in self.__slides]

        print(f'Number of slides: {len(self.__slides)}')

    def __len__(self):
        return len(self.__slides)

    def __getitem__(self, idx):
        slide_id = self.__slides[idx]
        with h5py.File(f'{self.__embeddings_path}/{slide_id}.h5', "r") as h5_file:
            # coords_np = h5_file["coords"][:]
            embeddings_np = h5_file["features"][:]

            # coords = torch.tensor(coords_np).float()
            slide_embedding = torch.tensor(embeddings_np)

        with h5py.File(f'{self.__embeddings_path_2}/{slide_id}.h5', "r") as h5_file:
            embeddings_np = h5_file["features"][:]
            patch_embeddings = torch.tensor(embeddings_np)

        with h5py.File(f'{self.__gecko_emb_path}/{slide_id}.h5', "r") as h5_file:
            # coords_np = h5_file["coords"][:]
            bag_feats_deep_np = h5_file["bag_feats_deep"][:]
            bag_feats_np = h5_file["bag_feats"][:]
            # bag_feat_attn_np = h5_file["attention_bag_feats"][:]

            gecko_deep_embedding = torch.tensor(bag_feats_deep_np)
            gecko_concept_embedding = torch.tensor(bag_feats_np)

        return slide_id, slide_embedding, patch_embeddings, gecko_deep_embedding, gecko_concept_embedding







