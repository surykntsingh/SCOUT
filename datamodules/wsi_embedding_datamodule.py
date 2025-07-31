import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from embedding_datasets.embedding_dataset import EmbeddingDataset, EmbeddingPredictDataset


class PatchEmbeddingDataModule(pl.LightningDataModule):

    def __init__(self,args, tokenizer, split_frac, shuffle = False):
        super().__init__()
        self.test_ds = None
        self.val_ds = None
        self.train_ds = None
        self.__batch_size = args.batch_size
        self.__shuffle = shuffle
        self.__num_workers = args.num_workers
        self.__embeddings_path = args.embeddings_path
        self.__reports_json_path = args.reports_json_path
        self.__max_seq_length = args.max_seq_length
        self.__split_frac = split_frac
        self.__tokenizer = tokenizer
        self.__embeddings_path_2 = args.embeddings_path_2

    def setup(self, stage=None):
        dataset = EmbeddingDataset(self.__embeddings_path, self.__reports_json_path, self.__tokenizer,
                              self.__max_seq_length, self.__embeddings_path_2)
        # print(dataset[0][1])
        self.train_ds, self.val_ds, self.test_ds = random_split(dataset, self.__split_frac)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.__batch_size, shuffle=self.__shuffle, collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.__batch_size, collate_fn = self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.__batch_size, collate_fn = self.collate_fn)

    @staticmethod
    def collate_fn(batch, device='cuda'):
        slide_ids, patch_feats_1,patch_feats_2, coord_feats, report_ids, report_masks, seq_length = zip(*batch)
        patch_feats1_pad = pad_sequence(patch_feats_1, batch_first=True).to(device)
        patch_feats2_pad = pad_sequence(patch_feats_2, batch_first=True).to(device)
        report_ids = torch.LongTensor(report_ids).to(device)
        # dummy_feat = torch.randn(patch_feats_pad.shape)
        # coord_feats_pad =  pad_sequence(coord_feats, batch_first=True)
        # patch_mask = torch.zeros(patch_feats_pad.shape[:2], dtype=torch.float32)
        # for i, p in enumerate(patch_feats):
        #     patch_mask[i, :p.shape[0]] = 1
        # print(slide_ids, len(patch_feats1), len(patch_feats2))
        return (slide_ids, patch_feats1_pad, patch_feats2_pad, report_ids,
                torch.FloatTensor(report_masks), seq_length)



class PatchEmbeddingDataPredictModule(pl.LightningDataModule):

    def __init__(self,args, tokenizer, shuffle = False, slide_ids=None):
        super().__init__()
        self.predict_ds = None
        self.__batch_size = args.batch_size
        self.__shuffle = shuffle
        self.__num_workers = args.num_workers
        self.__embeddings_path = args.embeddings_path
        self.__reports_json_path = args.reports_json_path
        self.__max_seq_length = args.max_seq_length
        self.__tokenizer = tokenizer
        self.__embeddings_path_2 = args.embeddings_path_2
        self.__slide_ids = slide_ids

    def setup(self, stage=None):
        self.predict_ds = EmbeddingPredictDataset(self.__embeddings_path, self.__reports_json_path, self.__tokenizer,
                              self.__max_seq_length, self.__embeddings_path_2, self.__slide_ids)
        # print(dataset[0][1])


    def predict_dataloader(self):
        return DataLoader(self.self.predict_ds, batch_size=self.__batch_size, shuffle=self.__shuffle, collate_fn = self.collate_fn)


    @staticmethod
    def collate_fn(batch, device='cuda'):
        slide_ids, patch_feats_1,patch_feats_2 = zip(*batch)
        patch_feats1_pad = pad_sequence(patch_feats_1, batch_first=True).to(device)
        patch_feats2_pad = pad_sequence(patch_feats_2, batch_first=True).to(device)


        return (slide_ids, patch_feats1_pad, patch_feats2_pad)





