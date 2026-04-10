import os
import json
import re
from collections import Counter

from utils.utils import read_json_file


class Tokenizer:

    def __init__(self, reports_json_path, dataset_type, threshold=1):
        self.__threshold = threshold
        self.__pattern = re.compile(r'\s+|[\w]+|[^\w\s]', re.UNICODE)
        self.clean_report = lambda x: x
        if dataset_type == 'tcga_brca' or dataset_type == 'histai':
            self.clean_report = self.clean_report_brca_v2
        self.__token2idx, self.__idx2token = self.create_vocabulary(reports_json_path)



    def create_vocabulary(self, reports_json_path):
        total_tokens = []
        reports = read_json_file(reports_json_path)

        # for report in reports:
        #     tokens = self.__split_text(self.clean_report(report['report']))
        #     # for token in tokens:
        #     #     total_tokens.append(token)
        #     total_tokens.extend(tokens)

        for split in reports:
            for example in reports[split]:
                tokens =  self.__split_text(self.clean_report(example['report']))
                total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.__threshold] + ['<unk>']
        vocab.sort()
        # print(f'vocab: {vocab}')
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        return token2idx, idx2token

    def get_token_by_id(self, id):
        return self.__idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.__token2idx:
            return self.__token2idx['<unk>']
        return self.__token2idx[token]

    def get_vocab_size(self):
        return len(self.__token2idx)

    def __split_text(self, text):
        # text = text.lower()
        # return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

        return [m.group(0) for m in self.__pattern.finditer(text)]
        # return text.split()

    def __call__(self, report):
        tokens = self.__split_text(self.clean_report(report))
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))

        # print(f'tokens: {tokens}, ids: {ids}, report: {self.clean_report(report)}')
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        # print(ids)
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                # if i >= 1:
                #     txt += ' '
                txt += self.get_token_by_id(idx)
            else:
                break
        # print(self.__idx2token)
        # print(f'ids: {ids}, txt: {txt}')
        return txt

    def batch_decode(self, ids_batch):
        # print(f'ids_batch: {ids_batch}, {ids_batch.shape}')
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

    def clean_report_brca(self, report):
        report_cleaner = lambda t: (t.replace('\n', ' ').replace('  ', ' ')
                                    .replace('  ', ' ').replace('  ', ' ')
                                    .replace(' 10. ', ' ').replace(' 11. ', ' ')
                                    .replace(' 12. ', ' ').replace(' 13. ', ' ')
                                    .replace(' 14.', ' ').replace(' 1. ', ' ')
                                    .replace(' 2. ', ' ').replace(' 3. ', ' ')
                                    .replace(' 4. ', ' ').replace(' 5. ', ' ')
                                    .replace(' 6. ', ' ').replace(' 7. ', ' ')
                                    .replace(' 8. ', ' ') .replace(' 9. ', ' ').strip().lower() + ' ').split('. ')
        sent_cleaner = lambda t: re.sub('[#,?;*!^&_+():-\[\]{}]', '', t.replace('"', '').
                                    replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens)
        return report

    def clean_report_brca_v2(self, report):
        # -------------------------------------------------------
        # 0. Remove invalid UTF-8 characters
        # -------------------------------------------------------
        report = report.encode("utf-8", "ignore").decode("utf-8")

        # -------------------------------------------------------
        # 1. Normalize whitespace and remove section numbering
        # -------------------------------------------------------
        text = report.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        # Remove bullets like: 1. , 10. , 14.
        text = re.sub(r"\b([1-9]|1[0-4])\.\s*", " ", text)

        # -------------------------------------------------------
        # 2. Remove measurement expressions
        # -------------------------------------------------------
        text = re.sub(
            r'\b(?:\d+(?:\.\d+)?\s*(?:x|×|by)\s*){2,}\d+(?:\.\d+)?\s*(?:cm|mm|µm|um)\b',
            'x units',
            text,
            flags=re.IGNORECASE
        )

        # Remove 2D only (A x B cm)
        text = re.sub(
            r'\b\d+(?:\.\d+)?\s*(?:x|×|by)\s*\d+(?:\.\d+)?\s*(?:cm|mm|µm|um)\b',
            'x units',
            text,
            flags=re.IGNORECASE
        )

        # Remove single measurements (A cm, A mm)
        text = re.sub(
            r'\b\d+(?:\.\d+)?\s*(?:cm|mm|µm|um)\b',
            'x units',
            text,
            flags=re.IGNORECASE
        )

        # Normalize spaces again
        text = re.sub(r"\s+", " ", text).strip().lower()+' '

        # -------------------------------------------------------
        # 3. Split into sentences by ". "
        # -------------------------------------------------------
        sentences = text.split(". ")

        # -------------------------------------------------------
        # 4. Clean each sentence: remove punctuation, quotes, noise
        # -------------------------------------------------------
        def clean_sentence(s):
            s = re.sub(r'[#,?;*!^&_+():\-\[\]{}]', '', s)
            s = s.replace('"', '').replace("'", "").replace("\\", "")
            return s.strip().lower()

        cleaned = [clean_sentence(s) for s in sentences if clean_sentence(s)]

        # -------------------------------------------------------
        # 5. Re-join cleaned sentences
        # -------------------------------------------------------
        return " . ".join(cleaned) + " ."
    