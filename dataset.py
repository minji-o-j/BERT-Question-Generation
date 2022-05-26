from os import device_encoding
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import torch
import pickle


class LoadDataset:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def make_data_pickle(self, data_path, pickle_path):  # data: json
        # open json
        with open(data_path, "r") as file:
            data = json.load(file)

        # special token
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        mask_token = self.tokenizer.mask_token

        example_list = []

        for d in tqdm(data[: len(data)], desc="***making pickle file...: "):
            example_pair = dict()
            target_text = d["question"]
            answer_start = d["answers"][0]["answer_start"]
            answers = d["answers"][0]["text"]

            input_text = f"{cls_token} {d['context'][:answer_start]} [HL] {d['context'][answer_start:answer_start+len(answers)]} [/HL] {d['context'][answer_start+len(answers):]} {sep_token}"
            tokenized_target = self.tokenizer.tokenize(target_text)  # tokenize question
            tokenized_text = self.tokenizer.tokenize(input_text, add_special_tokens=False)
            if len(tokenized_target + tokenized_text) + 2 >= 512:  # over bert_base size
                continue

            for i in range(0, len(tokenized_target) + 1):
                # tokenized
                tokenized_text.extend(tokenized_target[:i])  # tokenized_context + tokenized_question[:i]
                tokenized_text.append(mask_token)  # tokenized_context + tokenized_question[:i] + [MASK]
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                loss_ids = indexed_tokens.copy()

                if i == len(tokenized_target):
                    loss_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sep_token))[0])
                else:
                    loss_ids.append(self.tokenizer.convert_tokens_to_ids(tokenized_target[i]))

                loss_tensors = torch.tensor([loss_ids]).to(self.device)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                decodes_ids = self.tokenizer.decode(input_ids)
                example_pair[decodes_ids] = loss_tensors

            example_list.append(example_pair)

        with open(pickle_path, "wb") as f:
            pickle.dump(example_list, f, pickle.HIGHEST_PROTOCOL)

        return example_list

    def decompose_dataset(self, example_list):
        # 문장 list
        sentences = []
        # 정답 list
        labels = []

        for examples in tqdm(example_list, desc="***decompose dataset: "):
            sentences.extend(list(examples.keys()))
            labels.extend(list(examples.values()))

        # print(len(train_sentences), len(train_label))
        return sentences, labels

    def tokenized_dataset(self, data):  # data: 문장 list
        tokenized_sentence = self.tokenizer(
            data,
            padding=True,  # 문장의 길이가 짧다면 padding
            truncation=True,  # 문장이 길다면 truncate
            max_length=512,
            return_token_type_ids=True,  # roberta 모델에서는 False
            return_tensors="pt",  # Tensor로 반환!
            add_special_tokens=False,
        )
        return tokenized_sentence

    def make_labels(self, pad_len, labels):
        labels_list = []
        for label in tqdm(labels, desc="***make labels: "):
            target = torch.zeros(pad_len)
            try:
                target[: len(label[0])] = label[0]
            except:
                target = label[0][:-1][:pad_len]
            labels_list.append(target.tolist())
        labels_list = torch.tensor(labels_list).int()

        return labels_list


class SquadDataset(Dataset):
    def __init__(self, data):  # tokenized 된 것과 라벨이 들어옴
        self.data = data

    def __len__(self):  # data의 전체 길이
        return len(self.data)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        return item
