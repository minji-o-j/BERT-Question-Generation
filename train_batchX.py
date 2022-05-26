import torch.nn.functional as F
import numpy as np
import torch
import pickle
import random
import json
import os
import argparse

from dataset import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForMaskedLM, AutoConfig, AutoTokenizer, DataCollatorWithPadding


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_config():
    parser = argparse.ArgumentParser()

    """init options"""
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="(default: bert-base-uncased)")

    """train options"""
    parser.add_argument("--train_data_path", type=str, default="./data/squad_nqg/train.json", help="(default: ./data/squad_nqg/train.json)")
    parser.add_argument("--train_pickle_path", type=str, default="./squad_train_32.pickle", help="(default: squad_train.pickle)")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="(default: )")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rage (default: 5e-5)")

    # not frequently used
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="(default: )")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="(default: )")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="(default: )")
    parser.add_argument("--warmup_steps", type=float, default=0.0, help="(default: )")
    parser.add_argument("--max_steps", type=int, default=-1, help="(default: )")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="(default: )")

    # parser.add_argument("", type=, default=, help="(default: )")

    args = parser.parse_args()
    return args


def prepare_train_dataset(args, device, tokenizer):
    seed_everything(args.seed)

    load_dataset = LoadDataset(tokenizer, device)

    # check file
    if os.path.isfile(args.train_pickle_path):
        print("*** train file already exist!")
        with open(args.train_pickle_path, "rb") as f:
            example_list = pickle.load(f)
        print("*** finished to load train pickle file!")
    else:
        example_list = load_dataset.make_data_pickle(args.train_data_path, args.train_pickle_path)

    print(f"***train_len: {len(example_list)}")

    t_total = len(example_list) // args.gradient_accumulation_steps * args.num_train_epochs

    return t_total, example_list  # , train_dataloader


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.to(device)

    # add [HL], [/HL] token
    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[HL]", "[/HL]"]})
    # add token number
    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)

    t_total, example_list = prepare_train_dataset(args, device, tokenizer)
    print(f"***t_total: {t_total}")  ## len example_pair: {len(example_pair)}")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    model.train()

    for i in range(0, 60):
        eveloss = 0
        for example_pair in example_list:
            for k, v in example_pair.items():
                print('------------------------------')
                print(k)
                print('------------------------------')
                print(v)
                print('------------------------------')
                loss = model(k, labels=v)
                eveloss += loss.mean().item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        print("step " + str(i) + " : " + str(eveloss))


if __name__ == "__main__":
    args = get_config()
    train(args)
