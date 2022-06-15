from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import OrderedDict
import torch
import argparse

# api key 가져오기
f = open("./hf_key.txt", "r")
HUGGINGFACE_AUTH_TOKEN = f.readline()
f.close()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="(default: bert-base-uncased)")
    parser.add_argument("--model_path", type=str, default="model_weights_epoch5.pth", help="(default: model_weights_epoch5.pth)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_config()

    # set tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # add [HL], [/HL] token
    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[HL]", "[/HL]"]})
    # add token number
    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)

    # DataParallel model to single gpu
    state_dict = torch.load(args.model_path)
    keys = state_dict.keys()
    values = state_dict.values()

    new_keys = []
    for key in keys:
        new_key = key[7:]  # remove the 'module.' (module.bert.embeddings.position_ids --> bert.embeddings.position_ids)
        new_keys.append(new_key)

    new_dict = OrderedDict(list(zip(new_keys, values)))

    # load model
    model.load_state_dict(new_dict)
    model.to(device)

    # push to hub
    model.push_to_hub("BERT-Question-Generation", use_temp_dir=True, use_auth_token=HUGGINGFACE_AUTH_TOKEN)
    tokenizer.push_to_hub("BERT-Question-Generation", use_temp_dir=True, use_auth_token=HUGGINGFACE_AUTH_TOKEN)
