import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse

from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_config():
    parser = argparse.ArgumentParser()

    """init options"""
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="(default: bert-base-uncased)")
    parser.add_argument("--model_path", type=str, default="model_weights_4_72.512201225647.pth", help="(default: model_weights_4_72.512201225647.pth)")
    parser.add_argument("--test_file_path", type=str, default="./data/squad_nqg/test.json", help="(default: ./data/squad_nqg/test.json)")
    parser.add_argument("--save_txt_path", type=str, default="./test.txt", help="(default: ./test.txt)")

    """prediction options"""
    parser.add_argument("--max_question_token_len", type=int, default=20, help="(default: 20)")
    parser.add_argument("--max_len", type=int, default=512, help="(bert max token len, default: 512)")
    # parser.add_argument("", type=, default=, help="(default: )")

    args = parser.parse_args()
    return args


def inference(args):
    # set tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # add [HL], [/HL] token
    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[HL]", "[/HL]"]})
    # add token number
    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)

    model = nn.DataParallel(model)  # train에서 multi-GPU를 사용했으므로 불러올 때에도 이를 사용한다.

    # load model
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # load and set test file
    with open(args.test_file_path, "r") as file:
        data = json.load(file)

    context_list = []
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token

    for d in data:
        answers = d["answers"][0]["text"]
        answer_start = d["answers"][0]["answer_start"]
        input_text = f"{cls_token} {d['context'][:answer_start]} [HL] {d['context'][answer_start:answer_start+len(answers)]} [/HL] {d['context'][answer_start+len(answers):]} {sep_token}"
        context_list.append(input_text)

    # evaluation
    model.eval()
    pred_sentence_list = []  # 예측된 문장이 저장됨

    for context in tqdm(context_list, desc="making questions...: "):
        context_tokenized = tokenizer.encode(context, add_special_tokens=False)  # token id --> [CLS] + passage with [HL] + [SEP]
        pred_str_list = []  # 새롭게 예측된 토큰이 저장됨

        for _ in range(args.max_question_token_len):  # 최대 토큰 개수 20개로 제한
            pred_str_ids = tokenizer.convert_tokens_to_ids(pred_str_list + [mask_token])  # mask token을 뒤에 붙여줘야함
            predict_token = context_tokenized + pred_str_ids
            if len(predict_token) >= args.max_len:
                break
            predict_token = torch.tensor([predict_token])  # [[tokens]] 형태로 들어가야지 model에 들어갈 수 있음
            predictions = model(predict_token)  # [CLS] + tokenized passage with [HL] + [SEP] + predicted question + [MASK]
            predicted_index = torch.argmax(predictions[0][0][-1]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
            if "[SEP]" in predicted_token:
                break
            pred_str_list.append(predicted_token[0])  # 예측된 토큰 저장
        token_ids = tokenizer.convert_tokens_to_ids(pred_str_list)  # 생성된 토큰 저장

        pred_sentence_list.append(tokenizer.decode(token_ids))

    # save predicted
    with open(args.save_txt_path, "w", encoding="UTF-8") as f:
        for name in pred_sentence_list:
            f.write(name + "\n")


if __name__ == "__main__":
    args = get_config()
    inference(args)
