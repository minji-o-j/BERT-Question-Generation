import json


def question_preprocess(text):
    text.replace(' "', ' " ')
    text.replace('" ', ' " ')

    if len(text) == 0:
        return ""

    if text[-1] == "?":
        text = text[:-1] + " ?"
    else:
        text = text + " ?"

    return text.lower()


# squad json
data_path = "./data/squad_nqg/test.json"
with open(data_path, "r") as file:
    data = json.load(file)

context_list = []
gold_list = []
predict_list = []

for d in data:
    context_list.append(d["context"].replace("\n", ""))
    question = question_preprocess(d["question"])
    gold_list.append(question)

# print(len(context_list),len(gold_list))


# predicted answer file
predicted_answer = open("./test.txt", "r")
while True:
    line = predicted_answer.readline()
    if not line:
        break
    predict_list.append(question_preprocess(line.strip()))

predicted_answer.close()

# with open("./data/squad_nqg/src-test.txt", "w", encoding="UTF-8") as f:

with open("./data/squad_nqg/src-test.txt", "w", encoding="UTF-8") as f:
    for name in context_list:
        f.write(name + "\n")
    f.close()

with open("./data/squad_nqg/tgt-test.txt", "w", encoding="UTF-8") as f:
    for name in gold_list:
        f.write(name + "\n")
    f.close()
with open("./data/squad_nqg/predict_squad.txt", "w", encoding="UTF-8") as f:
    for name in predict_list:
        f.write(name + "\n")
    f.close()
