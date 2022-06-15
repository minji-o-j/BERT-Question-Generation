# BERT-Question-Generation
Try to implement a paper [A Recurrent BERT-based Model for Question Generation](https://aclanthology.org/D19-5821/) (EMNLP 2019)
<br>

---
---
## Score (SQuAD 73K, paragraph level context)
### In Original Paper 
Model|BLEU1|BLEU2|BLEU3|BLEU4|METEOR|ROUGE-L
:---:|:---:|:---:|:---:|:---:|:---:|:---:
BERT-HLSQG|49.73|34.60|26.13|20.33|23.88|48.23
<br>


### My Model (Epoch 5, bert-base-uncased)
Model|BLEU1|BLEU2|BLEU3|BLEU4|METEOR|ROUGE-L
:---:|:---:|:---:|:---:|:---:|:---:|:---:
BERT-HLSQG|52.26|33.19|22.48|15.64|22.66|45.48

<br>

---
---
## Quick start
### install package
```py
pip install -r requirements.txt 
```
### Train
```py
bash start.sh ## if you want to save log in nohup.out
# or
python train.py
```
### Inference
```py
python inference.py
```
### Scoring
```py
## setup scorer
python setup_scorer.py

## make srt-text, tgt-test file

## evaluation
python nqg/qgevalcap/eval.py \
  --src ./data/squad_nqg/src-test.txt \
  --tgt ./data/squad_nqg/tgt-test.txt \
  --out ./data/squad_nqg/predict_squad.txt
```
<br>

---
---
## Reference
https://github.com/voidful/BertGenerate
