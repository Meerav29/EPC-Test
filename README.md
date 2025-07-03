# EPC-Test
EPC - testing only.

## Transformer Classification

A new script `transformer_train.py` uses Hugging Face's `transformers` library to train a `DistilBERT` model on the questions dataset. The workflow mirrors `load.py` but leverages a pretrained transformer and `Trainer` for fine-tuning.

Run it with:

```bash
python transformer_train.py
```

Both `load.py` (logistic regression) and `transformer_train.py` use the same train/test split. In typical scenarios BERT-style models outperform TF‑IDF baselines by capturing more context. Numeric results are not included here because this environment lacks the necessary packages to execute the scripts. When run locally with the required dependencies, compare the printed classification reports to evaluate the improvement.
