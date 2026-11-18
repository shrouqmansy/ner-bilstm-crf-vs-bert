# Named Entity Recognition System

An end-to-end NLP sequence labeling project that identifies and classifies named entities
(**PER, ORG, LOC, MISC**) in text, comparing four architectures on the
[CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003) dataset, and deploying the
fine-tuned transformer as an interactive Gradio web app.

## Models compared

| Model | Description |
|---|---|
| LSTM | Unidirectional LSTM tagger, built from scratch (PyTorch) |
| BiLSTM | Bidirectional LSTM tagger, built from scratch |
| BiLSTM + CRF | BiLSTM with a CRF layer for joint tag-sequence decoding |
| DistilBERT | `distilbert-base-cased` fine-tuned with HuggingFace `Trainer` for token classification |

All from-scratch models use pre-trained **GloVe (100d)** word embeddings plus a **character-level CNN**
to handle out-of-vocabulary words.

## Pipeline

1. Load and explore CoNLL-2003 (IOB2 tagging scheme).
2. Build vocabulary, GloVe embedding matrix and character vocabulary.
3. Train LSTM, BiLSTM and BiLSTM+CRF with early stopping on validation F1.
4. Tokenize with `AutoTokenizer`, align word-level labels to subword tokens (`-100` for non-first subwords).
5. Fine-tune DistilBERT with `Trainer`.
6. Evaluate every model with **seqeval** (per-entity precision / recall / F1).
7. Analyze where the CRF improves boundary detection (invalid IOB transitions, boundary errors).
8. Visualize predicted spans with color-coded labels and deploy with Gradio.

## Results (CoNLL-2003 test set)

Entity-level F1 computed with seqeval:

| Model | PER F1 | ORG F1 | LOC F1 | MISC F1 | Overall F1 |
|---|---|---|---|---|---|
| LSTM | 0.9309 | 0.7416 | 0.8557 | 0.6788 | 0.8210 |
| BiLSTM | 0.9508 | 0.8500 | 0.9072 | 0.8006 | 0.8895 |
| BiLSTM + CRF | 0.9537 | 0.8769 | 0.9252 | 0.8022 | **0.9035** |
| DistilBERT | 0.9527 | 0.8595 | 0.9113 | 0.7553 | 0.8875 |

BiLSTM + CRF is the best model in this run. DistilBERT was fine-tuned for only 3 epochs, which is
likely why it does not beat it; more epochs or `bert-base-cased` would probably help (not tested).

### Where does the CRF help?

A softmax tagger predicts each tag independently, so it can produce invalid sequences such as
`O -> I-PER`, or cut a multi-word entity short. A CRF scores the whole tag sequence and decodes it
with Viterbi, which learns that `I-X` must follow `B-X` / `I-X`. Measured on the test set:

| Model | Invalid IOB transitions | Boundary errors (right type, wrong span) |
|---|---|---|
| LSTM | 325 | 310 |
| BiLSTM | 85 | 127 |
| BiLSTM + CRF | **0** | **82** |
| DistilBERT | 98 | 132 |

The CRF removes invalid transitions entirely and cuts boundary errors by about 35% compared with
the BiLSTM (e.g. it recovers the full span "Maccabi Tel Aviv" where the BiLSTM only tags "Tel Aviv").

## Note on DATE entities

CoNLL-2003 only has four entity types (PER, ORG, LOC, MISC) and no DATE class. To add DATE, use a
dataset such as OntoNotes 5 with the same code.

## Run the notebook

Open `NER_System.ipynb` in Google Colab, enable a GPU (Runtime -> Change runtime type -> T4 GPU)
and run all cells in order. Fine-tuning DistilBERT takes about 2 minutes on a T4.

## Run the web app

```bash
pip install -r requirements.txt
python app.py
```

The app loads the fine-tuned model from the `ner-distilbert/` folder (saved at the end of the
notebook with `trainer.save_model("ner-distilbert")`), or from a HuggingFace Hub id:

```bash
MODEL_PATH=your-username/ner-distilbert python app.py
```

Model weights are not stored in this repository (see `.gitignore`). Upload them to the
HuggingFace Hub, for example from the notebook:

```python
trainer.push_to_hub("your-username/ner-distilbert")
```

## Tech stack

PyTorch, HuggingFace Transformers and Datasets, pytorch-crf, seqeval, GloVe (gensim), Gradio.
