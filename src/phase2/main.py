### Q1: Train A Language Model
### Asumption you installed all the necessary libraries


import numpy as np
import tensorflow as tf
import collections
import numpy as np

# from transformers.data import tf_default_data_collator
from transformers.data.data_collator import tf_default_data_collator
import pandas as pd


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return tf_default_data_collator(features)


import pandas as pd

df = pd.read_csv("khabar.csv")
df = df[df["text"].apply(lambda x: len(x.split()) > 3)]
labels = list(df["label"].value_counts()[:10].index)
base_path = "/content/drive/MyDrive/NLP/"
for la in labels:
    df[df["label"] == la].to_csv(base_path + f"khabar_{la}.csv", index=None)
## mount drive
from google.colab import drive

drive.mount("/content/drive")


for ra, la in enumerate(labels):
    try:
        from transformers import TFAutoModelForMaskedLM

        model_checkpoint = "HooshvareLab/bert-fa-base-uncased"
        model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
        model(model.dummy_inputs)  # Build the model
        # model.summary()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        from datasets import load_dataset

        imdb_dataset = load_dataset(
            "csv",
            column_names=["text", "label"],
            data_files={
                "train": base_path + f"khabar_{la}.csv",
                "test": "/content/sample_khabar.csv",
            },
        )
        # imdb_dataset = load_dataset('imdb')

        # Use batched=True to activate fast multithreading!
        tokenized_datasets = imdb_dataset.map(
            tokenize_function, batched=True, remove_columns=["text", "label"]
        )
        chunk_size = 128

        lm_datasets = tokenized_datasets.map(group_texts, batched=True)

        from transformers import DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=0.15
        )

        train_size = 0.99
        test_size = 0.01

        downsampled_dataset = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )
        tf_train_dataset = downsampled_dataset["train"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=data_collator,
            shuffle=True,
            batch_size=32,
        )

        tf_eval_dataset = downsampled_dataset["test"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=data_collator,
            shuffle=False,
            batch_size=32,
        )

        from transformers import create_optimizer
        from transformers.keras_callbacks import PushToHubCallback
        import tensorflow as tf

        num_train_steps = len(tf_train_dataset)
        optimizer, schedule = create_optimizer(
            init_lr=2e-5,
            num_warmup_steps=1_000,
            num_train_steps=num_train_steps,
            weight_decay_rate=0.01,
        )
        model.compile(optimizer=optimizer)

        # Train in mixed-precision float16
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # callback = PushToHubCallback(
        #     output_dir=f"{model_name}-finetuned-imdb", tokenizer=tokenizer
        # )

        import math

        eval_loss = model.evaluate(tf_eval_dataset)
        print(f"Perplexity: {math.exp(eval_loss):.2f}")

        model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=5)

        eval_loss = model.evaluate(tf_eval_dataset)
        print(f"Perplexity: {math.exp(eval_loss):.2f}")

        ts = [
            "فرآیند کاهش آمارهای کرونا در تهران [MASK] است",
            "انصراف [MASK] به سود رئیسی صحت دارد",
            "جدیدترین خبرهای «حضور [MASK] در ورزشگاه ها»",
            "دوست جون ها شماها چه جوری خبر [MASK] به شوهرتون دادید",
            "کدام ستاره ها بالاترین [MASK] را دارند",
        ]
        for text in ts:
            inputs = tokenizer(text, return_tensors="np")
            token_logits = model(**inputs).logits
            # Find the location of [MASK] and extract its logits
            mask_token_index = np.argwhere(
                inputs["input_ids"] == tokenizer.mask_token_id
            )[0, 1]
            mask_token_logits = token_logits[0, mask_token_index, :]
            # Pick the [MASK] candidates with the highest logits
            # We negate the array before argsort to get the largest, not the smallest, logits
            top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()
            print(text)
            for token in top_5_tokens:
                print(
                    f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}"
                )

        model.save(base_path + f"models/model_khabar_online_{la}.md")
    except:
        continue


### Q2: BaseLine models
### Asumption you installed all the necessary libraries

import pandas as pd

base_path = "/content/drive/MyDrive/NLP/"
l = []
for la in labels:
    l.append(pd.read_csv(base_path + f"khabar_{la}.csv"))
df = pd.concat(l, axis=0, ignore_index=True)
df.head()
df = df[~df["label"].isnull().values]
df["label"] = df["label"].apply(lambda x: int(x))
df = df[~df["text"].isnull().values]

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(ngram_range=(1, 3), min_df=100)
X_train_counts = count_vect.fit_transform(df.text)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

### CLASSICAL MODELS

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_train_tf, df["label"], stratify=df["label"], test_size=0.2
)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

fig, ax = plt.subplots(figsize=(10, 10))
plot_confusion_matrix(clf, X_test, y_test, ax=ax)
plt.show()

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(class_weight="balanced").fit(
    np.stack(X_train, axis=0), y_train
)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
fig, ax = plt.subplots(figsize=(10, 10))
plot_confusion_matrix(clf, X_test, y_test, ax=ax)
plt.show()
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier().fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
fig, ax = plt.subplots(figsize=(10, 10))
plot_confusion_matrix(clf, X_test, y_test, ax=ax)
plt.show()

### Word2Vec models
from gensim.test.utils import common_texts

from gensim.models import Word2Vec

model = Word2Vec(sentences=df["text"], size=128, window=5, min_count=1, workers=4)
model.train(sentences=df["text"], total_words=1000, epochs=100)

from hazm import word_tokenize


def embed_doc(r):
    res = np.zeros(128)
    for w in r.split():
        if w in model.wv:
            res += model.wv[w]
    return res


df["doc_embed"] = df["text"].apply(embed_doc)


### Q3: Advanced Models
### Asumption you installed all the necessary libraries

from hazm import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

base_path = "/content/drive/MyDrive/NLP/"
l = []
for la in labels:
    l.append(pd.read_csv(base_path + f"khabar_{la}.csv"))
df = pd.concat(l, axis=0, ignore_index=True)
df.head()
df = df[~df["label"].isnull().values]
df["label"] = df["label"].apply(lambda x: int(x))
df = df[~df["text"].isnull().values]

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(ngram_range=(1, 3), min_df=100)
X_train_counts = count_vect.fit_transform(df.text)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

### CLASSICAL MODELS

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_train_tf, df["label"], stratify=df["label"], test_size=0.2
)

### Deep Models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(df["label"])

model_clf = keras.Sequential(
    [
        layers.Dense(128, activation="relu", name="layer1"),
        layers.Dense(64, activation="relu", name="layer2"),
        layers.Dropout(0.2),
        layers.Dense(20, activation="softmax", name="layer3"),
    ]
)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_train_tf, categorical_labels, stratify=df["label"], test_size=0.2
)
X_train.shape

# model_clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model_clf.fit(X_train, y_train, epochs=5, batch_size=128)

y_pred = np.argmax(model_clf.predict(X_test), axis=1)
y_pred.shape
df["label"].shape
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.show()

### RNN
max_features = 4978  # number of words to consider as features
maxlen = 30  # cut texts after this number of words (among top max_features most common words)

df["seq"] = df["text"].apply(
    lambda x: list(
        map(
            lambda y: count_vect.vocabulary_.get(y, 4978),
            tf.keras.preprocessing.text.text_to_word_sequence(x),
        )
    )
)
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(64, return_sequences=True))
model.summary()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["seq"], categorical_labels, stratify=df["label"], test_size=0.2
)
(input_train, y_train), (input_test, y_test) = (X_train, y_train), (X_test, y_test)
from keras.datasets import imdb
from keras.preprocessing import sequence


print("Pad sequences (samples x time)")
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print("input_train shape:", input_train.shape)
print("input_test shape:", input_test.shape)

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(layers.LSTM(32, dropout=0.2, return_sequences=True))
model.add(layers.LSTM(32, dropout=0.2, return_sequences=False))
model.add(layers.Dense(20, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
history = model.fit(
    input_train, y_train, epochs=20, batch_size=128, validation_split=0.05
)

model.save(base_path + "calssification/LSTM")
model_clf.save(base_path + "calssification/dense")


### Transformers


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import hazm
from cleantext import clean


from tqdm.notebook import tqdm

import os
import re
import json
import copy
import collections


def cleanhtml(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html)
    return cleantext


def cleaning(text):
    text = text.strip()

    # regular cleaning
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="0",
        replace_with_currency_symbol="",
    )

    # cleaning htmls
    text = cleanhtml(text)

    # normalizing
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)

    # removing wierd patterns
    wierd_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u200d"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\u3030"
        "\ufe0f"
        "\u2069"
        "\u2066"
        # u"\u200c"
        "\u2068" "\u2067" "]+",
        flags=re.UNICODE,
    )

    text = wierd_pattern.sub(r"", text)

    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)

    return text


data = df.copy()
import re

minlim, maxlim = 3, 256
# cleaning comments
data["cleaned_comment"] = data["text"].apply(cleaning)


# calculate the length of comments based on their words
data["cleaned_comment_len_by_words"] = data["cleaned_comment"].apply(
    lambda t: len(hazm.word_tokenize(t))
)

# remove comments with the length of fewer than three words
data["cleaned_comment_len_by_words"] = data["cleaned_comment_len_by_words"].apply(
    lambda len_t: len_t if minlim < len_t <= maxlim else len_t
)
data = data.dropna(subset=["cleaned_comment_len_by_words"])
data = data.reset_index(drop=True)

data.head()

data[["text", "cleaned_comment"]].to_csv(base_path + "khabar_cleaned.csv")

data = data[["cleaned_comment", "label"]]
data.columns = ["comment", "label"]
data.head()


data["label_id"] = data["label"].apply(lambda t: labels.index(t))

train, test = train_test_split(
    data, test_size=0.2, random_state=1, stratify=data["label"]
)
train, valid = train_test_split(
    train, test_size=0.1, random_state=1, stratify=train["label"]
)

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

x_train, y_train = train["comment"].values.tolist(), train["label_id"].values.tolist()
x_valid, y_valid = valid["comment"].values.tolist(), valid["label_id"].values.tolist()
x_test, y_test = test["comment"].values.tolist(), test["label_id"].values.tolist()

print(train.shape)
print(valid.shape)
print(test.shape)

from transformers import BertConfig, BertTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available.  Training on CPU ...")
else:
    print("CUDA is available!  Training on GPU ...")

# general config
MAX_LEN = 30
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

EPOCHS = 3
EEVERY_EPOCH = 1000
LEARNING_RATE = 2e-5
CLIP = 0.0

MODEL_NAME_OR_PATH = "HooshvareLab/bert-fa-base-uncased"
OUTPUT_PATH = "/content/bert-fa-base-uncased-sentiment-taaghceh/pytorch_model.bin"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# create a key finder based on label 2 id and id to label

label2id = {label: i for i, label in enumerate(labels)}
id2label = {v: k for k, v in label2id.items()}

print(f"label2id: {label2id}")
print(f"id2label: {id2label}")

# setup the tokenizer and configuration

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
config = BertConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    **{
        "label2id": label2id,
        "id2label": id2label,
    },
)

print(config.to_json_string())

idx = np.random.randint(0, len(train))
sample_comment = train.iloc[idx]["comment"]
sample_label = train.iloc[idx]["label"]

print(f"Sample: \n{sample_comment}\n{sample_label}")

tokens = tokenizer.tokenize(sample_comment)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"  Comment: {sample_comment}")
print(f"   Tokens: {tokenizer.convert_tokens_to_string(tokens)}")
print(f"Token IDs: {token_ids}")


encoding = tokenizer.encode_plus(
    sample_comment,
    max_length=32,
    truncation=True,
    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    return_attention_mask=True,
    padding="max_length",
    return_tensors="pt",  # Return PyTorch tensors
)

print(f"Keys: {encoding.keys()}\n")
for k in encoding.keys():
    print(f"{k}:\n{encoding[k]}")


class KhabarOnlineDataset(torch.utils.data.Dataset):
    """Create a PyTorch dataset for Khabar Online."""

    def __init__(self, tokenizer, comments, targets=None, label_list=None, max_len=128):
        self.comments = comments
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.label_map = (
            {label: i for i, label in enumerate(label_list)}
            if isinstance(label_list, list)
            else {}
        )

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])

        if self.has_target:
            target = self.label_map.get(
                str(self.targets[item]), str(self.targets[item])
            )

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        inputs = {
            "comment": comment,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
        }

        if self.has_target:
            inputs["targets"] = torch.tensor(target, dtype=torch.long)

        return inputs


def create_data_loader(x, y, tokenizer, max_len, batch_size, label_list):
    dataset = KhabarOnlineDataset(
        comments=x,
        targets=y,
        tokenizer=tokenizer,
        max_len=max_len,
        label_list=label_list,
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


label_list = list(map(str, labels))
train_data_loader = create_data_loader(
    train["comment"].to_numpy(),
    train["label"].to_numpy(),
    tokenizer,
    MAX_LEN,
    TRAIN_BATCH_SIZE,
    label_list,
)
valid_data_loader = create_data_loader(
    valid["comment"].to_numpy(),
    valid["label"].to_numpy(),
    tokenizer,
    MAX_LEN,
    VALID_BATCH_SIZE,
    label_list,
)
test_data_loader = create_data_loader(
    test["comment"].to_numpy(), None, tokenizer, MAX_LEN, TEST_BATCH_SIZE, label_list
)

sample_data = next(iter(train_data_loader))

print(sample_data.keys())

print(sample_data["comment"])
print(sample_data["input_ids"].shape)
print(sample_data["input_ids"][0, :])
print(sample_data["attention_mask"].shape)
print(sample_data["attention_mask"][0, :])
print(sample_data["token_type_ids"].shape)
print(sample_data["token_type_ids"][0, :])
print(sample_data["targets"].shape)
print(sample_data["targets"][0])


class CategorizerModel(nn.Module):
    def __init__(self, config):
        super(CategorizerModel, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME_OR_PATH)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


pt_model = CategorizerModel(config=config)
pt_model = pt_model.to(device)

print("pt_model", type(pt_model))

# sample data output

sample_data_comment = sample_data["comment"]
sample_data_input_ids = sample_data["input_ids"]
sample_data_attention_mask = sample_data["attention_mask"]
sample_data_token_type_ids = sample_data["token_type_ids"]
sample_data_targets = sample_data["targets"]

# available for using in GPU
sample_data_input_ids = sample_data_input_ids.to(device)
sample_data_attention_mask = sample_data_attention_mask.to(device)
sample_data_token_type_ids = sample_data_token_type_ids.to(device)
sample_data_targets = sample_data_targets.to(device)


# outputs = F.softmax(
#     pt_model(sample_data_input_ids, sample_data_attention_mask, sample_data_token_type_ids),
#     dim=1)

outputs = pt_model(
    sample_data_input_ids, sample_data_attention_mask, sample_data_token_type_ids
)
_, preds = torch.max(outputs, dim=1)

print(outputs[:5, :])
print(preds[:5])


def simple_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def acc_and_f1(y_true, y_pred, average="weighted"):
    acc = simple_accuracy(y_true, y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    return {
        "acc": acc,
        "f1": f1,
    }


def y_loss(y_true, y_pred, losses):
    y_true = torch.stack(y_true).cpu().detach().numpy()
    y_pred = torch.stack(y_pred).cpu().detach().numpy()
    y = [y_true, y_pred]
    loss = np.mean(losses)

    return y, loss


def eval_op(model, data_loader, loss_fn):
    model.eval()

    losses = []
    y_pred = []
    y_true = []

    with torch.no_grad():
        for dl in tqdm(data_loader, total=len(data_loader), desc="Evaluation... "):

            input_ids = dl["input_ids"]
            attention_mask = dl["attention_mask"]
            token_type_ids = dl["token_type_ids"]
            targets = dl["targets"]

            # move tensors to GPU if CUDA is available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            targets = targets.to(device)

            # compute predicted outputs by passing inputs to the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # convert output probabilities to predicted class
            _, preds = torch.max(outputs, dim=1)

            # calculate the batch loss
            loss = loss_fn(outputs, targets)

            # accumulate all the losses
            losses.append(loss.item())

            y_pred.extend(preds)
            y_true.extend(targets)

    eval_y, eval_loss = y_loss(y_true, y_pred, losses)
    return eval_y, eval_loss


def train_op(
    model,
    data_loader,
    loss_fn,
    optimizer,
    scheduler,
    step=0,
    print_every_step=100,
    eval=False,
    eval_cb=None,
    eval_loss_min=np.Inf,
    eval_data_loader=None,
    clip=0.0,
):

    model.train()

    losses = []
    y_pred = []
    y_true = []

    for dl in tqdm(data_loader, total=len(data_loader), desc="Training... "):
        step += 1

        input_ids = dl["input_ids"]
        attention_mask = dl["attention_mask"]
        token_type_ids = dl["token_type_ids"]
        targets = dl["targets"]

        # move tensors to GPU if CUDA is available
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        targets = targets.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # compute predicted outputs by passing inputs to the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # convert output probabilities to predicted class
        _, preds = torch.max(outputs, dim=1)

        # calculate the batch loss
        loss = loss_fn(outputs, targets)

        # accumulate all the losses
        losses.append(loss.item())

        # compute gradient of the loss with respect to model parameters
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        # perform optimization step
        optimizer.step()

        # perform scheduler step
        scheduler.step()

        y_pred.extend(preds)
        y_true.extend(targets)

        if eval:
            train_y, train_loss = y_loss(y_true, y_pred, losses)
            train_score = acc_and_f1(train_y[0], train_y[1], average="weighted")

            if step % print_every_step == 0:
                eval_y, eval_loss = eval_op(model, eval_data_loader, loss_fn)
                eval_score = acc_and_f1(eval_y[0], eval_y[1], average="weighted")

                if hasattr(eval_cb, "__call__"):
                    eval_loss_min = eval_cb(
                        model,
                        step,
                        train_score,
                        train_loss,
                        eval_score,
                        eval_loss,
                        eval_loss_min,
                    )

    train_y, train_loss = y_loss(y_true, y_pred, losses)

    return train_y, train_loss, step, eval_loss_min


optimizer = AdamW(pt_model.parameters(), lr=LEARNING_RATE, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()

step = 0
eval_loss_min = np.Inf
history = collections.defaultdict(list)


def eval_callback(epoch, epochs, output_path):
    def eval_cb(
        model, step, train_score, train_loss, eval_score, eval_loss, eval_loss_min
    ):
        statement = ""
        statement += "Epoch: {}/{}...".format(epoch, epochs)
        statement += "Step: {}...".format(step)

        statement += "Train Loss: {:.6f}...".format(train_loss)
        statement += "Train Acc: {:.3f}...".format(train_score["acc"])

        statement += "Valid Loss: {:.6f}...".format(eval_loss)
        statement += "Valid Acc: {:.3f}...".format(eval_score["acc"])

        print(statement)

        if eval_loss <= eval_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    eval_loss_min, eval_loss
                )
            )

            torch.save(model.state_dict(), output_path)
            eval_loss_min = eval_loss

        return eval_loss_min

    return eval_cb


for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs... "):
    train_y, train_loss, step, eval_loss_min = train_op(
        model=pt_model,
        data_loader=train_data_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        print_every_step=EEVERY_EPOCH,
        eval=True,
        eval_cb=eval_callback(epoch, EPOCHS, OUTPUT_PATH),
        eval_loss_min=eval_loss_min,
        eval_data_loader=valid_data_loader,
        clip=CLIP,
    )

    train_score = acc_and_f1(train_y[0], train_y[1], average="weighted")

    eval_y, eval_loss = eval_op(
        model=pt_model, data_loader=valid_data_loader, loss_fn=loss_fn
    )

    eval_score = acc_and_f1(eval_y[0], eval_y[1], average="weighted")

    history["train_acc"].append(train_score["acc"])
    history["train_loss"].append(train_loss)
    history["val_acc"].append(eval_score["acc"])
    history["val_loss"].append(eval_loss)
