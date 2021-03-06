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
            "???????????? ???????? ?????????????? ?????????? ???? ?????????? [MASK] ??????",
            "???????????? [MASK] ???? ?????? ?????????? ?????? ????????",
            "???????????????? ???????????? ?????????? [MASK] ???? ?????????????? ??????",
            "???????? ?????? ???? ?????????? ???? ???????? ?????? [MASK] ???? ?????????????? ??????????",
            "???????? ?????????? ???? ???????????????? [MASK] ???? ??????????",
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
