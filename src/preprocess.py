import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import spacy
import configparser


def load_json_item(path, columns=["title", "category"]):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append({k: v for k, v in json.loads(line).items() if k in columns})
    return pd.DataFrame(data, columns=columns)


def clean_doc(d):
    doc = []
    for t in d:
        # t.is_punct,
        if not any(
            [t.is_stop, t.is_digit, not t.is_alpha, t.is_space, t.lemma_ == "-PRON-"]
        ):
            doc.append(t.lemma_)
    return " ".join(doc)


def preprocess(articles):
    iter_articles = (article for article in articles)
    clean_articles = []
    for i, doc in enumerate(nlp.pipe(iter_articles, batch_size=100, n_process=8), 1):
        if i % 1000 == 0:
            print(f"{i / len(articles):.2%}", end=" ", flush=True)
        clean_articles.append(clean_doc(doc))
    return clean_articles


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configs/preprocess.ini")
    path = config["PREPROCESS"]["InputFilePath"]
    cat_col = config["PREPROCESS"]["CategoryColumn"]
    title_col = config["PREPROCESS"]["TitleColumn"]
    output_path = config["PREPROCESS"]["OutputFilePath"]
    spcaymodel = config["PREPROCESS"]["SpacyModel"]
    stop_words_url = config["PREPROCESS"]["StopWordURL"]
    columns = [cat_col, title_col]
    df = load_json_item(path)
    df["category"] = df["category"].apply(lambda x: x.lower())

    stop_words = set(pd.read_csv(stop_words_url, header=None, squeeze=True).tolist())

    nlp = spacy.load(spcaymodel)
    nlp.max_length = 6000000
    print(f"using model : {spcaymodel} with pipelines: {nlp.pipe_names}")
    df["cleanTitle"] = preprocess(df["title"])
    df.to_csv(output_path)
