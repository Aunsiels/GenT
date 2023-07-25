"""
This script turns the generations obtained with evaluate.py at turn them into a knowledge base.
It takes four arguments:
    * The generation file
    * The original KB, composed of four columns separated with a tab: subject, predicate, object, score
    * The output file where to write the results
    * The top generations to consider
    * The type of merging to perform (count, weighted, jaccard)
    * Jaccard parameter, if required

This script will generate a first preprocessed version of the generation.
"""
import os

import pandas as pd
from collections import Counter
from sys import argv

from nltk.corpus import stopwords

from spacy_accessor import get_default_annotator

GENERATIONS_FILE = argv[1]
KB = argv[2]
OUTPUT_FILE = argv[3]
TOP = argv[4]
TYPE = argv[5]


spacy_accessor = get_default_annotator()
all_stopwords = stopwords.words('english')


def lemmatize_triple(triple):
    if type(triple) == str:
        return set([x for x in spacy_accessor.lemmatize(triple) if x != "," and x not in all_stopwords])
    return set()


def jaccard(a, b):
    if a or b:
        return len(a.intersection(b)) / len(a.union(b))
    return 0


df1 = pd.read_csv(GENERATIONS_FILE, sep="\t", header=None)
columns = list(df1.columns)[1:]
original = list(df1.columns)[0]


def preprocess(df, weights):
    res = []
    for n_row, row in df.iterrows():
        if row[original] in weights:
            weight = weights[row[original]]
        else:
            print(row[original])
            continue
        lem_original = lemmatize_triple(row[original])
        found_nothing = False
        for i, column in enumerate(columns):
            if row[column] == "Nothing":
                found_nothing = True
                continue
            lem_trans = lemmatize_triple(row[column])
            jaccard_distance = jaccard(lem_original, lem_trans)
            if found_nothing:
                res.append([row[original], row[column], weight, -i, jaccard_distance])
            else:
                res.append([row[original], row[column], weight, i, jaccard_distance])
    print("Number of lines:", len(res))
    return pd.DataFrame(res, columns=["original", "translation", "weight", "rank", "jaccard"])


def aggregate_simple(df, top):
    res = []
    for _, row in df.iterrows():
        if row["rank"] >= 0 and row["rank"] < top:
            res.append(row["translation"])
    return Counter(res).most_common()


def aggregate_weighted(df, top, jaccard=-1.0):
    res = dict()
    for _, row in df.iterrows():
        if row["rank"] >= 0 and row["rank"] < top and float(row["jaccard"]) >= jaccard:
            res[row["translation"]] = res.get(row["translation"], 0) + row["weight"] / (row["rank"] + 1)
    return sorted(res.items(), key=lambda x: -x[1])


dir = os.path.dirname(GENERATIONS_FILE)
preprocessed_basename = "preprocessed_" + os.path.basename(GENERATIONS_FILE)
preprocessed_path = os.path.join(dir, preprocessed_basename)

print("Preprocessed filename:", preprocessed_path)

if not os.path.isfile(preprocessed_path):
    weights = dict()
    df_openkb = pd.read_csv(KB, names=["s", "p", "o", "r"], sep="\t")
    for _, row in df_openkb.iterrows():
        weights[str(row["s"]) + "," + str(row["p"]) + "," + str(row["o"])] = float(row["r"])
    preprocessed = preprocess(df1, weights)
    preprocessed.to_csv(preprocessed_path, sep="\t", index=False)
else:
    print("Preprocessed file found!")
    preprocessed = pd.read_csv(preprocessed_path, sep="\t")


top = int(TOP)
new_kb = None
if TYPE == "simple":
    new_kb = aggregate_simple(preprocessed, top)
elif TYPE == "weighted":
    new_kb = aggregate_weighted(preprocessed, top)
elif TYPE == "jaccard":
    new_kb = aggregate_weighted(preprocessed, top, float(argv[6]))
else:
    print("Incorrect type selected.")
    exit(1)

df = pd.DataFrame(new_kb, columns=["triple", "frequency"])
df.to_csv(OUTPUT_FILE, sep="\t", index=False)
