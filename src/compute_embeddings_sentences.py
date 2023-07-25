"""
This program computes embeddings for the statements in a KB using a language model. It takes 2 parameters:
    * A subject, predicate, object, ... knowledge base, tab separated.
    * An output filename.
"""
from sys import argv

import numpy as np
from sentence_transformers import SentenceTransformer

from predicate_conceptnet import predicate_mapping

IN_FILENAME = argv[1]
OUT_FILENAME = argv[2]
EMBEDDINGS_FILENAME = OUT_FILENAME + ".emb"
SENTENCES_FILENAME = OUT_FILENAME + ".sentences.txt"

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentences = []
with open(IN_FILENAME) as f:
    for line in f:
        line = line.strip().replace("_", " ").split("\t")
        if len(line) < 3:
            continue
        sentences.append(line[0] + "," + predicate_mapping.get(line[1], line[1]) + "," + line[2])

embeddings = model.encode(sentences)

np.save(EMBEDDINGS_FILENAME, embeddings)
with open(SENTENCES_FILENAME, 'w') as f:
    f.write("\n".join(sentences))



