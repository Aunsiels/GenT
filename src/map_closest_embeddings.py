"""
This program align two KBs given the embeddings of their statements. It takes three arguments:
    * The embeddings of the first KB as given in compute_embeddings_sentence.py
    * The embeddings of the first KB as given in compute_embeddings_sentence.py
    * The output filename, that will be composed of three columns (tab separated): statement of the first KB, statement
    of the second KB and a distance score between the two.
"""

from sys import argv
from sklearn.neighbors import NearestNeighbors
import numpy as np


OPENKB_DATA = argv[1]
OPENKB_SENTENCES = OPENKB_DATA + ".sentences.txt"
OPENKB_EMBEDDINGS = OPENKB_DATA + ".emb.npy"

CLOSEKB_DATA = argv[2]
CLOSEKB_SENTENCES = CLOSEKB_DATA + ".sentences.txt"
CLOSEKB_EMBEDDINGS = CLOSEKB_DATA + ".emb.npy"

OUTPUT_FILENAME = argv[3]

open_embeddings = np.load(OPENKB_EMBEDDINGS)
close_embeddings = np.load(CLOSEKB_EMBEDDINGS)
open_sentences = []
with open(OPENKB_SENTENCES) as f:
    for line in f:
        open_sentences.append(line.strip())
close_sentences = []
with open(CLOSEKB_SENTENCES) as f:
    for line in f:
        close_sentences.append(line.strip())


print("Fit the model...")
nbrs = NearestNeighbors(n_neighbors=1, n_jobs=10).fit(close_embeddings)

print("Predictions...")
distances, indices = nbrs.kneighbors(open_embeddings)

print("Save results...")
with open(OUTPUT_FILENAME, "w") as f:
    for open_sentence, indice, distance in zip(open_sentences, indices, distances):
        f.write(open_sentence + "\t" + close_sentences[indice[0]] + "\t" + str(distance[0]) + "\n")
