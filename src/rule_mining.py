from collections import Counter
from sys import argv

from rule import get_all_hypernyms
from spacy_accessor import get_default_annotator

ALIGNMENT_FILENAME = argv[1]
OUTPUT_FILENAME = argv[2]

SPACY_ACCESSOR = get_default_annotator()

ALIGNMENTS = []
predicates_words = []
cn_predicates = set()
with open(ALIGNMENT_FILENAME) as f:
    for i, line in enumerate(f):
        line = line.strip().strip("<|endoftext|>").split("[SEP]")
        spo0 = line[0].split("\t")
        ALIGNMENTS.append((i, spo0, line[1].split("\t")))
        for x in spo0[1].split(" "):
            predicates_words.append(x)


predicate_counter = Counter(predicates_words)
top_preds = [x[0] for x in predicate_counter.most_common(100)]

ALL_HYPERS = dict()
ALL_SO_HYPERS = set()
for alignment in ALIGNMENTS:
    idx = str(alignment[0])
    s0, p0, o0 = alignment[1]
    s1, p1, o1 = alignment[2]
    p0_words = p0.split(" ")
    s_hyper = get_all_hypernyms(s1)
    for hyper in s_hyper:
        if hyper not in ALL_HYPERS:
            ALL_HYPERS[hyper] = set()
        ALL_HYPERS[hyper].add(s1)
        ALL_SO_HYPERS.add(s1)
    o_hyper = get_all_hypernyms(o1)
    for hyper in o_hyper:
        if hyper not in ALL_HYPERS:
            ALL_HYPERS[hyper] = set()
        ALL_HYPERS[hyper].add(o1)
        ALL_SO_HYPERS.add(o1)


TOTAL = len(ALL_SO_HYPERS)
ALLOWED_HYPERS = {x[0] for x in ALL_HYPERS.items() if len(x[1]) >= 10 and len(x[1]) / TOTAL < 0.5}


KB = []
for alignment in ALIGNMENTS:
    idx = str(alignment[0])
    s0, p0, o0 = alignment[1]
    s1, p1, o1 = alignment[2]
    p0_words = p0.split(" ")
    s_hyper = get_all_hypernyms(s1)
    for hyper in s_hyper:
        if hyper in ALLOWED_HYPERS:
            KB.append(s1 + "#" + idx + "\t" + "ISA" + "\t" + hyper)
    o_hyper = get_all_hypernyms(o1)
    for hyper in o_hyper:
        if hyper in ALLOWED_HYPERS:
            KB.append(o1 + "#" + idx + "\t" + "ISA" + "\t" + hyper)
    for top_pred in top_preds:
        if top_pred in p0_words:
            KB.append(idx + "\t" + "CONTAINS" + "\t" + top_pred)
    s0_lemm = set(SPACY_ACCESSOR.lemmatize(s0))
    o0_lemm = set(SPACY_ACCESSOR.lemmatize(o0))
    s1_lemm = set(SPACY_ACCESSOR.lemmatize(s1))
    o1_lemm = set(SPACY_ACCESSOR.lemmatize(o1))
    if len(s1_lemm.intersection(s0_lemm)):
        KB.append(idx + "\tINSUBJ\t" + s1 + "#" + idx)
    if len(s1_lemm.intersection(o0_lemm)):
        KB.append(idx + "\tINOBJ\t" + s1 + "#" + idx)
    if len(o1_lemm.intersection(s0_lemm)):
        KB.append(idx + "\tINSUBJ\t" + o1 + "#" + idx)
    if len(o1_lemm.intersection(o0_lemm)):
        KB.append(idx + "\tINOBJ\t" + o1 + "#" + idx)
    KB.append(s1 + "#" + idx + "\t" + p1 + "\t" + o1 + "#" + idx)
    cn_predicates.add(p1)


print(",".join(cn_predicates))


with open(OUTPUT_FILENAME, "w") as f:
    f.write("\n".join(KB) + "\n")
