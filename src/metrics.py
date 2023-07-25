"""
The script evaluate the overlap of a knowledge base mapped to ConceptNet with ConceptNet, and the precision@k based on ConceptNet.
It takes at least two arguments:
    * The knowledge base you want to evaluate. The format is the following: a subject, predicate, object triple separated by a comma, then a tab, then the frequency/score for this triple. The file must be sorted by score/frequency.
    * ConceptNet, formatted with a subject, predicate, object, score separated by a tab.
    * A file of files used for training (they must contain the alignement, with the [SEP] token.
"""


import os
from sys import argv


KB = argv[1]
CONCEPTNET_PATH = argv[2]
if len(argv) > 3:
    TRAINING_FILES = argv[3:]
else:
    TRAINING_FILES = []


cn = set()

with open(CONCEPTNET_PATH) as f:
    for line in f:
        s,p,o,_ = line.strip().split("\t")
        cn.add((s + "," + p + "," + o).lower().replace("_", ""))


cn2 = set()
for filename in TRAINING_FILES:
    with open(filename) as f:
        for line in f:
            line = line.replace("<|endoftext|>", "")
            if "[SEP]" in line:
                spo = line.strip().split("[SEP]")[1].lower().replace("\t", ",").replace(" ", "")
            else:
                line = line.strip().lower().replace(" ", "").split("\t")
                spo = line[0].split(",")
                spo[1] = line[1]
                spo = ",".join(spo)
            cn2.add(spo)


def compute_overlap(filename):
    preds = set()
    last_index = -1
    last_cn_triple = ""
    rrs = []
    with open(filename) as f:
        for i, line in enumerate(f):
            split_temp = line.lower().split("\t")
            if len(split_temp) != 2:
                continue
            spo, _ = split_temp
            spo = spo.replace(" ", "")
            preds.add(spo)
            if spo in cn:
                last_index = len(preds)
                rrs.append(1.0 / last_index)
                last_cn_triple = spo
    preds_in_cn = cn.intersection(preds)
    print("Total number of predictions:", len(preds))
    print("Position last prediction in ConceptNet:", last_index, "(", last_index / len(preds) * 100 ,")")
    print(last_cn_triple)
    print("MRR:", sum(rrs) / len(rrs))
    print("Preds in ConceptNet:", len(preds_in_cn), ",", len(preds_in_cn) / len(cn) * 100, "%")
    not_in_overlap = preds_in_cn.difference(cn2)
    preds_in_train = preds_in_cn.intersection(cn2)
    print("Preds in ConceptNet and not in overlap:", len(not_in_overlap), len(not_in_overlap) / (len(cn) - len(preds_in_train)) * 100, "%")
    return preds_in_cn, not_in_overlap



def p_at_k(filename, at=-1):
    res = []
    with open(filename) as f:
        for i, line in enumerate(f):
            spo, _ = line.lower().split("\t")
            spo = spo.replace(" ", "")
            if i == at:
                break
            if spo in cn:
                res.append(1)
            else:
                res.append(0)
    return sum(res) / len(res) * 100



preds_in_cn, not_in_overlap = compute_overlap(KB)


print("Precision: ", p_at_k(KB))
for i, at in enumerate([1, 5, 10, 20, 50, 100, 200, 300, 1000, 5000, 10000]):
    temp = p_at_k(KB, at)
    print("Precision at:", at, ":", temp)
    recall = temp * at / len(cn)
    print("Recall at:", at, ":", recall)

