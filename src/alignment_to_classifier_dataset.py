from sys import argv

predicate_mapping = {
    "Created": "created",
    "HasFirstSubevent": "has first subevent",
    "HasSubevent": "has subevent",
    "CausesDesire": "causes desire",
    "InstanceOf": "instance of",
    "MotivatedByGoal": "motivated by goal",
    "DefinedAs": "defined as",
    "MadeOf": "made of",
    "Causes": "causes",
    "HasPrerequisite": "has prerequisite",
    "HasA": "has a",
    "PartOf": "part of",
    "CreatedBy": "created by",
    "Desires": "desires",
    "ReceivesAction": "receives action",
    "UsedFor": "used for",
    "DistinctFrom": "distinct from",
    "HasProperty": "has property",
    "AtLocation": "at location",
    "LocatedNear": "located near",
    "HasLastSubevent": "has last subevent",
    # Only ascent
    "CapableOf": "capable of",
    "SimilarTo": "similar to",
    "SymbolOf": "symbol of",
    "RelatedTo": "related to",
    "IsA": "is a"
}

inverse_mapping = {}
for x, y in predicate_mapping.items():
    inverse_mapping[y] = x

INITIAL_ALIGNMENTS = argv[1]
OUTPUT_FILE = argv[2]

res = []
with open(INITIAL_ALIGNMENTS) as f:
    for line in f:
        line = line.strip().replace("<|endoftext|>", "").split("[SEP]")
        spo_from = line[0].split("\t")
        spo_to = line[1].split("\t")
        if spo_from[0] == spo_to[0] and spo_from[2] == spo_to[2]:
            res.append(",".join(spo_from) + "\t" + inverse_mapping[spo_to[1]] + "\n")


with open(OUTPUT_FILE, "w") as f:
    f.write("".join(res))
