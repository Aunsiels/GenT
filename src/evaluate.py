# Evaluate and generate a model
# It takes 5 arguments
#  * The model name or directory
#  * The number of examples to consider per statements
#  * The test file when evaluating, or a file containg triples, separated by a tab, or a file containing subjects
#  * Either "evaluate", "generate", "subject", or "predicate"
#  * Where to write the results of the generation


from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sys import argv
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL = argv[1]
AT = argv[2]
TEST_FILE = argv[3]
ACTION = argv[4]

tokenizer = GPT2Tokenizer.from_pretrained(str(MODEL))
model = GPT2LMHeadModel.from_pretrained(str(MODEL), pad_token_id=tokenizer.eos_token_id).to(DEVICE)


def generate(text, n=5, length_multiplier=3, add_score=False):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
    length = len(input_ids[0])
    beam_outputs = model.generate(input_ids, max_length=length * length_multiplier, top_k=40, temperature=1.0,
                                  do_sample=False,
                                  top_p=0.9, repetition_penalty=1.0, num_return_sequences=n, num_beams=n,
                                  early_stopping=True, return_dict_in_generate=True,  output_scores=True)
    print("Output:\n" + 100 * '-')
    res = []
    for i, beam_output, score in zip(range(len(beam_outputs.sequences)), beam_outputs.sequences, beam_outputs.sequences_scores):
        generation = tokenizer.decode(beam_output, skip_special_tokens=True)
        if add_score:
            generation += "\t" + str(score.item())
        res.append(generation)
        print("{}, {}: {}".format(i, score, res[-1]))
    return res


def generate_csk(s, p, o, n=5):
    return generate(s + "\t" + p.replace("_", " ") + "\t" + o + "[SEP]", n)


def evaluate_precision(at=1):
    pos = 0
    neg = 0
    with open(TEST_FILE) as f:
        for line in f:
            line = line.strip()[:-len("<|endoftext|>")]
            print(line)
            source, target = line.split("[SEP]")
            source += "[SEP]"
            preds = generate(source, at)
            print("-" * 100)
            print("Target:", target)
            found = False
            for pred in preds:
                print("Prediction", pred.split("[SEP]")[1])
                if pred == line:
                    found = True
            if found:
                pos += 1
            else:
                print("Nothing matches")
                neg += 1
            print("Current precision at", at, ":", pos / (pos + neg))
    return pos / (pos + neg)


def generate_examples(at=1):
    res = []
    with open(TEST_FILE) as f:
        for i, line in enumerate(f):
            line = line.strip().split("\t")
            if len(line) < 3:
                continue
            s = line[0]
            p = line[1]
            o = line[2]
            preds = generate_csk(s, p, o, at)
            temp = s + "," + p + "," + o
            for pred in preds:
                temp += "\t" + pred.split("[SEP]")[1].replace("\t", ",")
            res.append(temp + "\n")
    with open(argv[5], "w") as f:
        f.write("".join(res))


ALL_RELATIONS = [
    "at location",
    "capable of",
    "causes",
    "causes desire",
    "created by",
    "defined as",
    "desires",
    "distinct from",
    "has a",
    "has first subevent",
    "has last subevent",
    "has prerequisite",
    "has property",
    "has subevent",
    "instance of",
    "located near",
    "made of",
    "motivated by goal",
    "part of",
    "receives action",
    "used for"
]


def generate_from_subjects_with_predicates(at=1):
    res = []
    with open(TEST_FILE) as f:
        for i, line in enumerate(f):
            subject = line.strip()
            for predicate in ALL_RELATIONS:
                preds = generate(subject + "\t" + predicate + "\t", at, length_multiplier=3, add_score=True)
                res += preds
    with open(argv[5], "w") as f:
        f.write("\n".join(res))


def generate_from_subjects(at=1):
    res = []
    with open(TEST_FILE) as f:
        for i, line in enumerate(f):
            subject = line.strip()
            preds = generate(subject + "\t", at, length_multiplier=20, add_score=True)
            res += preds
    with open(argv[5], "w") as f:
        f.write("\n".join(res))


if ACTION == "evaluate":
    evaluate_precision(at=int(AT))
elif ACTION == "generate":
    generate_examples(at=int(AT))
elif ACTION == "subject":
    generate_from_subjects(at=int(AT))
elif ACTION == "predicate":
    generate_from_subjects_with_predicates(at=int(AT))
