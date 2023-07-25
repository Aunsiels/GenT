""" Run a LM-based classifier

It takes as arguments:
    * An annotation file for an OpenKB. It is formatted as follows: first, a subject, predicate and object separated by
    a comma. Then, a tab and the sequence of annotations (predicates in the target KB), sepated by a tab.
    * A KB to translate, tab separated with four columns: subject, predicate, object, score
    * An output file for the translation

"""
import json

import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import sys

from predicate_conceptnet import predicate_mapping

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


KB_ANNOTATION = sys.argv[1]
KB_TO_TRANSLATE = sys.argv[2]
OUTPUT_FILENAME = sys.argv[3]
OUTPUT_GENERATIONS_FILENAME = OUTPUT_FILENAME + ".generations"


DO_EVAL = False
DO_TRAIN = False
DO_PRED = True
USE_CUDA = torch.cuda.is_available()


if DO_TRAIN:
    data = []
    all_annotations = set()

    with open(KB_ANNOTATION) as f:
        for line in f:
            line = line.strip().split("\t")
            if len(line) < 2:
                continue
            for annotation in line[1:]:
                if annotation and annotation != "Nothing":
                    data.append([line[0], annotation])
                    all_annotations.add(annotation)

    annotation_to_label = dict()
    label_to_annotation = list(all_annotations)
    for i, annotation in enumerate(label_to_annotation):
        annotation_to_label[annotation] = i

    for example in data:
        example[1] = annotation_to_label[example[1]]

    if DO_EVAL:
        idx_split = int(len(data) * 0.8)
    else:
        idx_split = len(data)

    # Preparing train data
    train_data = data[:idx_split]
    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text", "labels"]

    if DO_EVAL:
        # Preparing eval data
        eval_data = data[idx_split:]
        eval_df = pd.DataFrame(eval_data)
        eval_df.columns = ["text", "labels"]
    else:
        eval_df = None

    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=1)

    # Create a ClassificationModel
    model = ClassificationModel(
        'bert',
        'bert-large-cased',
        num_labels=len(all_annotations),
        args=model_args,
        use_cuda=USE_CUDA
    )

    # Train the model
    model.train_model(train_df)

    if DO_EVAL:
        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(eval_df)
        print("Result", result)

    with open("outputs/label_to_annotation.json", "w") as f:
        f.write(json.dumps(label_to_annotation))
else:
    model = ClassificationModel(
        'bert',
        'outputs',
        use_cuda=USE_CUDA
    )
    with open("outputs/label_to_annotation.json") as f:
        label_to_annotation = json.load(f)

if DO_PRED:
    KB = []
    with open(KB_TO_TRANSLATE) as f:
        for line in f:
            line = line.strip().split("\t")
            KB.append(",".join(line[:3]))

    idx = 0
    predictions = []
    while idx < len(KB):
        KB_temp = KB[idx: idx+500]
        idx += 500
        print(idx)
        # Make predictions with the model
        predictions_temp, _ = model.predict(KB_temp)
        predictions += predictions_temp

    TRANSLATION = []
    GENERATIONS = []
    with open(KB_TO_TRANSLATE) as f:
        for i, line in enumerate(f):
            line = line.strip().split("\t")
            if len(line) != 4:
                continue
            TRANSLATION.append(line[0] + "," + label_to_annotation[predictions[i]] + "," + line[2] + "\t" + line[3] + "\n")
            GENERATIONS.append(line[0] + "," + line[1] + "," + line[2] + "\t" +
                               line[0] + "," + predicate_mapping.get(label_to_annotation[predictions[i]], " ")
                               + "," + line[2]
                               + "\n")

    with open(OUTPUT_FILENAME, "w") as f:
        f.write("".join(TRANSLATION))
    with open(OUTPUT_GENERATIONS_FILENAME, "w") as f:
        f.write("".join(GENERATIONS))
