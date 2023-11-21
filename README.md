# Mapping and Cleaning Open Commonsense Knowledge Bases with Generative Translation

This repository contains the code for the paper __Mapping and Cleaning Open Commonsense Knowledge Bases with Generative Translation__. To cite our work:

```
@InProceedings{10.1007/978-3-031-47240-4_20,
author="Romero, Julien
and Razniewski, Simon",
editor="Payne, Terry R.
and Presutti, Valentina
and Qi, Guilin
and Poveda-Villal{\'o}n, Mar{\'i}a
and Stoilos, Giorgos
and Hollink, Laura
and Kaoudi, Zoi
and Cheng, Gong
and Li, Juanzi",
title="Mapping and Cleaning Open Commonsense Knowledge Bases with Generative Translation",
booktitle="The Semantic Web -- ISWC 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="368--387",
abstract="Structured knowledge bases (KBs) are the backbone of many knowledge-intensive applications, and their automated construction has received considerable attention. In particular, open information extraction (OpenIE) is often used to induce structure from a text. However, although it allows high recall, the extracted knowledge tends to inherit noise from the sources and the OpenIE algorithm. Besides, OpenIE tuples contain an open-ended, non-canonicalized set of relations, making the extracted knowledge's downstream exploitation harder. In this paper, we study the problem of mapping an open KB into the fixed schema of an existing KB, specifically for the case of commonsense knowledge. We propose approaching the problem by generative translation, i.e., by training a language model to generate fixed-schema assertions from open ones. Experiments show that this approach occupies a sweet spot between traditional manual, rule-based, or classification-based canonicalization and purely generative KB construction like COMET. Moreover, it produces higher mapping accuracy than the former while avoiding the association-based noise of the latter. Code and data are available. (https://github.com/Aunsiels/GenT, julienromero.fr/data/GenT)",
isbn="978-3-031-47240-4"
}

```

## Get the Data

Our generated new datasets, Quasimodo-GenT and Ascent-GenT, are available on HuggingFace:
  * Quasimodo-GenT : [https://huggingface.co/datasets/Aunsiels/Quasimodo-GenT](https://huggingface.co/datasets/Aunsiels/Quasimodo-GenT)
  * Ascent-GenT : [https://huggingface.co/datasets/Aunsiels/Ascent-GenT](https://huggingface.co/datasets/Aunsiels/Ascent-GenT)

These are the variants using the LM-based alignment, inverted, and with 10 generations per triple. Please ask if you require other settings.

## Get the models

The models are available on HuggingFace:
  * Quasimodo
    * [Rule-based alignment](https://huggingface.co/Aunsiels/QuasimodoGenT-Rule)
    * [LM-based alignment](https://huggingface.co/Aunsiels/QuasimodoGenT-LM)
    * [LM-based alignment inversed](https://huggingface.co/Aunsiels/QuasimodoGenT-LMINV)
  * Ascent++
    * [Rule-based alignment](https://huggingface.co/Aunsiels/AscentGenT-Rule)
    * [LM-based alignment](https://huggingface.co/Aunsiels/AscentGenT-LM)
    * [LM-based alignment inversed](https://huggingface.co/Aunsiels/AnscentGenT-LMINV)

## Alignment

### Rule Based

The rule-based alignment script is src/alignment.py. It takes three arguments:
    * The KB we want to predict, as a tsv
    * The KB we want to translate, as a tsv
    * The output file

```bash
python3 alignment.py conceptnet.tsv kb.tsv alignment.txt
```

### LM Based

The LM-based alignment script is src/map\_closest\_embeddings.py. It takes three arguments:
    * The embeddings of the first KB as given in compute\_embeddings\_sentence.py
    * The embeddings of the first KB as given in compute\_embeddings\_sentence.py
    * The output filename, that will be composed of three columns (tab-separated): a statement of the first KB, a statement of the second KB, and a distance score between the two.

Executing this script requires a preprocessing step for each KB with the script src/compute\_embeddings\_sentence.py. It takes two arguments:
    * A subject, predicate, object, ... knowledge base, tab-separated.
    * An output filename.

### Train/Test Split

You can create a train/test split with the following command.

```bash
awk '{if(rand()<0.9) {print > data/train.txt} else {print > data/test.txt}}' $alignment_file
```

## Finetuning

### GPT-2

For the GPT-2 finetuning, we used the script provided by HuggingFace, [https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run\_clm.py](run_clm.py). The script with our parameters is provided in src/learn.sh. It takes three arguments: the train file, the test file, and the output directory.

```bash
bash learn.sh data/train.txt data/test.txt data/model
```

### T5

The script for finetuning T5 is provided in src/train\t5.py. It takes three parameters: the train file, the test file, and the output directory.

### LLaMa

The script for finetuning LLaMa is provided in src/llama\_finetuning.py. It takes three arguments: the base model name, the train file, and the test file. The output directory is experiments/ by default.

## KB Generation

### Initial Translation

Using the script src/evaluate.py, we can generate translations for given triples. This script takes five arguments:
  * The model name or directory
  * The number of examples to consider per statement
  * The test file when evaluating, or a file containing triples, separated by a tab, or a file containing subjects
  * Either "evaluate", "generate", "subject" or "predicate"
      * "evaluate" is used to evaluate triple alignment generation (see below)
      * "generate" translates triples into the new schema
      * "subject" generates triples given a subjects
      * "predicate" generates triples given subjects and a list of relations (the ones in ConceptNet by default)
  * Where to write the results of the generation


```bash
python3 evaluate.py data/model 10 $from_file generate data/generated_alignments.tsv
```

For "subject" and "predicate" generations, you need post-processing using the script src/create\_subject\_kb.py. It takes two arguments:
  * The input file
  * The output file

### Merging And Ranking For KB Creation

Next, we need to turn the generations from the previous script into a new knowledge base, with the statements ranked. We use the script src/turn\_generation\_into\_kb.py. It takes four arguments:
    * The generation file
    * The original KB, composed of four columns separated by a tab: subject, predicate, object, score
    * The output file where to write the results
    * The top generations to consider
    * The type of merging to perform (count, weighted, jaccard)
    * Jaccard parameter, if required

```bash
python3 turn_generation_into_kb.py data/generated_alignments.tsv $from_file $output_kb
```

## Evaluation

Some results presented in the paper might differ from the one generated by the scripts here. This is because we normalized the results as explained in the paper.

### Automatic Triple Alignment Metrics

To perform a simple evaluation of the alignment generation using a test set, we can use the same src/evaluate.py script presented above in "evaluate" mode.

```bash
python3 evaluate.py data/model 10 data/test.txt evaluate
```

For a more complete evaluation, you can use the script src/automatic\_alignment\_evaluation.py. It takes two arguments:
  * The generation file
  * The test file


### KB Metrics

We can compute the final metrics using the script src/metrics.py. It takes at least two arguments: 
    * The knowledge base you want to evaluate. The format is the following: a subject, predicate, object triple separated by a comma, then a tab, and then the frequency/score for this triple. The file must be sorted by score/frequency.
    * ConceptNet, formatted with a subject, predicate, object, and score separated by a tab.
    * A file of files used for training (they must contain the alignment, with the [SEP] token.

```bash
python3 metrics.py $output_kb $to_file data/train.txt data/test.txt
```

## Baselines

### Manual Mapping

The script to perform the manual mapping is src/manual\_mapping.py. It takes three arguments:
    * The OpenKB, formatted as subject, predicate, object, score and separated by a tab.
    * The mapping file. It must be a TSV file of two columns. The first one represents the open relationship, and the second one is the translation
    * The output filename

We provide the mappings we used in src/data/quasimodo\_manual\_mapping\_to\_conceptnet.tsv and ascent\_manual\_mapping\_to\_conceptnet.tsv.

After the generation, you must generate the final KB, as explained above.

### Classifier Mapping

The script is src/lm\_classifier.py. It takes three arguments:
    * An annotation file for an OpenKB. It is formatted as follows: first, a subject, predicate, and object separated by a comma. Then, a tab and the sequence of annotations (predicates in the target KB), separated by a tab.
    * A KB to translate, tab-separated with four columns: subject, predicate, object, score
    * An output file for the translation

The alignments generated previously can be adapted using the script src/alignment\_to\_classifier\_dataset.py. It takes two arguments:
    * A path to the old alignments
    * A path to the output file

### Rule Mining

An example of steps to run for the rule mining baseline is shown in src/run\_amie.sh. You will need Amie. You can find it on GitHub [https://github.com/lajus/amie/](https://github.com/lajus/amie/).
