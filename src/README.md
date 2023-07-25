# Commonsense Mapping

## Complete Process

We provide a script to run the entire pipeline: run\_all.sh.

It takes three arguments:

    * The KB we want to map, as a TSV file containing four columns: subject, predicate, object, score
    * The KB we want to imitate the schema (like ConceptNet), as a TSV file containing four columns: subject, predicate, object, score
    * The output KB

## Making the alignment

We provide the python script alignment.py. It takes three arguments:

    * The KB we want to predict, as a tsv
    * The KB we want to translate, as a tsv
    * The output file

```bash
python3 alignment.py conceptnet.tsv kb.tsv alignment.txt
```


## Learning the model

You first need to clone the transformer repository [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers). Inside this repository, create a python virtual environment called venv and install the required libraries. Then, you can use the script learn.sh. In this file, you can configure the location of the transformer repository, the cache directory and the parameters for finetuning GPT2 (base model, epochs, ...).

The learn.sh script takes three arguments: the training file, the testing file and the output directory.

## Evaluation

We provide a python script evaluate.py to evaluate the generations based on a test file. It takes four arguments in this case:

 * The model name or directory
 * The number of examples to consider per statements
 * The test file
 * The keyword "evaluate"

```bash
python3 evaluate.py my_model 10 test.txt evaluate
```

## Generation

We provide a python script evaluate.py to generate mappings from a file. It takes five arguments in this case:

 * The model name or directory
 * The number of examples to consider per statements
 * The file containing OpenIE triples, separated by a tab
 * The keyword "generate"
 * The output file

```bash
python3 evaluate.py my_model 10 kb.tsv generate output_generation.tsv
```

The output file is a TSV file, where the first column is the original triple and the others are the generations.


## From Generations to KB

We provide the script turn\_generation\_into\_kb.py that turns the generations from the previous step into a final knowledge base. It takes three arguments:

    * The generation file
    * The orginal KB, composed of four columns separated with a tab: subject, predicate, object, score
    * The output file where to write the results


```bash
python3 turn_generation_into_kb.py output_generation.tsv kb.tsv result_kb.tsv
```


## Metrics

We provide a script to evaluate the precision and overlap with ConceptNet automatically: metrics.py. It requires at least two arguments:

    * The knowledge base you want to evaluate. The format is the following: a subject, predicate, object triple separated by a comma, then a tab, then the frequency/score for this triple. The file must be sorted by score/frequency.
    * ConceptNet, formatted with a subject, predicate, object, score separated by a tab.
    * A file of files used for training (they must contain the alignement, with the [SEP] token.


```bash
python3 metrics.py kb.tsv conceptnet.tsv train.txt test.txt
```
