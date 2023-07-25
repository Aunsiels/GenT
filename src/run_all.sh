mkdir -p data

from_file=$1
to_file=$2
output_kb=$3


alignment_file="data/alignment.txt"

echo "Computing the alignment"

python3 alignement.py $to_file $from_file $alignement_file

awk '{if(rand()<0.9) {print > data/train.txt} else {print > data/test.txt}}' $alignment_file

echo "Learning the mapping model"

bash learn.sh data/train.txt data/test.txt data/model

echo "Evaluating based on test file"

python3 evaluate.py data/model 10 data/test.txt evaluate

echo "Generating mapping"

python3 evaluate.py data/model 10 $from_file generate data/generated_alignments.tsv

echo "Generating the final knowledge base"

python3 turn_generation_into_kb.py data/generated_alignments.tsv $from_file $output_kb

echo "Compute final metrics"

python3 metrics.py $output_kb $to_file data/train.txt data/test.txt
