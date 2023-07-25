relations="has first subevent,instance of,made of,at location,has a,defined as,located near,causes,used for,distinct from,has prerequisite,causes desire,capable of,has last subevent,motivated by goal,desires,has subevent,part of,has property,created by,receives action"


python3 rule_mining.py data/ascentpp/conceptnet_ascentpp_alignment.txt data/ascentpp/ascentpp_rule_mining_base.tsv


java -jar amie3.jar -htr "$relations" -maxad 5 -bexr "$relations" -const data/ascentpp/ascentpp_rule_mining_base.tsv > amie_rules_ascentpp.tsv
# You need to clean manually the output file to remove the debug messages at the top and the bottom

python3 apply_rule_mining.py data/ascentpp.tsv amie_rules_ascentpp.tsv data/ascentpp/ascentpp_conceptnet_rule_mining_unmerged.tsv

python3 merge_rule_mining_generations.py data/ascentpp/ascentpp_conceptnet_rule_mining_unmerged.tsv data/ascentpp/ascentppconceptnet_rule_mining.tsv
