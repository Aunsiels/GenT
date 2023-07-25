from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import torch
from sys import argv

# Takes two arguments:
#   * The name of the base model (for the tokenizer)
#   * The finetuned model path

base_model = argv[1]
model_str = argv[2]

tokenizer = LlamaTokenizer.from_pretrained(base_model)

model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model,
    model_str,
    torch_dtype=torch.float16,
)

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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
