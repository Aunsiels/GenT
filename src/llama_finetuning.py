import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from sys import argv
 
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
 
import torch
 
 
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device", DEVICE)

BASE_MODEL = argv[1]
CUTOFF_LEN = 256

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
 
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
 
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        return_tensors=None,
    )
    #if (
    #    result["input_ids"][-1] != tokenizer.eos_token_id
    #    and len(result["input_ids"]) < CUTOFF_LEN
    #    and add_eos_token
    #):
    #    result["input_ids"].append(tokenizer.eos_token_id)
    #    result["attention_mask"].append(1)

    # result["labels"] = result["input_ids"].copy()
    return result

train_data = []
test_data = []

with open(argv[2]) as f:
    for line in f:
        train_data.append(tokenize(line.strip().replace("<|endoftext|>", "")))


with open(argv[3]) as f:
    for line in f:
        test_data.append(tokenize(line.strip().replace("<|endoftext|>", "")))


val_data = test_data

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 3000
OUTPUT_DIR = "experiments"


model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard",
    weight_decay=0.1,
    lr_scheduler_type="cosine"
)

data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt"
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)

model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

model = torch.compile(model)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
