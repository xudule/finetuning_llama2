from transformers import TrainingArguments, AutoModelForCausalLM, Trainer, AutoTokenizer
from transformers import set_seed
from peft import LoraConfig

from inference import *
from dataset import *

set_seed(42)

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def print_tokenizer_info(t):
    vocab = t.get_vocab()
    print("Vocabulary size:", len(vocab))

print_tokenizer_info(tokenizer)

qa_split = read_and_tockenize_dataset(tokenizer)

input = "Where are you from?"
print("="*20, inference(input, model, tokenizer), "="*20)

model_dir = "test_trainer"
training_args = TrainingArguments(output_dir=model_dir,
                                  num_train_epochs=1,
                                  evaluation_strategy="epoch")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

model.add_adapter(peft_config)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=qa_split['train'],
    eval_dataset=qa_split['test'],
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics,
)
trainer.train()

#save the model then reload it
set_seed(42)
trainer.save_model(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

print("*"*50)
print(inference(input, model, tokenizer))
