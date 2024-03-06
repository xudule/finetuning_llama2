from transformers import TrainingArguments, AutoModelForCausalLM, Trainer, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from transformers import set_seed
from peft import LoraConfig, get_peft_model
# import torch

from inference import *
from dataset import *

set_seed(42)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name",
        help="original model name", type=str,
        default="meta-llama/Llama-2-7b-chat-hf", nargs='?')
parser.add_argument("--low_memory_mode",
        help="Some GPU has limited memory. Apply quantization to avoid out of memory issue.",
        type=int, default=1, nargs='?')
parser.add_argument("--early_stopping",
        help="Use early stopping. Deactivate by default.",
        type=int, default=0, nargs='?')

args, unknown = parser.parse_known_args()

model_name = args.model_name

start_time = time.time()
if args.low_memory_mode:
    model = AutoModelForCausalLM.from_pretrained(model_name,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            # torch_dtype=torch.bfloat16
            )
else:
    model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"Loading model time: {time.time() - start_time} seconds")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def print_tokenizer_info(t):
    vocab = t.get_vocab()
    print("Vocabulary size:", len(vocab))

print_tokenizer_info(tokenizer)

qa_split = read_and_tockenize_dataset(tokenizer)
qa_split = split_train_test(qa_split)

model_dir = "test_trainer"
training_args = TrainingArguments(output_dir=model_dir,
                                  num_train_epochs=300,
                                  logging_steps=20,
                                  #fp16=True,
                                  evaluation_strategy = "steps",
                                  save_strategy = "steps",
                                  save_steps=40,
                                  save_total_limit=4,
                                  eval_steps=20,
                                  load_best_model_at_end = True if args.early_stopping else False,
                                  )

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=qa_split['train'],
    eval_dataset=qa_split['test'],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)] if args.early_stopping else None,
    # compute_metrics=compute_metrics,
)

start_time = time.time()
trainer.train()
print(f"Training time: {(time.time() - start_time)/60} minutes")

#save the model
trainer.save_model(model_dir)