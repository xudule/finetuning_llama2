from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed, pipeline
import argparse
import time
import os, sys

from dataset import *

def inference(input, model, tokenizer, max_output_length=500):
    mypipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    sequences = mypipeline(
    input,
    eos_token_id=tokenizer.eos_token_id,
    max_length=max_output_length,
    )
    return sequences[0]['generated_text'][len(input):]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", help="original model name", type=str, default="meta-llama/Llama-2-7b-chat-hf", nargs='?')
    parser.add_argument("--finetuned_model", help="finetuned model name", type=str, default="stable_model", nargs='?')
    parser.add_argument("--question", help="Provide your question to the model", type=str, default="Which location in Vietnam do you like the most?", nargs='?')
    parser.add_argument("--low_memory_mode",
        help="Some GPU has limited memory. Apply quantization to avoid out of memory issue. Only applied when --device = gpu",
        type=int, default=1, nargs='?')
    parser.add_argument("--device", help="Running on device: cpu, gpu", type=str, default="gpu", nargs='?')
    args, unknown = parser.parse_known_args()

    set_seed(42)

    input = args.question
    model_name = args.model_name
    device = "cuda"
    quant = False
    if args.device != "gpu":
        device = "cpu"
    elif args.low_memory_mode:
        quant = True

    def print_separation_line():
        return print('-'*50)

    print_separation_line()
    print("Configurations:")
    print(f"Original model: {model_name}")
    print(f"Finetuned model: {args.finetuned_model}")
    print(f"device: {device}")
    print(f"Quantization: {'yes' if quant else 'no'}")
    print_separation_line()

    print("Question: ", input)
    print("Before Finetuning: ")
    input = generate_prompt(input)

    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"Time Loading model: {time.time() - start_time} seconds")

    if quant:
        set_seed(42)
        start_time = time.time()
        model = model.half()
        print(f"Time Quantizing model: {time.time() - start_time} seconds")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #Do we need any pad and truncation for token?

    start_time = time.time()
    model.to(device)
    print(f"Time load to {device}: {time.time() - start_time} seconds")

    start_time = time.time()
    print("Response: ", inference(input, model, tokenizer))
    print(f"Inference time: {time.time() - start_time} seconds")

    model_dir = args.finetuned_model
    if not os.path.isdir(model_dir):
        print("Finetuned model in current directory does not exist")
        sys.exit(1)

    print_separation_line()
    print("After Finetuning: ")

    set_seed(42)
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    if quant:
        model = model.half()
    print(f"Time Loading model: {time.time() - start_time} seconds")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    start_time = time.time()
    model.to(device)
    print(f"Time load to {device}: {time.time() - start_time} seconds")

    print("Reponse:")
    start_time = time.time()
    print(inference(input, model, tokenizer))
    print(f"Time inference: {time.time() - start_time} seconds")
    sys.exit(0)
