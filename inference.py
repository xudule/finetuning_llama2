from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
import argparse

def inference(input, model, tokenizer):
    encoded_input = tokenizer.encode(input, return_tensors="pt", truncation=True, max_length=1000)

    # Increase the max length if longer response is needed, but the generation time will be longer.
    output_raw = model.generate(input_ids=encoded_input, max_length=100)
    output_decoded = tokenizer.batch_decode(output_raw, skip_special_tokens=True)

    return output_decoded[0][len(input):]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="original model name", type=str, default="meta-llama/Llama-2-7b-chat-hf", nargs='?')
    parser.add_argument("--finetuned_model", help="finetuned model name", type=str, default="stable_model", nargs='?')
    parser.add_argument("--question", help="Provide your question to the model", type=str, default="Which location in Vietnam do you like the most?", nargs='?')
    args = parser.parse_args()

    set_seed(42)
    input = args.question

    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #Do we need any pad and truncation for token?

    print("Question: ", input)
    print("Before Finetuning: ")
    print("Response: ", inference(input, model, tokenizer))
    
    # Load the model
    set_seed(42)
    model_dir = args.finetuned_model
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    print("*"*50)
    print("After Finetuning: ")
    print("Reponse:")
    print(inference(input, model, tokenizer))
