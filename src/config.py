import json

def read_config_file(json_path):
        with open(json_path, 'r') as f:
            config = json.load(f)
        return config

class FinetuningConfig:
    def __init__(self, json_config_path):
        self.json_config = read_config_file(json_config_path)
        self.training_args = self.json_config["TrainingArguments"]
        self.lora_config = self.json_config["peft"]["lora"]


if __name__ == "__main__":
    from transformers import TrainingArguments
    config = FinetuningConfig("config/low_memory_mode.json")
    print(config.training_args)

    from peft import LoraConfig
    lora_config = config.lora_config
    print(lora_config)
    peft_config = LoraConfig(**lora_config)
    #print(peft_config)