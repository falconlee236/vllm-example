import torch

from transformers import pipeline, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel

ADAPTER_PATH = "./output/adapter/mnc_adapter"
BASE_PATH = "./output/model"
BNB_CONFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# input
text = "Who is a Elon Musk?"

model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    quantization_config=BNB_CONFG,
    torch_dtype=torch.float16,
    device_map = 'auto',
)
print(f"base model = {model=}")
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
default_generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16
)
print(f"this is base model result: {default_generator(text)}")



lora_model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    quantization_config=BNB_CONFG,
    torch_dtype=torch.float16,
    device_map = 'auto',
)
print(f"lora model = {lora_model=}")


lora_generator = pipeline(
    task="text-generation",
    model=lora_model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16
)
print(f"this is lora model result: {lora_generator(text)}")

