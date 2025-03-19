import sys
import time
import torch

from loguru import logger

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def init_loguru():
    logger.remove()
    # NOTE: 추후 로거에 HOSTNAME과 같은 정보를 내포하는게 좋을지?
    format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <fg #0000ff>"
        + time.tzname[0]
        + "</fg #0000ff> | <level>{level: <8}</level> | <bg #0000ff>{name}</bg #0000ff>:<fg #0000ff>{function}</fg #0000ff>:<fg #0000ff>{line}</fg #0000ff> - <level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=format,
        filter=lambda record: record["level"].name == "DEBUG",
        colorize=True,
    )
    logger.add(
        sys.stdout,
        format=format,
        filter=lambda record: record["level"].name != "DEBUG",
        colorize=True,
    )

init_loguru()

# Settings
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
LORA_CONFG = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)
BNB_CONFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
BASE_PATH = "./output/model"
ADAPTER_PATH = "./output/adapter"

# model, tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    # quantization_config=BNB_CONFG,
    torch_dtype=torch.float16,
    device_map = 'auto'
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# save to local path
model.save_pretrained(BASE_PATH)
tokenizer.save_pretrained(BASE_PATH)
logger.debug(f"default model: {model=}")


peft_model = get_peft_model(
    model=model, 
    peft_config=LORA_CONFG,
    adapter_name="mnc_adapter"
)
logger.debug(f"LoRA model: {peft_model=}")

peft_model.save_pretrained(ADAPTER_PATH)
print(peft_model.print_trainable_parameters())


