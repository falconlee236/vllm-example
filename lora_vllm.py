from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ADAPTER_PATH = "./output/adapter/mnc_adapter"
BASE_PATH = "./output/model"

text = "Who is a Elon Musk?"
prompts = [
    text,
]

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
)

llm_lora = LLM(
    model=BASE_PATH,
    enable_lora=True,
    gpu_memory_utilization=0.95,
    tensor_parallel_size=4,
)

lora_outputs = llm_lora.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("mnc_adapter", 1, ADAPTER_PATH)
)


print("result with lora")
for output in lora_outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(f"lora_id: {output.lora_request.lora_int_id}, lora_path: {output.lora_request.lora_path}")



"""
result with lora
Prompt: 'Who is a Elon Musk?', Generated text: ' Elon Musk is a South African-born entrepreneur, inventor, and investor. He is the CEO of SpaceX, Tesla, Neuralink, and The Boring Company. He is known for his ambitious goals in areas such as space exploration, electric cars, and renewable energy.'
lora_id: 1, lora_path: ./output/adapter/mnc_adapter
"""