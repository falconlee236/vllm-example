from vllm import LLM, SamplingParams

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


llm = LLM(
    model=BASE_PATH,
    gpu_memory_utilization=0.95,
    tensor_parallel_size=4,
)

outputs = llm.generate(
    prompts,
    sampling_params,
)
        
print("result without lora")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    
"""
result without lora
Prompt: 'Who is a Elon Musk?', Generated text: ' Elon Musk is a South African-born entrepreneur, inventor, and investor. He is the CEO of SpaceX, Tesla, Neuralink, and The Boring Company. He is known for his ambitious goals in areas such as electric cars, space exploration, and renewable energy.'
"""