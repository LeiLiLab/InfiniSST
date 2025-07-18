from vllm import LLM, SamplingParams

prompt_template = """You will be provided with instructions, and a sentence in English, and your task is to segment the sentence according to the instructions. Always answer in the following JSON format:{{'segments':List[English]}} 

Instructions: 
1. Break down the sentence into manageable phrases that each contain enough information to be accurately interpreted. 
2. Each phrase should be no more than five words if possible while preserving grammatical integrity and interpretability. 

English: {}"""
sent = "And in some countries -- let me pick Afghanistan -- 75 percent of the little girls don't go to school. And I don't mean that they drop out of school in the third or fourth grade -- they don't go."
prompts = [
    prompt_template.format(sent)
]
sampling_params = SamplingParams(top_k=1)
llm = LLM(model="/compute/babel-13-9/siqiouya/deepseek-r1-distill-qwen-32b/", tensor_parallel_size=2)

results = llm.generate(prompts, sampling_params)
for result in results:
    print(result.text)
