with open('prompt_st', 'r') as f:
    prompts = f.readlines()
    prompts = [prompt.split('. ')[1].strip().replace('"', "") for prompt in prompts]
    print(prompts)
