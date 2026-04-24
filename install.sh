conda create -n omni_inference python=3.12 -y
conda activate omni_inference
pip install uv
uv pip install simulstream simuleval jupyter nvitop soundfile pycryptodome qwen-omni-utils vllm[audio] simalign --torch-backend=auto