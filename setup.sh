pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
pip install "fastapi[all]"
pip install sse-starlette
mkdir model
cd model
#wget https://huggingface.co/TheBloke/vicuna-7B-v1.5-16K-GGML/resolve/main/vicuna-7b-v1.5-16k.ggmlv3.q2_K.bin
wget https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q4_0.gguf 
cd ..
export MODEL=model/codellama-7b.Q4_0.gguf  
python3 -m llama_cpp.server --model $MODEL  --n_gpu_layers 1