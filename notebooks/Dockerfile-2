# this docker file downloads the model when you run the docker image in a container for the very first time. after that it will pick up automatically from the path
FROM nvcr.io/nvidia/pytorch:22.12-py3
USER root
ENV PYTHONUNBUFFERED=TRUE
RUN python3 -m pip install --upgrade pip
RUN pip uninstall torch -y
RUN pip install vllm
RUN pip install --force-reinstall -v "pydantic==2.2.0"
ENV hugging_face_token=none
RUN mkdir meta-llama
CMD huggingface-cli login --token $hugging_face_token && python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --host 0.0.0.0 --port 32768 --max-num-batched-tokens 4096 --use-np-weights  --download-dir  meta-llama/