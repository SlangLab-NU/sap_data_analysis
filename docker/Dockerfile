FROM nvcr.io/nvidia/nemo:23.10

# Add only what's definitely missing
RUN pip install --no-cache-dir \
    pandas \
    jiwer \
    tqdm

RUN pip install styletts2

WORKDIR /workspace