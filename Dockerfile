FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    unzip \
    ffmpeg \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
ENV ARCH=${TARGETARCH}
# Install Miniconda
RUN if [ "$ARCH" = "arm64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    fi && \
    wget --quiet $MINICONDA_URL -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Create the conda environment and install Python 3.9
RUN conda create -n infinisst -y python=3.9

# Activate conda env for future RUN commands
SHELL ["conda", "run", "-n", "infinisst", "/bin/bash", "-c"]

# Install Python packages
RUN pip install --upgrade pip==23.3 

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 

RUN pip install transformers==4.47.0 evaluate jiwer lightning accelerate deepspeed rotary_embedding_torch torchtune sentence-transformers wandb tensorboardX matplotlib soundfile simuleval jupyter jieba unbabel-comet simalign praat-textgrids peft
#     pip install flash-attn --no-build-isolation

    
    # && \
    # pip install flash-attn --no-build-isolation    

# Install fairseq
RUN git clone https://github.com/facebookresearch/fairseq.git && \
    mv fairseq fairseq-0.12.2 && \
    cd fairseq-0.12.2 && \
    git checkout 0.12.2-release && \
    pip install -e .

# Set working directory
WORKDIR /app

# Copy in the model downloader and other repo files
COPY download_models.sh /app/
COPY . /app/

RUN chmod +x /app/download_models.sh && \
    if [ -d "/app/iwslt25/InfiniSST" ]; then chmod +x /app/iwslt25/InfiniSST/*.sh; fi

# 🔽 Add this line to actually run the script
RUN bash /app/download_models.sh

# Set env variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default entry: shell with the conda env
CMD ["/bin/bash"]