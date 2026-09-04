# syntax=docker/dockerfile:1.7

FROM docker.io/verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d

ARG AGENT_R1_REVISION=b124aa46534cbf2fb8bc8af11405774984c42ac7
ARG VERL_REVISION=f9c855f7cf04d603c9546bc01776c74806a879c1

RUN git clone https://github.com/verl-project/verl.git /opt/verl \
    && git -C /opt/verl checkout "${VERL_REVISION}" \
    && python3 -m pip install --no-deps --no-cache-dir -e /opt/verl

RUN git clone https://github.com/AgentR1/Agent-R1.git /opt/Agent-R1 \
    && git -C /opt/Agent-R1 checkout "${AGENT_R1_REVISION}"

RUN python3 -m pip install --no-deps --no-cache-dir \
    FlagEmbedding==1.3.5 \
    faiss-cpu==1.12.0 \
    huggingface-hub==0.34.4

COPY . /workspace/deep-research-rl
RUN python3 -m pip install --no-deps --no-cache-dir -e /workspace/deep-research-rl

ENV DEEP_RESEARCH_RL_CONTAINER_IMAGE="docker.io/verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d" \
    HYDRA_FULL_ERROR=1 \
    PYTHONPATH="/workspace/deep-research-rl/src:/opt/Agent-R1" \
    TOKENIZERS_PARALLELISM=false \
    VLLM_USE_V1=1

WORKDIR /opt/Agent-R1
