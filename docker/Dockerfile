FROM mirrors.tencent.com/tencent-light/py3.8.12-cuda11.2.2-cudnn8.1.1.33-1-nccl2.8.4-1-tf2.4.3-torch1.9.0-torchvision0.10.0-torchaudio0.9.0-lightcc3.0.0-light3.0.0-devel:latest

RUN pip3 install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN git clone https://github.com/facebookresearch/detectron2.git

ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN python3 -m pip install -e detectron2

COPY mask2former/modeling/pixel_decoder/ops /root/pixel_decoder/ops
WORKDIR /root/pixel_decoder/ops
RUN python3 setup.py build install

RUN pip3 install --no-cache-dir git+https://github.com/kornia/kornia

WORKDIR /home