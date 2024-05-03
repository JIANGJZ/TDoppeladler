FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workspace/
ENV PYTHONPATH /workspace/
#COPY requirements.txt /workspace/
COPY TDoppeladler /workspace/fuse/

RUN pip install -r /workspace/fuse/requirements.txt

RUN pip install -e /workspace/fuse
