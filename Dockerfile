FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN pip3 install --no-cache-dir sagemaker-training

COPY requirements.txt /opt/ml/requirements.txt

RUN pip3 install --no-cache-dir -r /opt/ml/requirements.txt

WORKDIR /opt/ml/code

ENV SAGEMAKER_PROGRAM train.py