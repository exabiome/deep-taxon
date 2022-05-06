FROM nvcr.io/nvidia/pytorch:22.01-py3
RUN pip install --no-cache-dir hdmf==3.1.1 pytorch_lightning==1.5.8 scikit-bio
