FROM nvcr.io/nvidia/pytorch:22.04-py3
RUN pip install --no-cache-dir hdmf==3.2.1 pytorch_lightning==1.6.3 scikit-bio==0.5.7 wandb==0.12.16 torch_optimizer==0.3.0
