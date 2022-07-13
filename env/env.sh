ENV_NAME=${1:?Please specify an environment name}
conda create --yes -n $ENV_NAME pytorch torchvision torchaudio cudatoolkit=11.3 torchtext -c pytorch
# conda install --yes -n $ENV_NAME -c conda-forge "h5py>=3.6=mpi_*"
conda run -n $ENV_NAME pip install --no-cache-dir hdmf==3.2.1 pytorch_lightning==1.6.3 scikit-bio==0.5.7 wandb==0.12.16 torch_optimizer==0.3.0
