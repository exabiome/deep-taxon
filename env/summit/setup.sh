ENV_NAME=${1:?"please provide environment name"}
module load ibm-wml-ce/1.7.1.a0-0
conda create --name $ENV_NAME --clone ibm-wml-ce-1.7.1.a0-0
conda activate $ENV_NAME
module load hdf5/1.10.4
pip uninstall h5py
pip install --no-binary=h5py h5py
pip install hdmf==2.2.0
pip install pytorch-lightning
