ENV_NAME=${1:?"please provide environment name"}
module load ibm-wml-ce/1.7.1.a0-0
conda create --name $ENV_NAME --clone ibm-wml-ce-1.7.1.a0-0
conda install -n $ENV_NAME -c conda-forge hdmf
pip install pytorch-lightning
