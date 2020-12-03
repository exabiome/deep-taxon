ENV_NAME=${1:?"please provide environment name"}
module load open-ce/0.1.0
conda create --name $ENV_NAME --clone open-ce-0.1.0
conda activate $ENV_NAME
pip install hdmf==2.2.0
pip install seaborn
conda install -c conda-forge -n $ENV_NAME umap-learn
