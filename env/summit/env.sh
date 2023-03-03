ENV_NAME=${1:?Please specify an environment name}

module load open-ce-olcf/1.5.2-py39-0
conda create --clone -n $ENV_NAME --clone open-ce-olcf-1.5.2-py39-0 --yes
conda install -c conda-forge umap-learn -n $ENV_NAME
