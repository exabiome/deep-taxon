REPO_DIR=deep-index # change this if necessary, this should be the repository directory
ENV_NAME=exabiome_38

conda create -n $ENV_NAME python=3.8
conda activate $ENV_NAME
conda install pytorch-lightning -c conda-forge -n $ENV_NAME

# at this point, make sure hte python command is pointing to the one in your conda environment
#  if it is not, do the following:
# module unload python
# conda deactivate
# conda activate $ENV_NAME

cd $REPO_DIR
python setup.py build
python setup.py develop 
# if the last command failed, try running it again
