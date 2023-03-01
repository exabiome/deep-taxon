ENV_NAME=${1:?Please specify an environment name}
module load python
conda create --yes -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=11.7 torchtext cython -c pytorch -c nvidia
if [ $? -ne 0 ]; then echo "Failed to create environment $ENV_NAME"; return 1; fi
conda activate $ENV_NAME
MPICC="cc -target-accel=nvidia80 -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
module load cray-hdf5-parallel
HDF5_MPI=ON CC=cc pip install -v --force-reinstall --no-cache-dir --no-binary=h5py --no-build-isolation --no-deps h5py
pip install --no-cache-dir -r requirements.txt
