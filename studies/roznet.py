from exabiome.nn.optim import optuna_run

cmdline = "roznet ../input/ar122_r89.genomic.small.deep_index.input.h5 ./optuna -L -W 1000 -S 1000 -e 10 -b 256 --lr 0.01"

optuna_run(cmdline)
