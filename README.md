# exabiome
This package contains executable models for each of the main steps in the neural network training process. There exist
an executable for each of the following steps:

1. Converting data
2. Training networks
3. Network inference
4. Summarizing network outputs

## Installation
To ensure proper functioning of this package, it should be installed in its own conda environment, cloned from
the specification file in the `env` subdirectory.

```bash
git clone git@github.com:exabiome/deep-taxon.git
cd deep-taxon
conda create --name myclone --file `env/python_38.txt`
conda activate myclone
python setup.py install
```

## Commands
All commands can be accessed with the `deep-index` executable. Below is the `deep-index` usage statement, which
lists the available commands.

```bash
Usage: deep-index <command> [options]
Available commands are:

    train           Run training with PyTorch Lightning
    lr-find         Run Lightning Learning Rate finder
    cuda-sum        Summarize what Torch sees in CUDA land
    infer           Run inference using PyTorch
    summarize       Summarize training/inference results
    sample-gtdb     Sample taxa from a tree
    make-fof        Run function make_fof from exabiome.gtdb.make_fof
    prepare-data    Aggregate sequence data GTDB using a file-of-files
    ncbi-path       Print path at NCBI FTP site to stdout
    ncbi-fetch      Retrieve sequence data from NCBI FTP site using rsync
```

## Sampling taxa to train with
This command will sample taxa from a GTDB tree.
```bash
deep-index sample-gtdb
```

## Downloading data from NCBI
This command will retrieve sequence files from NCBI. 
```bash
deep-index ncbi-fetch
```

## Converting Data
This command can be used to convert sequence data into an aggregated file with data prepared for training.
```bash
deep-index prepare-data
```

## Training neural networks
To train neural networks, we use PyTorch Lightning. This code can be executed with the following command.

```bash
deep-index train
```
This command will split up the input dataset into training, validation, and testing data. The seed used to do this
will be saved in the checkpoint, so subsequent use, such as for testing, will have the same split.

## Doing inference with neural networks

This command will compute network outputs for each sample from all sub-datasets. To run, you must provide
this command with the checkpoint produced during training. When it is finished, it will save the results in
the same directory that the input checkpoint file was saved.

```bash
deep-index infer
```

## Network output summary

After computing model outputs, the outputs can be summarized using the follwoing command. This will produce a
PNG figure with a scatter plot of a 2D UMAP embedding if the model outputs. It will also build a simple 
random forest classifier and plot a classification report 

```bash
deep-index summarize
```
