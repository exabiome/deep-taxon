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
All commands can be accessed with the `deep-taxon` executable. Below is the `deep-taxon` usage statement, which
lists the available commands.

```bash
Usage: deep-taxon <command> [options]
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
deep-taxon sample-gtdb
```

## Downloading data from NCBI
This command will retrieve sequence files from NCBI. 
```bash
deep-taxon ncbi-fetch
```

## Converting Data
This command can be used to convert sequence data into an aggregated file with data prepared for training.
```bash
deep-taxon prepare-data
```

## Training neural networks
To train neural networks, we use PyTorch Lightning. This code can be executed with the following command.

```bash
deep-taxon train
```
This command will split up the input dataset into training, validation, and testing data. The seed used to do this
will be saved in the checkpoint, so subsequent use, such as for testing, will have the same split.

## Doing inference with neural networks

This command will compute network outputs for each sample from all sub-datasets. To run, you must provide
this command with the checkpoint produced during training. When it is finished, it will save the results in
the same directory that the input checkpoint file was saved.

```bash
deep-taxon infer
```

## Network output summary

After computing model outputs, the outputs can be summarized using the follwoing command. This will produce a
PNG figure with a scatter plot of a 2D UMAP embedding if the model outputs. It will also build a simple 
random forest classifier and plot a classification report 

```bash
deep-taxon summarize
```

# Example workflow

Before preparing an input file for training a network, you will need to download the necessary
input files from the [Genome Taxonomy Database](https://gtdb.ecogenomic.org/) (GTDB). 
Files can be downloaded [here](https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/). You
will need to download the metadata file (i.e. `*_metadata*`) and the tree file (i.e. `*.tree`)

### Step 1 - Sample the GTDB tree

Once you have a metadata file and a tree file, you can run `sample-gtdb` to generate a list of NCBI accessions.

```bash
$ deep-taxon sample-gtdb ar122_metadata_r89.tsv ar122_r89.tree > my_accessions.txt
```

### Step 2 - Download files from NCBI

Next, pass `my_accessions.txt` into `ncbi-fetch` to obtain sequence files for the accessions you
have chosen. If you already have files downloaded, you can skip this step. This command calls `rsync`,
so if you already have the files downloaded, it will not re-download them.

```bash 
$ deep-taxon ncbi-fetch -f my_accessions.txt ncbi_sequences
```

Note that you will need to use the `-f` flag to indication that first arguemnt is a file containing a 
list of accessions. 
The second argument is where sequence files get downloaded to. `ncbi-fetch` will
preserve the directory structure from the NCBI FTP site. Do not modify this, as the following command,
`prepare-data` will expect this directory structure.
If you are downloading many files and would like to speed things up, use `-p` to run
downloads in parallel.

### Step 3 - Converting to training input file

Now that sequence files are downloaded, sequence data can be converted into a input file for training.

```bash
$ deep-taxon prepare-data -V -G my_accessions.txt ncbi_sequences ar122_metadata_r89.tsv ar122_r89.tree my_input.h5
```

This will convert *genomic* sequence (i.e. `-G` flag) for the accessions you stored in `my_accessions.txt`. Data
will be read from the directory `ncbi_sequences`. 

## Getting non-representative genomes

The previous workflow will generate an input file for *representative* genomes. You may want to use non-representatives.
To do this, you can use the command `deep-taxon sample-nonrep`

```bash
$ deep-taxon sample-nonrep my_accessions.txt ar122_metadata_r89.tsv > nonrep_accessions.txt
```

This will print the accessions of non-representative genomes to the file `nonrep_accessions.txt`. You can also 
get the paths to the sequence files these for these strains by supplying a directory with the NCBI files. You can use
the flags `-G`, `-C`, or `-P` to get the genomes, gene coding sequences, or protein sequences, respectively. By default,
genome paths will be printed if you only provide the path to the NCBI Fasta directory. 

Once you have a list of accessions, you can run _Steps 2 and 3_ from above to finish building an input file for inference
on held-out genomes.

