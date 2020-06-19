# exabiome
This package contains executable models for each of the main steps in the neural network training process. There exist
an executable for each of the following steps:

1. Converting ddata
2. Training networks
3. Network inference
4. Summarizing network outputs

## Converting Data

```bash
python -m exabiome.tools.prepare_data
```

## Training neural networks
To train neural networks, we use PyTorch Lightning. This code can be executed with the following command.

```bash
python -m exabiome.nn.train
```
This command will split up the input dataset into training, validation, and testing data. The seed used to do this
will be saved in the checkpoint, so subsequent use, such as for testing, will have the same split.

Running with DistributedDataParallel i.e. multiple GPUs. using latest version of PyTorch Lightning (as of June 19, 2020) 
does not work with executable modules. To get around this, use the following form of calling the training code:

```bash
python bin/train.py
```

## Doing inference with neural networks

This command will compute network outputs for each sample from all sub-datasets. To run, you must provide
this command with the checkpoint produced during training. When it is finished, it will save the results in
the same directory that the input checkpoint file was saved.

```bash
python -m exabiome.nn.infer
```

## Network output summary

After computing model outputs, the outputs can be summarized using the follwoing command. This will produce a
PNG figure with a scatter plot of a 2D UMAP embedding if the model outputs. It will also build a simple 
random forest classifier and plot a classification report 

```bash
python -m exabiome.nn.summarize
```
