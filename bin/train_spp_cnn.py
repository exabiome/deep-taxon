from exabiome.nn.model import SPP_CNN
from exabiome.nn.train import parse_args, run_serial
import torch.optim as optim

args = parse_args("Train CNN with Spatial Pyramidal Pooling")

input_nc = 4
if args['protein']:
    input_nc = 26

model = SPP_CNN(input_nc, 100, kernel_size=13)
optimizer = optim.Adam(model.parameters(), lr=0.001)

run_serial(model=model, optimizer=optimizer, **args)
