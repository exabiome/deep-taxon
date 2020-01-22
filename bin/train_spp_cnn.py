from exabiome.nn.model import SPP_CNN
from exabiome.nn.train import parse_args, run_serial
import torch.optim as optim

args = parse_args("Train CNN with Spatial Pyramidal Pooling",
                  [['-E', '--emb_nc'], dict(type=int, default=0, help='the number of embedding channels. default is no embedding')])

input_nc = 4
if args['protein']:
    input_nc = 26

model = SPP_CNN(input_nc, 64, kernel_size=7, emb_nc=args['emb_nc'])
optimizer = optim.Adam(model.parameters(), lr=args['lr'])

if args['emb_nc'] > 0:
    args['ohe'] = False

run_serial(model=model, optimizer=optimizer, **args)
