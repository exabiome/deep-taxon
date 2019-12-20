from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import exabiome.sequence
from exabiome.nn import get_loader
from exabiome.nn.model import SPP_CNN

path = "../../exabiome.git/untracked/test_fof/test.h5"
#loader = get_loader(path, batch_size=64, shuffle=True)
loader = get_loader(path, batch_size=2, shuffle=True)

net = SPP_CNN(26, 100)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

n_epochs = 1

before = datetime.now()
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        seqs, emb, orig_lens = data

        # zero the parameter gradients
        optimizer.zero_grad()

        ## Modify here to adjust learning rate
        ## Create new Optimizer object?

        # forward + backward + optimize
        outputs = net(seqs, orig_len=orig_lens)


        loss = criterion(outputs, emb)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 40 == 39:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.6e' %
                  (epoch + 1, i + 1, running_loss/40))
            running_loss = 0.0
after = datetime.now()
print('Finished Training. Took', (after-before).total_seconds(), 'seconds')

