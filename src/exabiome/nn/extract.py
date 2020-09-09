from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def extract(argv=None):
    '''Exract loss plots from TensorBoard event file'''
    import os.path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('event_file', type=str, help='the TensorBoard event file', nargs='+')

    args = parser.parse_args(argv)

    for f in args.event_file:
        output = os.path.join(os.path.dirname(f), 'loss.png')
        plot(f, output)


def plot(event_file, output):

    ea = event_accumulator.EventAccumulator(event_file, #'events.out.tfevents.x.ip-x-x-x-x',
     size_guidance={ # see below regarding this argument
         event_accumulator.COMPRESSED_HISTOGRAMS: 500,
         event_accumulator.IMAGES: 4,
         event_accumulator.AUDIO: 4,
         event_accumulator.SCALARS: 0,
         event_accumulator.HISTOGRAMS: 1,
     })

    ea.Reload() # loads events from file

    # Your scalars will be different
    epoch_df = pd.DataFrame(ea.Scalars('epoch'))
    v_df = pd.DataFrame(ea.Scalars('val_loss'))
    t_df = pd.DataFrame(ea.Scalars('loss'))


    epoch = list()
    val_loss = list()
    loss = list()
    value = epoch_df['value']
    wall_time = epoch_df['wall_time']
    val_values = v_df['value']

    val_time = v_df['wall_time']
    loss_values = t_df['value']
    loss_time = t_df['wall_time']

    for ep in np.unique(value):
        ep_end = wall_time[value == ep].values[-1]
        ep_val = val_values[val_time <= ep_end].values[-1]
        ep_loss = loss_values[loss_time <= ep_end].values[-1]
        epoch.append(ep)
        loss.append(ep_loss)
        val_loss.append(ep_val)

    epoch = np.array(epoch)
    loss = np.array(loss)
    val_loss = np.array(val_loss)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epoch, loss, color='black', label='training loss')
    plt.plot(epoch, val_loss, color='red', label='validation loss')
    plt.title('Loss')
    plt.gca().set_yscale('log')
    plt.legend()

    print(f'{event_file}\n - Epoch {epoch[-1]} - training loss = {loss[-1]:0.3f}, validation loss = {val_loss[-1]:0.3f}')

    plt.subplot(1,2,2)
    loss_values = loss_values.values
    val_values = val_values.values
    loss_values = np.abs(loss_values[1:] - loss_values[:-1])
    val_values = np.abs(val_values[1:] - val_values[:-1])
    plt.plot(loss_values, color='black', label='training_loss')
    plt.plot(val_values, color='red', label='validation')
    plt.title('Delta')
    plt.gca().set_yscale('log')
    plt.legend()

    plt.savefig(output)
