import sys
import numpy as np
import torch
import pickle
import argparse
import os.path
import matplotlib.pyplot as plt


def plot(lossd, log=False):
    plt.plot(lossd['train_loss'], label='train')
    plt.plot(lossd['test_loss'], label='test')
    if log:
        plt.xscale('log')
        plt.yscale('log')
    plt.legend()

def get_loss(checkpoint_path):
    print('reading %s' % checkpoint_path, file=sys.stderr)
    try:
        checkpoint = torch.load(checkpoint_path)
    except Exception as e:
        print(f'unable to read {checkpoint_path} - {type(e)}: {e}')
        return
    ret = dict(
        train_loss = checkpoint['train_loss'],
        test_loss = checkpoint['test_loss']
    )
    return ret

def main(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-f', '--figure', type=str, default=None)

    args = parser.parse_args(args=args)

    dat = {f : L for f in args.files if (L := get_loss(f)) is not None}

    out = open(args.output, 'wb') if args.output is not None else sys.stdout

    pickle.dump(dat, out)

    if args.figure:
        s = len(os.path.commonprefix(list(dat.keys())))
        ncols = int(np.ceil(np.sqrt(len(dat))))
        nrows = int(np.ceil(len(dat)/ncols))
        plt.figure(figsize=(ncols*5.5, nrows*7.5/2))
        for i, k in enumerate(dat):
            plt.subplot(nrows, ncols, i+1)
            plot(dat[k], log=False)
            plt.title(k[s:-4])
        plt.savefig(args.figure)


if __name__ == '__main__':
    main()


