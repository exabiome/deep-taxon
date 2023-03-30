import argparse
import sys
import glob
import os.path
import os
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
import scipy.stats as stats

from ..utils import parse_logger, ccm


def get_color_markers(n):
    all_colors = sns.color_palette('tab20b')[::4] + sns.color_palette('tab20c')[::4] +\
    sns.color_palette('tab20b')[1::4] + sns.color_palette('tab20c')[1::4] +\
    sns.color_palette('tab20b')[2::4] + sns.color_palette('tab20c')[2::4] +\
    sns.color_palette('tab20b')[3::4] + sns.color_palette('tab20c')[3::4]
    ret = []
    c = n
    while c > 0:
        ret.extend(all_colors[0:min(c, len(all_colors))])
        c -= len(all_colors)
    return ret


def read_outputs(path):
    ret = dict()
    with h5py.File(path, 'r') as f:
        if 'viz_emb' in f:
            ret['viz_emb'] = f['viz_emb'][:]
        ret['labels'] = f['labels'][:]
        if 'train' in f:
            ret['train_mask'] = f['train'][:]
        if 'test' in f:
            ret['test_mask'] = f['test'][:]
        ret['outputs'] = f['outputs'][:]
        if 'validate' in f:
            ret['validate_mask'] = f['validate'][:]
        ret['orig_lens'] = f['orig_lens'][:]
        if 'seq_ids' in f:
            ret['seq_ids'] = f['seq_ids'][:]
    return ret


def plot_results(path, tvt=True, pred=True, fig_height=7, logger=None, name=None):
    plot_count = 1
    if logger is None:
        logger = logging.getLogger()

    # read data
    if isinstance(path, str):
        path = read_outputs(path)

    n_plots = 1
    if pred is not False:
        n_plots += 1

    if tvt:
        if all(k in path for k in ('train', 'validate', 'test')):
            n_plots += 1
        else:
            tvt = False

    labels = path['labels']
    outputs = path['outputs']

    if 'viz_emb' in path:
        logger.info('found viz_emb')
        viz_emb = path['viz_emb']
    else:
        logger.info('calculating UMAP embeddings for visualization')
        from umap import UMAP
        umap = UMAP(n_components=2)
        viz_emb = umap.fit_transform(outputs)

    color_labels = getattr(pred, 'classes_', None)
    if color_labels is None:
        color_labels = labels
    class_pal = get_color_markers(len(np.unique(color_labels)))
    colors = np.array([class_pal[i] for i in color_labels])

    # set up figure
    fig_height = 7
    plt.figure(figsize=(n_plots*fig_height, fig_height))

    logger.info('plotting embeddings with species labels')
    # plot embeddings
    ax = plt.subplot(1, n_plots, plot_count)
    plot_seq_emb(viz_emb, labels, ax, pal=class_pal)
    if name is not None:
        plt.title(name)
    plot_count += 1

    # plot train/validation/testing data
    train_mask = None
    test_mask = None
    validate_mask = None
    if tvt:
        logger.info('plotting embeddings train/validation/test labels')
        train_mask = path['train_mask']
        test_mask = path['test_mask']
        validate_mask = path['validate_mask']
        pal = ['gray', 'red', 'yellow']
        plt.subplot(1, n_plots, plot_count)
        dsubs = ['train', 'validation', 'test'] # data subsets
        dsub_handles = list()
        for (mask, dsub, col) in zip([train_mask, validate_mask, test_mask], dsubs, pal):
            plt.scatter(viz_emb[mask, 0], viz_emb[mask, 1], s=0.1, c=[col], label=dsub)
            dsub_handles.append(Circle(0, 0, color=col))
        plt.legend(dsub_handles, dsubs)
        plot_count += 1

    # run some predictions and plot report
    if pred is not False:
        if pred is None or pred is True:
            logger.info('No classifier given, using RandomForestClassifier(n_estimators=30)')
            pred = RandomForestClassifier(n_estimators=30)
        elif not (hasattr(pred, 'fit') and hasattr(pred, 'predict')):
            raise ValueError("argument 'pred' must be a classifier with an SKLearn interface")

        X_test = outputs
        y_test = labels
        if not hasattr(pred, 'classes_'):
            train_mask = path['train_mask']
            test_mask = path['test_mask']
            X_train = outputs[train_mask]
            y_train = labels[train_mask]
            logger.info(f'training classifier {pred}')
            pred.fit(X_train, y_train)
            X_test = outputs[test_mask]
            y_test = labels[test_mask]
            logger.info(f'getting predictions')
        y_pred = pred.predict(X_test)

        logger.info(f'plotting classification report')
        # plot classification report
        ax = plt.subplot(1, n_plots, plot_count)
        plot_clf_report(y_test, y_pred, ax=ax, pal=class_pal)

    plt.tight_layout()
    return pred


def aggregated_chunk_analysis(path, clf, fig_height=7):
    # read data
    if isinstance(path, str):
        path = read_outputs(path)

    labels = path['labels']
    outputs = path['outputs']
    seq_ids = path['seq_ids']
    olens = path['orig_lens']

    viz_emb = None
    if 'viz_emb' in path:
        viz_emb = path['viz_emb']
    else:
        viz_emb = UMAP(n_components=2).fit_transform(X)

    uniq_seqs = np.unique(seq_ids)
    X_mean = np.zeros((uniq_seqs.shape[0], outputs.shape[1]))
    X_median = np.zeros((uniq_seqs.shape[0], outputs.shape[1]))
    y = np.zeros(uniq_seqs.shape[0], dtype=int)
    seq_len = np.zeros(uniq_seqs.shape[0], dtype=int)
    seq_viz = np.zeros((uniq_seqs.shape[0], 2))

    for seq_i, seq in enumerate(uniq_seqs):
        seq_mask = seq_ids == seq
        uniq_labels = labels[seq_mask]
        if not np.all(uniq_labels == uniq_labels[0]):
            raise ValueError(f'Found more than one label for sequence {seq}')
        y[seq_i] = uniq_labels[0]
        X_mean[seq_i] = outputs[seq_mask].mean(axis=0)
        X_median[seq_i] = np.median(outputs[seq_mask], axis=0)
        seq_viz[seq_i] = viz_emb[seq_mask].mean(axis=0)
        seq_len[seq_i] = olens[seq_mask].sum()

    seq_len = np.log10(seq_len)

    color_labels = getattr(clf, 'classes_', None)
    if color_labels is None:
        color_labels = labels
    class_pal = get_color_markers(len(np.unique(color_labels)))

    fig, axes = plt.subplots(nrows=3, ncols=3, sharey='row', figsize=(21, 21))

    # classifier from MEAN of outputs
    output_mean_preds = clf.predict(X_mean)
    make_plots(y, output_mean_preds, axes[:,0], class_pal, seq_len, 'Mean classification', seq_viz)

    # classifier from MEDIAN of outputs
    output_median_preds = clf.predict(X_median)
    make_plots(y, output_median_preds, axes[:,1], class_pal, seq_len, 'Median classification', seq_viz)

    # classifier from voting with chunk predictions
    all_preds = clf.predict(outputs)
    vote_preds = np.zeros_like(output_mean_preds)
    for seq_i, seq in enumerate(uniq_seqs):
        seq_mask = seq_ids == seq
        vote_preds[seq_i] = stats.mode(all_preds[seq_mask])[0][0]

    make_plots(y, vote_preds, axes[:,2], class_pal, seq_len, 'Vote classification', seq_viz)

    plt.tight_layout()


def make_plots(true, pred, axes, pal, seq_len, title=None, seq_viz=None):
    if seq_viz is not None:
        plot_seq_emb(seq_viz, pred, axes[0], pal=pal)
        axes[0].set_title(title)
        axes[0].set_xlabel('Mean of first UMAP dimension')
        axes[0].set_ylabel('Mean of second UMAP dimension')
        axes = axes[1:]
    plot_clf_report(true, pred, axes[0], pal=pal)
    plot_acc_of_len(true, pred, seq_len, axes[1])


def plot_acc_of_len(y_true, y_pred, seq_len, ax):
    nbins = 100
    _, edges = np.histogram(seq_len, bins=nbins)
    avg_len = np.zeros(nbins)
    acc = np.zeros(nbins)
    correct = y_true == y_pred
    filt = list()
    for i in range(nbins):
        mask = np.logical_and(seq_len >= edges[i], seq_len < edges[i+1])
        if mask.any():
            acc[i] = correct[mask].mean()
            avg_len[i] = seq_len[mask].mean()
            filt.append(i)
    avg_len = avg_len[filt]
    acc = acc[filt]
    ax.scatter(avg_len, acc)


def plot_seq_emb(X, labels, ax, pal=None):
    # plot embeddings
    uniq_labels = np.unique(labels)
    class_handles = list()
    if pal is None:
        pal = sns.color_palette('tab20', len(uniq_labels))
    for cl in uniq_labels:
        col = pal[cl]
        mask = labels == cl
        ax.scatter(X[mask,0], X[mask,1], s=0.5, c=[col], label=cl)
        class_handles.append(Circle(0, 0, color=col))


def plot_clf_report(y_true, y_pred, ax, pal=None):
    uniq_labels = np.unique(y_true)
    if pal is None:
        pal = sns.color_palette('tab20', len(uniq_labels))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report)
    subdf = df[[str(l) for l in uniq_labels]].iloc[:-1]
    subplot = subdf.plot.barh(ax=ax, color=pal, legend=False)
    ax.text(np.mean(subplot.get_xlim())*1.4, np.max(subplot.get_ylim())*0.95, 'accuracy: %0.6f' % report['accuracy'])


def plot_pr_curve(y_true, X, ax, pal=None, labels=None):
    for i in np.unique(y_true):
        precision, recall, thresh = precision_recall_curve(y_true, X[:, i], pos_label=i)
        ax.plot(precision, recall, color=pal[i])

def make_clf_plots(true, X, axes, pal, seq_len):
    pred = np.argmax(X, axis=1)
    plot_clf_report(true, pred, axes[0], pal=pal)
    plot_acc_of_len(true, pred, seq_len, axes[1])
    plot_pr_curve(true, X, axes[2], pal=pal)


def classification_aggregate(path, verbose=False):
    """Aggregate outputs from a classifier

    outputs should be a softmax
    """
    labels = path['labels']
    outputs = path['outputs']
    seq_ids = path['seq_ids']
    olens = path['orig_lens']

    viz_emb = None
    if 'viz_emb' in path:
        viz_emb = path['viz_emb']
        seq_viz = np.zeros((uniq_seqs.shape[0], 2))

    uniq_seqs = np.unique(seq_ids)
    X_mean = np.zeros((uniq_seqs.shape[0], outputs.shape[1]))
    X_median = np.zeros((uniq_seqs.shape[0], outputs.shape[1]))
    y = np.zeros(uniq_seqs.shape[0], dtype=int)
    seq_len = np.zeros(uniq_seqs.shape[0], dtype=int)

    vote_preds = np.zeros_like(X_mean)

    all_preds = np.argmax(outputs, axis=1)

    it = enumerate(uniq_seqs)
    if verbose:
        from tqdm import tqdm
        it = tqdm(it, total=uniq_seqs.shape[0])
    for seq_i, seq in it:
        seq_mask = seq_ids == seq
        uniq_labels = labels[seq_mask]
        if not np.all(uniq_labels == uniq_labels[0]):
            raise ValueError(f'Found more than one label for sequence {seq}')
        y[seq_i] = uniq_labels[0]
        X_mean[seq_i] = outputs[seq_mask].mean(axis=0)
        X_median[seq_i] = np.median(outputs[seq_mask], axis=0)
        if viz_emb is not None:
            seq_viz[seq_i] = viz_emb[seq_mask].mean(axis=0)
        seq_len[seq_i] = olens[seq_mask].sum()
        uniq, counts = np.unique(all_preds[seq_mask], return_counts=True)
        vote_preds[seq_i][uniq] = counts/counts.sum()

    return y, X_mean, X_median, vote_preds, seq_len


def classifier_summarize(argv=None):
    '''Summarize training/inference results'''
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='the HDF5 file with network outputs or a directory containing a single outputs file')
    parser.add_argument('-o', '--outdir', type=str, default=None, help='the output directory for figures')

    args = parser.parse_args(args=argv)
    if os.path.isdir(args.input):
        outputs = list(glob.glob(f'{args.input}/outputs.h5'))
        if len(outputs) != 1:
            print(f'More than one outputs file in {args.input}, please specify the exact file')
            sys.exit(1)
        args.input = outputs[0]

    outdir = os.path.dirname(args.input)
    if args.outdir is not None:
        outdir = args.outdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    fig_path = os.path.join(outdir, 'summary.png')
    agg_res_path = os.path.join(outdir, 'aggregated_results.h5')

    logger = parse_logger('')
    logger.info(f'saving figure to {fig_path}')
    logger.info(f'saving aggregated outputs to {agg_res_path}')

    logger.info('reading outputs')
    outputs = read_outputs(args.input)

    logger.info('aggregating outputs across sequences')
    y_true_seq, mean, median, vote, seq_len = classification_aggregate(outputs, verbose=True)
    seq_len = np.log10(seq_len)

    n_classes = outputs['outputs'].shape[1]
    class_pal = get_color_markers(n_classes)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    make_clf_plots(y_true_seq, mean, axes[:, 0], class_pal, seq_len)
    axes[0, 0].set_title('Mean')
    make_clf_plots(y_true_seq, median, axes[:, 1], class_pal, seq_len)
    axes[0, 1].set_title('Median')
    make_clf_plots(y_true_seq, vote, axes[:, 2], class_pal, seq_len)
    axes[0, 2].set_title('Vote')

    with h5py.File(agg_res_path, 'w') as f:
        f.create_dataset('labels', data=y_true_seq)
        f.create_dataset('mean', data=mean)
        f.create_dataset('median', data=median)
        f.create_dataset('vote', data=vote)

    plt.savefig(fig_path, dpi=100)


def summarize(argv=None):
    '''Summarize training/inference results'''
    import argparse
    import pickle
    from ..utils import parse_logger
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='the HDF5 file with network outputs or a directory containing a single outputs file')
    parser.add_argument('classifier', type=str, nargs='?', default=None, help='the classifier to use for predictions')
    parser.add_argument('-A', '--aggregate_chunks', action='store_true', default=False,
                        help='aggregate chunks within sequences and perform analysis')
    parser.add_argument('-o', '--outdir', type=str, default=None, help='the output directory for figures')

    args = parser.parse_args(args=argv)
    if os.path.isdir(args.input):
        outputs = list(glob.glob(f'{args.input}/outputs.h5'))
        if len(outputs) != 1:
            print(f'More than one outputs file in {args.input}, please specify the exact file')
            sys.exit(1)
        args.input = outputs[0]

    outdir = os.path.dirname(args.input)
    if args.outdir is not None:
        outdir = args.outdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    fig_path = os.path.join(outdir, 'summary.png')
    logger = parse_logger('')

    plt.figure(figsize=(21, 7))
    pretrained = False
    if args.classifier is not None:
        with open(args.classifier, 'rb') as f:
            pred = pickle.load(f)
        pretrained = True
    else:
        pred = RandomForestClassifier(n_estimators=30)
    outputs = read_outputs(args.input)
    pred = plot_results(outputs, pred=pred, name='/'.join(args.input.split('/',)[-2:]), logger=logger)
    logger.info(f'saving figure to {fig_path}')
    plt.savefig(fig_path, dpi=100)
    if not pretrained:
        clf_path = os.path.join(outdir, 'summary.rf.pkl')
        logger.info(f'saving classifier to {clf_path}')
        with open(clf_path, 'wb') as f:
            pickle.dump(pred, f)

    if args.aggregate_chunks:
        logger.info(f'running summary by aggregating chunks within sequences')
        aggregated_chunk_analysis(outputs, pred)
        agg_fig_path = os.path.join(outdir, 'summary.aggregated.png')
        logger.info(f'saving figure to {agg_fig_path}')
        plt.savefig(agg_fig_path, dpi=100)


def get_profile_data(argv=None):
    """Extract profiling data from PL log"""

    import argparse
    import os
    import re

    parser = argparse.ArgumentParser()

    parser.add_argument('log_files', nargs='+', help='the log files to parse')

    args = parser.parse_args(argv)

    N_PROFILE_ROWS = 32
    re_delim = re.compile('\s*\|[_\s]*')
    time_col = -1
    keys =  ['model_forward',
             'model_backward',
             'on_batch_end',
             'optimizer_step',
             'get_train_batch']

    gpu_re = re.compile('_g(\d+)_')
    batch_re = re.compile('_b(\d+)_')
    nodes_re = re.compile('n(\d+)_')

    all_data = list()
    print(args.log_files, file=sys.stderr)
    for log_file in args.log_files:
        found_report = False
        n_nodes = nodes_re.search(log_file).groups(0)[0].strip()
        n_gpu = gpu_re.search(log_file).groups(0)[0].strip()
        batch_size = batch_re.search(log_file).groups(0)[0].strip()
        with open(log_file, 'r', encoding='ISO-8859-1') as f:
            rank = 0
            for line in f:
                if line.startswith('Action'):
                    data = {
                        'n_nodes': n_nodes,
                        'n_gpu': n_gpu,
                        'batch_size': batch_size,
                        'rank': rank,
                    }
                    n_rows_read = 0
                    found_report = True
                    columns = re_delim.split(line)
                    for i, c in enumerate(columns):
                        if 'Total' in c:
                            time_col = i
                elif found_report:
                    if n_rows_read == N_PROFILE_ROWS: # done reading table
                        found_report = False
                        all_data.append(data)
                        data = dict()
                        rank += 1
                        n_rows_read = 0
                        continue
                    elif line.startswith('-----'):
                        continue
                    if line.startswith('Total'):
                        ar = re_delim.split(line)
                        data['total'] = ar[time_col].strip()
                    else:
                        ar = re_delim.split(line)
                        data[ar[0]] = ar[time_col].strip()
                        n_rows_read += 1
    pd.DataFrame(data=all_data).to_csv(sys.stdout)


def plot_loss(argv=None):
    import argparse
    import os

    import matplotlib.pyplot as plt
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, nargs='+', help='the metrics.csv file')
    parser.add_argument('-o', '--outdir', type=str, help='the output directory', default=None)
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing plots if they exist', default=False)
    parser.add_argument('-L', '--labels', type=str, help='a comma-separated string with labels for each CSV', default=None)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-V', '--validation', action='store_true', default=False, help='only plot validation')
    grp.add_argument('-R', '--training', action='store_true', default=False, help='only plot training')
    #parser.add_argument('cols', nargs='*', type=str, help='the metrics.csv file')
    args = parser.parse_args(argv)

    if not (args.validation or args.training):
        args.validation = True
        args.training = True

    def plot(ax, x, y, **kwargs):
        mask = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
        x = x[mask]
        y = y[mask]
        ax.plot(x, y, **kwargs)

    outdir = args.outdir

    if outdir is None:
        if len(args.csv) == 1:
            outdir = os.path.dirname(args.csv[0])
        else:
            outdir = '.'
    loss_path = os.path.join(outdir, 'loss.png')
    acc_path = os.path.join(outdir, 'accuracy.png')
    if (os.path.exists(loss_path) or os.path.exists(acc_path)) and not args.force:
        badpath = loss_path
        if os.path.exists(acc_path):
            badpath = acc_path
        print(f'Output file {badpath} exists - exiting. Use -f or remove file and rerun', file=sys.stderr)
        sys.exit(1)


    if args.labels is not None:
        labels = args.labels.split(',')
        if len(labels) != len(args.csv):
            print(f'found {len(labels)} labels, but found {len(args.csv)} CSV files', file=sys.stderr)
            sys.exit(1)
    else:
        labels = [f'csv-{d}' for d in range(len(args.csv))]


    loss_fig, loss_ax = plt.subplots(1, figsize=(7, 5))
    epoch_max = 0
    loss_max = 0
    epoch_ratio = None
    epoch_x = None


    acc_fig, acc_ax = None, None
    acc_max = 0

    colors = np.array(sns.color_palette('tab20', len(args.csv) * 2).as_hex()).reshape((len(args.csv),2)).tolist()
    for col, csv, label in zip(colors, args.csv, labels):
        df = pd.read_csv(csv, header=0)

        x = df.step.values
        tr_loss = df.training_loss.values
        epoch = df.epoch.values
        if epoch_ratio is None or epoch.max() > epoch_max:
            epoch_max = epoch.max()
            epoch_ratio = epoch
            epoch_x = x

        if args.training:
            plot(loss_ax, x, tr_loss, color=col[1], label=f'{label} (tr)')
        if 'validation_loss' in df and args.validation:
            va_loss = df.validation_loss.values
            plot(loss_ax, x, va_loss, color=col[0], ls='--', label=f'{label} (val)')
            loss_max = max(loss_max, np.nanmax(tr_loss))

        columns = df.columns
        if 'training_acc' in df.columns and args.training:
            if acc_fig is None:
                acc_fig, acc_ax = plt.subplots(1, figsize=(7, 5))

            tr_acc = df.training_acc.values
            plot(acc_ax, x, tr_acc, color=col[1], label=f'{label} (tr)')
        if 'validation_acc' in df.columns and args.validation:
            if acc_fig is None:
                acc_fig, acc_ax = plt.subplots(1, figsize=(7, 5))
            va_acc = df.validation_acc.values
            plot(acc_ax, x, va_acc, color=col[0], ls='--', label=f'{label} (val)')

    def add_epoch(ax, maxval):
        # calculate epoch values to fit on Axes
        epoch_ax = ax.twinx()
        ep = epoch_ratio
        plot(epoch_ax, epoch_x, ep, color='gray', ls=':', alpha=0.5, label='epoch')

        # set the limits
        ymin, ymax = epoch_ax.get_ylim()
        if ymin < 0:
            ymin = -0.05 * ep.max()
        if ymax < 1:
            ymax = 1.05 * ep.max()
        epoch_ax.set_ylim(ymin, ymax)

        # clean up the ticks
        if ep.max() < 8:
            ticks = np.arange(ep.max()).astype(int)
        else:
            ticks = np.arange(0, ep.max(), ep.max()//8).astype(int)
        epoch_ax.set_yticks(ticks)
        epoch_ax.set_yticklabels([str(_) for _ in ticks])

        # Add legend to epoch Axes so legend gets drawn over epoch line
        handles, labels = ax.get_legend_handles_labels()
        epoch_ax.legend(handles, labels, bbox_to_anchor=(1.1,1), loc="upper left")
        epoch_ax.set_ylabel('Epoch')

    loss_ax.set_ylabel('Loss')
    add_epoch(loss_ax, loss_max)

    loss_fig.tight_layout()
    print(f'saving loss figure to {loss_path}')
    loss_fig.savefig(loss_path, dpi=100)


    if acc_ax is not None:
        acc_ax.set_ylabel('Accuracy')
        add_epoch(acc_ax, acc_max)

        acc_fig.tight_layout()
        print(f'saving accuracy figure to {acc_path}')
        acc_fig.savefig(acc_path, dpi=100)


def aggregate_chunks(argv=None):
    import torch
    import torch.nn.functional as F
    from contextlib import contextmanager

    from deep_taxon.utils import parse_logger

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="the inference file with individual sample (i.e. chunks) outputs")
    parser.add_argument("output", type=str, help="the file to save aggregated outputs to")
    parser.add_argument('-F', '--features', action='store_true', help='outputs are features i.e. do not softmax and compute predictions', default=False)
    parser.add_argument('--fwd_only', action='store_true', help='only forward strand was used', default=False)
    parser.add_argument('-w', '--weighted', action='store_true', help='weight chunks by original length', default=False)

    args = parser.parse_args(argv)

    args.prob = not args.features

    logger = parse_logger('')

    rank = 0
    size = 1
    f_kwargs = dict()
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        # load this after so we can get usage
        # statement without having to loading MPI
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if size > 1:
            f_kwargs['driver'] = 'mpio'
            f_kwargs['comm'] = comm
    else:
        logger.info('OMPI_COMM_WORLD_RANK not set in environment -- not using MPI')


    if rank == 0:
        logger.info(f'{args}')

    f = h5py.File(args.input, 'r', **f_kwargs)

    logger.info(f'rank {rank} - loading network outputs')
    chunk_seq_ids = f['seq_ids'][:]
    chunk_outputs = f['outputs']
    chunk_labels = f['labels'][:]
    chunk_lens = f['orig_lens'][:]

    uniq_seqs = np.unique(chunk_seq_ids)
    total_seqs = len(uniq_seqs)

    if size > 1:
        q, r = divmod(total_seqs, size)
        if rank < r:
            q += 1
            b = rank * q
            seq_idx = np.arange(b, b + q)
        else:
            offset = (q + 1) * r
            b = (rank - r) * q + offset
            seq_idx = np.arange(b, b + q)

        N = len(seq_idx)
    else:
        seq_idx = np.s_[:]
        N = total_seqs

    if args.prob:
        logger.info('data are probabilities. computing and saving predictions')
        seq_preds = np.zeros(N, dtype=int)
        seq_votes = np.zeros((N, chunk_outputs.shape[1]), dtype=float)

    seq_labels = np.zeros(N, dtype=int)
    seq_lens = np.zeros(N, dtype=int)
    seq_outputs = np.zeros((N, chunk_outputs.shape[1]), dtype=float)
    seq_outputs_var = np.zeros((N, chunk_outputs.shape[1]), dtype=float)

    uniq_seqs = uniq_seqs[seq_idx]

    gpu = torch.device('cuda:%d' % (rank % torch.cuda.device_count()))

    weights = None
    if args.weighted:
        weights = torch.tensor(chunk_lens, device=gpu)

    logger.info(f'rank {rank} - aggregating chunk outputs by sequence')
    for i, seq in enumerate(uniq_seqs):
        mask = chunk_seq_ids == seq
        idx = np.where(mask)[0]
        sub_outputs = chunk_outputs[idx]

        tmp_outputs = torch.tensor(sub_outputs, device=gpu)
        if args.prob:
            tmp_outputs = F.softmax(tmp_outputs, dim=1)
            shape = tmp_outputs.shape
            seq_votes[i] = (tmp_outputs.argmax(dim=1).bincount(minlength=shape[1])/shape[0]).cpu()
            seq_outputs_var[i] = tmp_outputs.var(dim=0, unbiased=False).cpu()
            if args.weighted:
                w = weights[mask]
                seq_outputs[i] = ((tmp_outputs.T*w).T.sum(dim=0) / (w.sum())).cpu()
            else:
                seq_outputs[i] = tmp_outputs.mean(dim=0).cpu()
            seq_preds[i] = seq_outputs[i].argmax()
        else:
            seq_outputs_var[i] = tmp_outputs.var(dim=0, unbiased=False).cpu()
            seq_outputs[i] = tmp_outputs.mean(dim=0).cpu()

        sub_labels = chunk_labels[mask]
        seq_labels[i] = sub_labels[0]
        seq_lens[i] = chunk_lens[mask].sum()


        # delete this for good measure so we don't leave data on the GPU
        del tmp_outputs

    if args.fwd_only:
        seq_lens = seq_lens // 2

    if size > 1:
        comm.Barrier()

    f.close()

    logger.info(f'saving results to {args.output}')
    out = h5py.File(args.output, 'w', **f_kwargs)

    N = total_seqs
    seq_outputs_dset = out.create_dataset('outputs', shape=(N, seq_outputs.shape[1]), dtype=float)
    seq_outputs_var_dset = out.create_dataset('outputs_var', shape=(N, seq_outputs_var.shape[1]), dtype=float)
    if args.prob:
        seq_preds_dset = out.create_dataset('preds', shape=(N,), dtype=int)
        seq_votes_dset = out.create_dataset('votes', shape=(N, seq_outputs.shape[1]), dtype=float)



    seq_labels_dset = out.create_dataset('labels', shape=(N,), dtype=int)
    seq_lens_dset = out.create_dataset('lengths', shape=(N,), dtype=int)
    seq_ids_dset = out.create_dataset('seq_ids', shape=(N,), dtype=int)

    logger.info(f'rank {rank} - writing outputs')
    with ccm(size > 1, seq_outputs_dset.collective):
        seq_outputs_dset[seq_idx] = seq_outputs

    logger.info(f'rank {rank} - writing outputs variance')
    with ccm(size > 1, seq_outputs_var_dset.collective):
        seq_outputs_var_dset[seq_idx] = seq_outputs_var

    logger.info(f'rank {rank} - writing labels')
    with ccm(size > 1, seq_labels_dset.collective):
        seq_labels_dset[seq_idx] = seq_labels

    if args.prob:
        logger.info(f'rank {rank} - writing predictions')
        with ccm(size > 1, seq_preds_dset.collective):
            seq_preds_dset[seq_idx] = seq_preds
            seq_votes_dset[seq_idx] = seq_votes

    logger.info(f'rank {rank} - writing sequence lengths')
    with ccm(size > 1, seq_lens_dset.collective):
        seq_lens_dset[seq_idx] = seq_lens

    logger.info(f'rank {rank} - writing sequence ids')
    with ccm(size > 1, seq_ids_dset.collective):
        seq_ids_dset[seq_idx] = uniq_seqs

    if size > 1:
        comm.Barrier()

    logger.info(f'rank {rank} - done')
    out.close()


def taxonomic_accuracy(argv=None):
    #import ..sequence as seq
    from ..sequence import DeepIndexFile
    from ..utils import get_logger
    from hdmf.common import get_hdf5io
    import h5py
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder


    levels = DeepIndexFile.taxonomic_levels

    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=str, help='the summarized sequence NN outputs')
    parser.add_argument("input", type=str, help='the training input data')
    parser.add_argument("output", type=str, help="the path to save resutls to")
    parser.add_argument("-l", "--level", type=str, choices=levels, help='the taxonomic level')

    args = parser.parse_args(argv)

    logger = get_logger()

    logger.info(f'reading {args.input}')
    io = get_hdf5io(args.input, 'r')
    difile = io.read()

    with h5py.File(args.summary, 'r') as f:

        logger.info(f'loading summary results from {args.summary}')

        n_classes = None
        if 'outputs' in f:
            n_classes = f['outputs'].shape[1]
        elif 'n_classes' in f['labels'].attrs:
            n_classes = f['labels'].attrs['n_classes']

        level = None
        classes = None
        if args.level is None:
            if n_classes is None:
                print(f"Could not find number of classes in {args.summary}. Without this, I cannot guess what the taxonomic level is")
                exit(1)
            for lvl in levels[:-1]:
                n_classes_lvl = difile.taxa_table[lvl].elements.data.shape[0]
                if n_classes == n_classes_lvl:
                    classes = difile.taxa_table[lvl].elements.data
                    level = lvl
            if level is None:
                n_classes_lvl = difile.taxa_table['species'].data.shape[0]
                if n_classes == n_classes_lvl:
                    level = 'species'
                    classes = difile.taxa_table['species'].data[:]
                else:
                    print("Cannot determine which level to use. Please specify with --level option", file=sys.stderr)
                    exit(1)
        else:
            level = args.level

        logger.info(f'computing accuracy for {level}{" and higher" if level else ""}')

        seq_preds = f['preds'][:].astype(int)
        seq_labels = f['labels'][:].astype(int)
        seq_lens = f['lengths'][:].astype(int)

    mask = seq_labels != -1
    seq_preds = seq_preds[mask]
    seq_labels = seq_labels[mask]
    seq_lens = seq_lens[mask]

    logger.info(f'Keeping {mask.sum()} of {mask.shape[0]} ({mask.mean()*100:.1f}%) sequences after discarding uninitialized sequences')

    ## I used this code to double check that genus elements were correct
    # seq_ids = f['seq_ids'][:]
    # genome_ids = difile.seq_table['genome'].data[:][seq_ids]
    # taxon_ids = difile.genome_table['taxon_id'].data[:][genome_ids]
    # classes = difile.taxa_table['genus'].elements.data[:]

    logger.info('loading taxonomy table')
    # do this because h5py.Datasets cannot point-index with non-unique indices
    for col in difile.taxa_table.columns:
        col.transform(lambda x: x[:])

    to_drop = ['taxon_id']
    for lvl in levels[::-1]:
        if lvl == level:
            break
        to_drop.append(lvl)

    # orient table to index it by the taxonomic level and remove columns we cannot get predictions for
    taxdf = difile.taxa_table.to_dataframe()

    n_orig_classes = {col: np.unique(taxdf[col]).shape[0] for col in taxdf}

    taxdf = taxdf.drop(to_drop, axis=1).\
                  set_index(level).\
                  groupby(level).\
                  nth(0).\
                  filter(classes, axis=0)

    logger.info('encoding taxonomy for quicker comparisons')
    # encode into integers for faster comparisons
    encoders = dict()
    new_dat = dict()
    for col in taxdf.columns:
        enc = LabelEncoder().fit(taxdf[col])
        encoders[col] = enc
        new_dat[col] = enc.transform(taxdf[col])
    enc_df = pd.DataFrame(data=new_dat, index=taxdf.index)

    # a helper function to transform results into a DataFrame
    def get_results(true, pred, lens, n_classes):
        mask = true == pred
        n_classes = "%s / %s" % ((np.unique(true).shape[0]), n_classes)
        return {'seq-level': "%0.1f" % (100*mask.mean()), 'base-level': "%0.1f" % (100*lens[mask].sum()/lens.sum()), 'n_classes': n_classes}

    results = dict()
    for colname in enc_df.columns:
        logger.info(f'computing results for {colname}')
        col = enc_df[colname].values
        results[colname] = get_results(col[seq_labels], col[seq_preds], seq_lens, n_orig_classes[colname])

    logger.info(f'computing results for {level}')
    results[level] = get_results(seq_labels, seq_preds, seq_lens, n_orig_classes[level])

    results['n'] = {'seq-level': len(seq_lens), 'base-level': seq_lens.sum(), 'n_classes': '-1'}

    results = pd.DataFrame(data=results)
    results.to_csv(args.output, sep=',')
    print(results)

def aggregate_seqs(argv=None):
    #import ..sequence as seq
    from ..sequence import DeepIndexFile
    from ..utils import get_logger
    from hdmf.common import get_hdf5io
    import h5py
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from tqdm import tqdm



    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=str, help='the summarized sequence NN outputs')
    parser.add_argument("output", type=str, help="the path to save results to")

    args = parser.parse_args(argv)

    logger = get_logger()

    f = h5py.File(args.summary, 'r')

    logger.info(f'loading summary results from {args.summary}')
    seq_outputs = f['outputs']
    seq_labels = f['labels'][:].astype(int)
    seq_lens = f['lengths'][:].astype(int)

    uniq_labels = np.unique(seq_labels)
    n_labels = len(uniq_labels)

    outf = h5py.File(args.output, 'w')
    agg_outputs = outf.create_dataset('outputs', shape=(n_labels, seq_outputs.shape[1]), dtype=float)
    agg_lens = outf.create_dataset('lengths', shape=(n_labels,), dtype=int)
    agg_labels = outf.create_dataset('labels', shape=(n_labels,), dtype=int)

    for i, lbl in tqdm(enumerate(uniq_labels), total=n_labels):
        mask = np.where(uniq_labels == lbl)[0]
        lens = seq_lens[mask]
        outputs = seq_outputs[mask]
        wave = np.average(outputs, weights=lens, axis=0)
        agg_outputs[i] = wave
        agg_lens[i] = lens.sum()
        agg_labels[i] = lbl

    outf.close()
    f.close()

def train_confidence_model(argv=None):
    import pickle

    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    import sklearn.metrics as skmet
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import MaxAbsScaler
    from ..utils import get_logger

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="the path to save the model to")
    parser.add_argument("-s", "--summary", type=str, help='the summarized sequence NN outputs')
    parser.add_argument('-O', '--onnx', default=False, action='store_true', help='Save data in ONNX format')
    parser.add_argument("-k", "--topk", metavar="TOPK", type=int, help="use the TOPK probabilities for building confidence model", default=None)
    parser.add_argument('-j', '--n_jobs', metavar='NJOBS', nargs='?', const=True, default=None, type=int,
                        help='the number of jobs to use for cross-validation. if NJOBS is not specified, use the number of folds i.e. 5')
    parser.add_argument('-c', '--cvs', metavar='NFOLDS', default=5, type=int,
                        help='the number of cross-validation folds to use. default is 5')
    parser.add_argument("-e", "--eval", action='store_true', help='evaluate the train confidence model', default=False)
    parser.add_argument("-f", "--force", action='store_true', help='force training i.e. do not use output if it exists', default=False)

    args = parser.parse_args(argv)

    if args.n_jobs and isinstance(args.n_jobs, bool):
        args.n_jobs = args.cvs

    logger = get_logger()

    X, y = None, None

    if args.summary:
        # train a new model
        logger.info(f"reading outputs summary data from {args.summary}")
        f = h5py.File(args.summary, 'r')

        true = f['labels'][:]
        pred = f['preds'][:]

        # the top-k probabilities, in descending order
        maxprobs = f['maxprob'][:, :args.topk]
        lengths = f['lengths'][:]

        f.close()

        X = np.concatenate([lengths[:, np.newaxis], maxprobs], axis=1)
        y = (true == pred).astype(int)

    if os.path.exists(args.output) and not args.force:
        # just convert the given model to ONNX
        with open(args.output, 'rb') as f:
            lr = pickle.load(f)
    else:
        scaler = MaxAbsScaler()
        scaler.fit(X)

        lrcv = LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=[.1, .5, .7, .9, .95, .99, 1], n_jobs=args.n_jobs, cv=args.cvs)
        logger.info(f"building confidence model with \n{lrcv}")

        lrcv.fit(scaler.transform(X), y)

        lr = LogisticRegression()
        lr.coef_ = scaler.transform(lrcv.coef_)
        lr.intercept_ = lrcv.intercept_
        lr.classes_ = lrcv.classes_

    if args.onnx:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        logger.info(f"saving {lr} to ONNX file {args.output}")
        initial_type = [('float_input', FloatTensorType([None, lr.coef_.shape[1]]))]
        onx = convert_sklearn(lr, target_opset=12, initial_types=initial_type, options={type(lr): {'zipmap': False}})
        with open(args.output, "wb") as f:
            f.write(onx.SerializeToString())
    else:
        logger.info(f"pickling {lr} to {args.output}")
        with open(args.output, 'wb') as f:
            pickle.dump(lr, f)

    if args.eval:
        if X is not None:
            logger.info(f"Evaluating model found in {args.output}")
            y_prob = lr.predict_proba(X)[:, 1] #cross_val_predict(lr, X, cv=args.cvs)

            base = os.path.splitext(args.output)[0]

            metrics = dict()

            fpr, tpr, _ = skmet.roc_curve(y, y_prob)
            auc = skmet.auc(fpr, tpr)
            metrics['auc'] = auc

            skmet.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            plt.title(f"AUC = {auc:0.4f}")
            plt.savefig(f'{base}.roc.png')

            prec, rec, _ = skmet.precision_recall_curve(y, y_prob)
            auc = skmet.auc(rec, prec)
            metrics['aupr'] = auc

            skmet.PrecisionRecallDisplay(precision=prec, recall=rec).plot()
            plt.title(f"AUPR = {auc:0.4f}")
            plt.savefig(f'{base}.pr.png')

            y_pred = (y_prob > 0.5).astype(int)

            metrics['acc'] = skmet.accuracy_score(y, y_pred)
            metrics['prec'] = skmet.precision_score(y, y_pred, average='macro')
            metrics['prec_w'] = skmet.precision_score(y, y_pred, average='weighted')
            metrics['rec'] = skmet.recall_score(y, y_pred, average='macro')
            metrics['rec_w'] = skmet.recall_score(y, y_pred, average='weighted')
            metrics['f1'] = skmet.f1_score(y, y_pred, average='macro')
            metrics['f1_w'] = skmet.f1_score(y, y_pred, average='weighted')

            print(pd.Series(data=metrics))

        else:
            logger.info("cannot evaluate calibration model without data")




if __name__ == '__main__':
    main()
