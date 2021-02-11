import sys
import glob
import os.path
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import scipy.stats as stats


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

        # we won't have these three if we are looking
        # at non-representatives
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

    viz_emb = None
    if 'viz_emb' in path:
        logger.info('found viz_emb')
        viz_emb = path['viz_emb']
    # else:
    #     logger.info('calculating UMAP embeddings for visualization')
    #     from umap import UMAP
    #     umap = UMAP(n_components=2)
    #     viz_emb = umap.fit_transform(outputs)
    else:
        n_plots = 1

    color_labels = getattr(pred, 'classes_', None)
    if color_labels is None:
        color_labels = labels
    class_pal = get_color_markers(len(np.unique(color_labels)))

    # set up figure
    fig_height = 7
    plt.figure(figsize=(n_plots*fig_height, fig_height))

    if viz_emb:
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
        y_pred = pred
        y_test = labels
        if not isinstance(pred, (np.ndarray, list)):
            if pred is None or pred is True:
                logger.info('No classifier given, using RandomForestClassifier(n_estimators=30)')
                pred = RandomForestClassifier(n_estimators=30)
            elif not (hasattr(pred, 'fit') and hasattr(pred, 'predict')):
                raise ValueError("argument 'pred' must be a classifier with an SKLearn interface")

            X_test = outputs
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

    uniq_seqs = np.unique(seq_ids)
    X_mean = np.zeros((uniq_seqs.shape[0], outputs.shape[1]))
    X_median = np.zeros((uniq_seqs.shape[0], outputs.shape[1]))
    y = np.zeros(uniq_seqs.shape[0], dtype=int)
    seq_len = np.zeros(uniq_seqs.shape[0], dtype=int)
    seq_viz = None
    if viz_emb is not None:
        seq_viz = np.zeros((uniq_seqs.shape[0], 2))

    for seq_i, seq in enumerate(uniq_seqs):
        seq_mask = seq_ids == seq
        uniq_labels = labels[seq_mask]
        if not np.all(uniq_labels == uniq_labels[0]):
            raise ValueError(f'Found more than one label for sequence {seq}')
        y[seq_i] = uniq_labels[0]
        X_mean[seq_i] = outputs[seq_mask].mean(axis=0)
        X_median[seq_i] = np.median(outputs[seq_mask], axis=0)
        if seq_viz is not None:
            seq_viz[seq_i] = viz_emb[seq_mask].mean(axis=0)
        seq_len[seq_i] = olens[seq_mask].sum()

    seq_len = np.log10(seq_len)

    fig, axes = None, None
    figsize_factor = 7
    class_pal = None
    if isinstance(clf, (list, np.ndarray)):
        nrows = 2
        ncols = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*figsize_factor, ncols*figsize_factor))
        axes = np.expand_dims(axes, axis=1)
        all_preds = np.argmax(outputs, axis=1)
        class_pal = get_color_markers(outputs.shape[1])
    else:
        color_labels = getattr(clf, 'classes_', None)
        if color_labels is None:
            color_labels = labels
        class_pal = get_color_markers(len(np.unique(color_labels)))

        nrows = 3 if seq_viz is not None else 2
        ncols = 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey='row', figsize=(nrows*figsize_factor, ncols*figsize_factor))

        # classifier from MEAN of outputs
        output_mean_preds = clf.predict(X_mean)
        make_plots(seq_viz, y, output_mean_preds, axes[:,0], class_pal, seq_len, 'Mean classification')

        # classifier from MEDIAN of outputs
        output_median_preds = clf.predict(X_median)
        make_plots(seq_viz, y, output_median_preds, axes[:,1], class_pal, seq_len, 'Median classification')

        # classifier from voting with chunk predictions
        all_preds = clf.predict(outputs)

    vote_preds = np.zeros(X_mean.shape[0], dtype=int)
    for seq_i, seq in enumerate(uniq_seqs):
        seq_mask = seq_ids == seq
        vote_preds[seq_i] = stats.mode(all_preds[seq_mask])[0][0]

    make_plots(seq_viz, y, vote_preds, axes[:,-1], class_pal, seq_len, 'Vote classification')

    plt.tight_layout()


def make_plots(seq_viz, true, pred, axes, pal, seq_len, title):
    if seq_viz is not None:
        plot_seq_emb(seq_viz, pred, axes[-3], pal=pal)
        axes[0].set_title(title)
        axes[0].set_xlabel('Mean of first UMAP dimension')
        axes[0].set_ylabel('Mean of second UMAP dimension')
    plot_clf_report(true, pred, axes[-2], pal=pal)
    plot_acc_of_len(true, pred, seq_len, axes[-1])


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
    type_group = parser.add_argument_group('Problem type').add_mutually_exclusive_group()
    type_group.add_argument('-C', '--classify', action='store_true', help='run a classification problem', default=False)
    type_group.add_argument('-M', '--manifold', action='store_true', help='run a manifold learning problem', default=False)

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

    outputs = read_outputs(args.input)
    if args.classify:
        plt.figure(figsize=(7, 7))
        labels = outputs['labels']
        model_outputs = outputs['outputs']
        if 'test_mask' in outputs:
            mask = outputs['test_mask']
            labels = labels[mask]
            model_outputs = model_outputs[mask]

        pred = np.argmax(model_outputs, axis=1)
        class_pal = get_color_markers(model_outputs.shape[1])
        colors = np.array([class_pal[i] for i in labels])
        ax = plt.gca()
        plot_clf_report(labels, pred, ax=ax, pal=class_pal)
    else:
        plt.figure(figsize=(21, 7))
        pretrained = False
        if args.classifier is not None:
            with open(args.classifier, 'rb') as f:
                pred = pickle.load(f)
            pretrained = True
        else:
            pred = RandomForestClassifier(n_estimators=30)
        pred = plot_results(outputs, pred=pred, name='/'.join(args.input.split('/',)[-2:]), logger=logger)
        if not pretrained:
            clf_path = os.path.join(outdir, 'summary.rf.pkl')
            logger.info(f'saving classifier to {clf_path}')
            with open(clf_path, 'wb') as f:
                pickle.dump(pred, f)
    logger.info(f'saving figure to {fig_path}')
    plt.savefig(fig_path, dpi=100)

    if args.aggregate_chunks:
        logger.info(f'running summary by aggregating chunks within sequences')
        aggregated_chunk_analysis(outputs, pred)
        agg_fig_path = os.path.join(outdir, 'summary.aggregated.png')
        logger.info(f'saving figure to {agg_fig_path}')
        plt.savefig(agg_fig_path, dpi=100)


if __name__ == '__main__':
    main()
