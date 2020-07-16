import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle
import pandas as pd

def plot_results(path, tvt=True, pred=True, fig_height=7, rf_kwargs=None):
    n_plots = 1
    if pred:
        n_plots += 1
    if tvt:
        n_plots += 1
    plot_count = 1

    if rf_kwargs is None:
        rf_kwargs = dict()

    fig_height = 7
    plt.figure(figsize=(n_plots*fig_height, fig_height))

    # read data
    dat = None
    with h5py.File(path, 'r') as f:
        if 'viz_emb' in f:
            dat = f['viz_emb'][:]
        labels = f['labels'][:]
        train_mask = f['train'][:]
        test_mask = f['test'][:]
        outputs = f['outputs'][:]
        validate_mask = f['validate'][:]
        taxon_id = f['taxon_id'][:]
    if dat is None:
        from umap import UMAP
        umap = UMAP(n_components=2)
        dat = umap.fit_transform(outputs[:])

    uniq_labels = np.unique(labels)
    n_classes = len(uniq_labels)
    class_pal = sns.color_palette('tab10', n_classes)
    colors = np.array([class_pal[i] for i in labels])

    # plot embeddings
    plt.subplot(1, n_plots, plot_count)
    class_handles = list()
    for cl, col in zip(uniq_labels, class_pal):
        mask = labels == cl
        plt.scatter(dat[mask,0], dat[mask,1], s=0.5, c=[col], label=cl)
        class_handles.append(Circle(0, 0, color=col))
    plt.legend(class_handles, taxon_id)
    plt.title('/'.join(path.split('/')[-2:]))
    plot_count += 1

    # plot train/validation/testing data
    if tvt:
        pal = ['gray', 'red', 'yellow']
        plt.subplot(1, n_plots, plot_count)
        dsubs = ['train', 'validation', 'test'] # data subsets
        dsub_handles = list()
        for (mask, dsub, col) in zip([train_mask, validate_mask, test_mask], dsubs, pal):
            plt.scatter(dat[mask, 0], dat[mask, 1], s=0.1, c=[col], label=dsub)
            dsub_handles.append(Circle(0, 0, color=col))
        plt.legend(dsub_handles, dsubs)
        plot_count += 1

    # run some predictions and plot report
    if pred:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score

        # build random forest classifier
        rf_kwargs.setdefault('n_estimators', 30)
        rfc = RandomForestClassifier(**rf_kwargs)
        X_train = outputs[train_mask]
        y_train = labels[train_mask]
        rfc.fit(X_train, y_train)
        X_test = outputs[test_mask]
        y_test = labels[test_mask]
        y_pred = rfc.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # plot classification report
        df = pd.DataFrame(report)
        subdf = df[[str(l) for l in uniq_labels]].iloc[:-1]
        ax = plt.subplot(1, n_plots, plot_count)
        subdf.plot.barh(ax=ax, color=class_pal, legend=False)
        #plt.legend(taxon_id, loc='lower left', ncol=len(uniq_labels))
        plt.title('accuracy: %.4f' %  report['accuracy'])
    plt.tight_layout()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='the HDF5 file with network outputs')
    parser.add_argument('output', type=str, nargs='?', help='the file to save the summary figure to')

    args = parser.parse_args()
    if args.output is None:
        s = args.input
        if s.endswith('.h5'):
            s = s[:-3]
        args.output = s + '.png'
    plt.figure(figsize=(21, 7))
    plot_results(args.input)
    plt.savefig(args.output, dpi=100)


if __name__ == '__main__':
    main()

