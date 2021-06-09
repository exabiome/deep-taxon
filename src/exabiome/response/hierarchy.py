import numpy as np
import pandas as pd


_ranks = {'phylum', 'class', 'order', 'family', 'genus'}
def check_rank(rank):
    if rank not in _ranks:
        raise ValueError(f'please choose rank from {_ranks}')
    return rank

def collapse(dist, mdf, rank, linkage='average'):
    """
    Args:
        dist:         the original distance matrix
        mdf:          the metadata DataFrame
        rank:         the rank to collapse on
        linkage:      the linkage method to use when collapsing
                      Options are 'average', 'complete', 'single'
                      Default is 'average'

    Return: ndarray (n_target_taxon, n_target_taxon)
        square distance matrix for the specificed taxonomic rank
    """
    rank = check_rank(rank)
    if linkage == 'average':
        _link = np.mean
    elif linkage == 'complete':
        _link = np.max
    elif linkage == 'single':
        _link = np.min
    else:
        raise ValueError("Unrecognized linkage - '%s'. Please choose "
                         "from 'average', 'complete', or 'single'")

    r_types = np.unique(mdf[rank])
    r_dist = np.zeros((r_types.shape[0], r_types.shape[0]))
    for i, type_i in enumerate(r_types):
        mask_i = mdf[rank] == type_i
        dist_i = dist[mask_i]
        for j, type_j in enumerate(r_types[i+1:], i+1):
            mask_j = mdf[rank] == type_j
            r_dist[i,j] = _link(dist_i[:,mask_j])
            r_dist[j,i] = r_dist[i,j]
    return r_dist, r_types


def expand_labels(parent_rank, labels, target_rank, mdf):
    """
    Args:
        parent_rank:  the upper taxonomic rank
        labels:       the labels for each unique parent taxon
        target_rank:  the lower taxonomic rank
        mdf:          the metadata DataFrame

    Return: ndarray
        array with the target taxonomic rank labels
    """
    parent_rank = check_rank(parent_rank)
    target_rank = check_rank(target_rank)
    df = pd.DataFrame()
    df[parent_rank] = np.unique(mdf[parent_rank])
    if len(df[parent_rank]) != labels.shape[0]:
        raise ValueError("please provide one label for each parent")
    for d in range(labels.shape[1]):
        df['col%d' % d] = labels[:,d]

    res = mdf.merge(df, on=parent_rank)
    # grouping by target rank assumes there are
    # no parent ranks that contain the same target rank
    # For example, the same genus name does not exist in different
    # families
    if target_rank == 'species':
        return res[df.columns[1:]].values
    else:
        return res.groupby(target_rank)[df.columns[1:]].mean().values


def supervised_emb(dist, mdf, parent_rank, parent_labels, target_rank, model, linkage='average'):
    """
    Args:
        dist:               the original distance matrix
        mdf:                the metadata DataFrame
        parent_rank:        the upper taxonomic rank
        parent_labels:      the embedding for each parent
        target_rank:        the lower taxonomic rank
        model:              the UMAP object to run supervised embedding with
        mdf:                the metadata DataFrame
        linkage:            the linkage method to use when collapsing
                            distance matrix
                            Options are 'average', 'complete', 'single'
                            Default is 'average'

    Returns: (t_emb, t_dist, targets)
        t_emb:              the embedding of the target taxonomic rank
        t_dist:             the distance matrix for the target taxonomic rank
        targets:            the names for taxons in the target taxonomic rank
    """
    t_dist, targets = collapse(dist, mdf, target_rank)
    t_labels = expand_labels(parent_rank, parent_labels, target_rank, mdf)
    model.set_params(target_metric='l2')
    t_emb = model.fit_transform(t_dist, y=t_labels)
    return t_emb, t_dist, targets
