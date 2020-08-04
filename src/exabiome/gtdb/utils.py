import os

def get_taxa_id(path):
    c, n = os.path.basename(path).split('_')[0:2]
    return c + '_' + n
