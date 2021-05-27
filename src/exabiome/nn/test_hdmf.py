from pkg_resources import resource_filename
from os.path import join
from hdmf.common import load_namespaces, get_class, get_manager, EnumData, DynamicTable, available_namespaces
from hdmf.backends.hdf5 import HDF5IO
import exabiome.sequence  # load input data namespaces


load_namespaces(join(resource_filename(__name__, '../hdmf-ml'), 'schema', 'ml', 'namespace.yaml'))

TrainValidationTestMask = get_class('TrainValidationTestMask', 'hdmf-ml')
CrossValidationMask = get_class('CrossValidationMask', 'hdmf-ml')
ClassProbability = get_class('ClassProbability', 'hdmf-ml')
ClassLabel = get_class('ClassLabel', 'hdmf-ml')
RegressionOutput = get_class('RegressionOutput', 'hdmf-ml')
ClusterLabel = get_class('ClusterLabel', 'hdmf-ml')
EmbeddedValues = get_class('EmbeddedValues', 'hdmf-ml')
ResultsTable = get_class('ResultsTable', 'hdmf-ml')


manager = get_manager()

input_data_path = 'ar122_r95.rep.h5'
with HDF5IO(input_data_path, 'r', manager=manager) as io:
    read_file = io.read()
    taxa_table = read_file.taxa_table
    phylum_col = taxa_table.phylum
    print('input data loaded')

# test with adding all columns for now even though not all should be used at the same time
tvt_mask = TrainValidationTestMask(name='tvt_mask', description='tvt mask', elements=['train', 'validate', 'test'])
cv_mask = CrossValidationMask(name='cv_mask', description='cv mask', n_splits=10)
class_prob = ClassProbability(name='class_prob', description='', training_label=phylum_col)
class_label = ClassLabel(name='class_label', description='', training_label=phylum_col)
reg_output = RegressionOutput(name='reg_output', description='', training_label=phylum_col)
cluster_label = ClusterLabel(name='cluster_label', description='')
embedded_vals = EmbeddedValues(name='embedded_vals', description='')
cols = [tvt_mask, tvt_mask.elements, cv_mask, cv_mask.elements, class_prob, class_label, reg_output, cluster_label,
        embedded_vals]
table = ResultsTable(name='results', description='ml results')

table.add_row(
    tvt_mask='train',
    cv_mask=2,
    class_prob=0.15,
    class_label=3,
    reg_output=1.1,
    cluster_label=4,
    embedded_vals=[1., 2.]  # TODO should this be indexed?
)
# add two rows because of hdmf#594
table.add_row(
    tvt_mask='train',
    cv_mask=2,
    class_prob=0.15,
    class_label=3,
    reg_output=1.1,
    cluster_label=4,
    embedded_vals=[1., 2.]
)

with HDF5IO('test.h5', 'w', manager=manager) as io:
    io.write(table)


with HDF5IO('test.h5', 'r', manager=manager) as io:
    read_table = io.read()
    print(read_table)
    # TODO on read read_table.tvt_mask is not set correctly
    breakpoint()
    print(read_table['tvt_mask'])
    print('hi')
    # TODO cannot do read_table['tvt_mask'][:], see hdmf#595
