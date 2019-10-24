from pkg_resources import resource_filename
from os.path import join
from hdmf.common import load_namespaces

load_namespaces(join(resource_filename(__name__, 'schema'), 'deep_index.namespace.yaml'))
