from .lit import AbstractLit

_models = dict()

def add(k, cls):
    global _models
    _models[k] = cls

def model(short_hand):
    def dec(cls):
        global _models
        _models[short_hand] = cls
        #add(short_hand, cls)
        cls.short_hand = short_hand
        return cls
    return dec


#from . import roznet
import importlib
import pkgutil


def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results

import_submodules(__name__)

#import pkgutil
#
#__all__ = []
#for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
#    __all__.append(module_name)
#    _module = loader.find_module(module_name).load_module(module_name)
#    globals()[module_name] = _module
