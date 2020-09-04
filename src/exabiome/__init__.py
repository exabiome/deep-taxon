_attr_name = '_command'
def command(name):
    def dec(func):
        setattr(func, _attr_name, name)
        return func
    return dec

def get_command(func):
    return getattr(func, _attr_name, None)

from . import nn, gtdb, tools
