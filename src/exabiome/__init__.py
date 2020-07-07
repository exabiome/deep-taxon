_commands = dict()
def command(name):
    def dec(func):
        _commands[name] = func
        return func
    return dec

def get_commands():
    return _commands.copy()

from . import nn
