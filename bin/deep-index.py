import sys
import exabiome

cmds = exabiome.get_commands()


def print_help():
    print('Usage: deep-index <command> [options]')
    print('Available commands are:\n')
    for c, f in cmds.items():
        nspaces = 16 - len(c)
        desc = ''
        if isinstance(f.__doc__, str):
            desc = f.__doc__.split('\n')[0]
        else:
            desc = 'Run function %s from %s' % (f.__name__, f.__module__)
        print(f'    {c}' + ' '*nspaces + desc)

if len(sys.argv) == 1:
    print_help()
else:
    cmd = sys.argv[1]
    func = cmds[cmd]
    func(sys.argv[2:])
