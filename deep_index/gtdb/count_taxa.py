import sys
import numpy as np


count = 1

n_open = 0
n_close = 0

close = False
closed_chars = set()
with open(sys.argv[1], 'r') as f:
    while True:
        c = f.read(1)
        if not c:
            break
        if close:
            closed_chars.add(c)
        close = False
        if c == '(':
            n_open += 1
            count += 1
        elif c == ')':
            n_close += 1
            close = True

n_floats = count*(count-1)//2
total_size = n_floats*np.dtype('float').itemsize/(1024**3)

print(count, n_floats, total_size)
print(n_open, n_close)
print(closed_chars)
