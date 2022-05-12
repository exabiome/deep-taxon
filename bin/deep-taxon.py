import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', 'src'))
sys.path.insert(0, path)

from deep_taxon import main
if __name__ == '__main__':
    main()
