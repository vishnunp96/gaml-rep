import sys,os

print(sys.path)

sys.path.insert(0, os.path.abspath('../metadata'))

print(sys.path)

import directoryactions
