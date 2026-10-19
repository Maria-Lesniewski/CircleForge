import numpy as np
import sys

if len(sys.argv) != 3:
    print("Pass the file and column to bin resp")
    print("i.e. bin_this.py <file> <colidx>")
    exit(1)

this = sys.argv[1]
col = int(sys.argv[2])

data = np.loadtxt(this)

to_bin = data[:,col]

hist, bins = np.histogram(to_bin, bins=50)
db = bins[1] - bins[0]
print(db)
print(bins)
np.savetxt("binned"+ this, np.column_stack((bins[:-1],hist/sum(hist*db))))
