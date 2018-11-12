import numpy as np
import json


with open('polblog_label.txt') as f:
    lst = json.load(f)

nnz = 0
for item in lst:
    if item is list:
        nnz += len(item)
    else:
        nnz += 1

out = np.zeros( [nnz,2])

count=0
for i,item in zip(range(len(lst)),lst):
    if item is list:
        for j in item:
            out[count,0] = i
            out[count,1] = j-1
            count += 1 
    else:
        out[count,0] = count
        out[count,1] = item
        count += 1 


np.savetxt('polblog_label.lst',out,fmt='%d\t%d',delimiter='\t')