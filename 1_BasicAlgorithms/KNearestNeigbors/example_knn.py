import numpy as np
import matplotlib.pyplot as plt
import warnings

from collections import Counter
from math import sqrt
from matplotlib import style

style.use('fivethirtyeight')


def knn(data,pt,k=3):
#     if len(data)>=k:
#         warnings.warn('Choose k larger')

    dist = []
    for el in data:
        dist.append([np.linalg.norm(np.array(el[:-1])-np.array(pt)),el[-1]])
    
    votes = [el[-1] for el in sorted(dist)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
       
    return vote_result




if __name__ == "__main__":
    dataset = [[1,2,'k'],
               [2,3,'k'],
               [3,1,'k'],
               [6,5,'r'],
               [7,7,'r'],
               [8,6,'r']]

    pt = [5,7]
    res = knn(dataset,pt)
    print(res)




#     label = ['r','k']

#     print([i+[k] for i,k in zip(pt,label)])

#     for i,k in zip(pt,label):
#         print(i)
#         print(k)
#         print(i+[k])

#     for i in dataset:
#         for ii in dataset[i]:
#             plt.scatter(ii[0],ii[1],s=100,color=i)
# 
#     plt.scatter(pt[0],pt[1],s=100,color='g')
# 
#     plt.show()

#     dataset = {'k':[[1,2],[2,3],[3,1]],
#                'r':[[6,5],[7,7],[8,6]]}
