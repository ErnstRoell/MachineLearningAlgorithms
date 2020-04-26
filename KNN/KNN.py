import numpy as np
import matplotlib.pyplot as plt
import warnings

from collections import Counter
from math import sqrt
from matplotlib import style
import pickle

style.use('fivethirtyeight')

class KNN():
    def __init__(self,k=3,file_name=None):
        self.k = k
        if file_name is not None:
            self.load(file_name)

    def fit(self,data,labels):
        self.data = [i+[k] for i,k in zip(data,labels)]

    def predict(self,pts):
        res = []
        for pt in pts:
            dist = []
            for el in self.data:
                dist.append([np.linalg.norm(np.array(el[:-1])-np.array(pt)),el[-1]])
            
            votes = [el[-1] for el in sorted(dist)[:self.k]]
            vote_result = Counter(votes).most_common(1)[0][0]
            res.append(vote_result)

        return res
    
    def save(self,file_name="tmp.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump([self.data,self.k],f)
    
    def load(self,file_name="tmp.pkl"):
        with open(file_name,"rb") as f:
            self.data, self.k =  pickle.load(f)

    def score():
        pass


if __name__ == "__main__":
    dataset = [[1,2],
               [2,3],
               [3,1],
               [6,5],
               [7,7],
               [8,6]]

    labels = ['k','k','k','r','r','r']
    pts = [[5,7],[9,9],[1,1]]
    clf = KNN()
    clf.fit(dataset,labels)
    clf.save()
    clf.load()
    res = clf.predict(pts)
    print(res)

