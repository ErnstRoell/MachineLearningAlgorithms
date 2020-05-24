from statistics import mean 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

class LinearRegression():
    def __init__(self,file_name=None):
        if file_name is not None:
            self.load(file_name)


    def fit(self,xs,ys):
        self.m = (mean(xs)*mean(ys) - mean(xs*ys)) / \
                (mean(xs)**2 - mean(xs**2))
        self.b = mean(ys) - self.m*mean(xs)

    def predict(self,xs):
        return [(self.m*x + self.b) for x in xs]
    
    def save(self,file_name="tmp.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump([self.m,self.b],f)
    
    def load(self,file_name=None):
        with open(file_name,"rb") as f:
            self.m, self.b =  pickle.load(f)

    def score():
        pass


if __name__ == "__main__":
    style.use('fivethirtyeight')

    # Here we load a model and use it to predict
    xs = np.array([1,2,3,4,5,6],dtype=np.float64)
    ys = np.array([5,4,6,5,6,7],dtype=np.float64)

    clf = LinearRegression('tmp.pkl')
    y_predict = clf.predict(xs)

    # Now we train a custom model and save it 

    clf = LinearRegression()
    clf.fit(xs,ys)
    clf.save()

    # Now we predict
    y_predict = clf.predict(xs)

    # Plot the results 
    plt.scatter(xs,ys)
    plt.plot(xs,y_predict)
    plt.show()
