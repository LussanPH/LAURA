import random as rd
import numpy as np
from math import e
from math import exp
from sklearn import datasets
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, taxa, num):
        self.pesos = []
        i = 0
        self.teta = -1
        self.taxa = taxa
        self.num = num
        self.gerarXy()
        while(i != len(self.X[1])):
            self.pesos.append(rd.uniform(-1, 1))
            i+=1    

    def gerarXy(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        not_virginica_indices = np.where(y != 2)
        self.X = X[not_virginica_indices]
        self.y = y[not_virginica_indices]

    def functionG(self, u):
        r = 1/(1 + exp(-u)) 
        return r
    
    def somatório(self, ind, pesos):
        soma = np.dot(ind, pesos) + self.teta
        return soma   
            
    def fronteiraDeDecisao(self, x):
        return -(self.pesos[0]*x + self.teta)/self.pesos[2]         

    def rodarPercep(self):
        for _ in range(self.num):
            for i, ind in enumerate(self.X):
                yR = self.y[i]
                u = self.somatório(ind, self.pesos)       
                yP = self.functionG(u)
                for k in range(len(self.pesos)):
                    self.pesos[k] += self.taxa*(yR - yP) * ind[k] 
                self.teta += self.taxa*(yR - yP)
            print(self.pesos)    
            self.gerarGrafico() 
            i+=1   
        
    def gerarGrafico(self):
        fig, varx = plt.subplots() 
        x1 = []
        x2 = []
        x = []
        y = []
        i = 0
        k = 0
        j = 0
        m = 2
        while(i != len(self.X)):
            x1.append(self.X[i][k])  
            x2.append(self.X[i][m]) 
            i += 1 
        x = np.linspace(np.min(self.X[:, 0]), np.max(self.X[:, 0]), 100)
        y = self.fronteiraDeDecisao(x)               
        varx.scatter(x1, x2, c=self.y)#forma o grafico a função plot
        varx.plot(x, y, c="#FF0000")
        plt.xlabel("Característica 1")
        plt.ylabel("Característica 3")
        plt.title("Gráfico de Decisão do Perceptron")
        plt.show()#printa o grafico a função show        





Ptron = Perceptron(0.1, 20)
Ptron.rodarPercep()
