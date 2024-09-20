

import math 
import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas
from sklearn.preprocessing import LabelEncoder

from sklearn import datasets

class Perceptron:
    def __init__(self, taxa, pesos, atributes, teta, f):
        i = 0
        self.atributes = atributes
        self.teta = teta
        self.taxa = taxa
        self.pesos = pesos
        self.f = f

    def functionG(self, u):
        r = 1/(1 + math.exp(-u)) 
        return r
    
    def ReLu(self, u):
        if(u > 0):
            return u
        else:
            return 0

    def somatório(self, ind, pesos):
        soma = np.dot(ind, pesos) + self.teta
        return soma      
    
    def rodarPercep(self):
        u = self.somatório(self.atributes, self.pesos)
        if(self.f == 1):       
            yP = self.functionG(u)
        else:
            yP = self.ReLu(u)    
        return yP



class MLP:
    def __init__(self, cOcultas, neuroniosCOcultas, neuroniosTotais, tax, base, id, nSaidas, classificacao):
        self.cOc = cOcultas
        self.nCOc = neuroniosCOcultas
        self.nTotais = neuroniosTotais
        self.tax = tax
        self.id = id#1 se a base possuir o id na data e 0 se não
        self.nSaidas = nSaidas
        self.tetas = []
        k = 0
        vez = 0
        self.data = pandas.read_csv(base)
        self.classificacao = classificacao
        n = np.max(self.nCOc)
        lista = [nSaidas, n, len(self.data.iloc[0, :].values) - 2, len(self.data.iloc[0, :].values) - 1]
        m = np.max(lista) #Verifica o maior dado para criar uma matriz que n seja menor q o numero de pesos               
        self.weights = np.zeros(shape = (self.nTotais, m))  
        for _ in range(self.nTotais):
            self.tetas.append(-1)
        for _ in range(self.cOc):
            self.gerarPesos(self.nCOc[_ - vez], k, vez, self.nCOc[_])#tem q ser menos v para a camada receber o numero de pesos equivalente ao numero de neuronios na camada anteriror
            k += self.nCOc[_]
            if(vez == 0):
                vez+=1  
        self.gerarPesos(self.nCOc[-1], k, 1, self.nSaidas)   #TA GERANDO OS PESOS E AS BIAS, FALTA RODAR E OS SELF X E Y, QUE SÃO FORA DELE PQ ELE SO RECEBE OS PARAMETROS E CRIA UM MLP          
        self.tetas = np.array(self.tetas)
        self.weights = np.array(self.weights)             
        self.gerarXy()
        self.rodarMLP()
                    
    def gerarXy(self):
        label_encoder = LabelEncoder()
        self.data['classificacao'] = label_encoder.fit_transform(self.data[self.classificacao])
        mapping = {label: idx for idx, label in enumerate(self.data['classificacao'].unique())}
        self.data['classificacao'] = self.data['classificacao'].map(mapping)
        self.X = self.data.iloc[:, :-2].values
        self.y = self.data.iloc[:, -1].values
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(self.X, self.y, test_size=0.3, random_state=1)

    def gerarPesos(self, _, k, vez, num):
        if(self.id == 0 and vez == 0):
            for _2 in range(_):
                for _3 in range(len(self.data.iloc[0, :].values) - 1):
                    self.weights[_2][_3] = rd.uniform(-1, 1)
        elif(self.id == 1 and vez == 0):
            for _2 in range(_):
                for _3 in range(len(self.data.iloc[0, :].values) - 2):
                    self.weights[_2][_3] = rd.uniform(-1, 1)
        else:
            for _2 in range(num):
                for _3 in range(_):
                    self.weights[_2 + k][_3] = rd.uniform(-1, 1)                                     
    
    def softmax(self, saidas):
        probabilidades = []
        exps = []
        for _ in saidas:
            exps.append(math.exp(_))
        somatorio = sum(exps)
        for _ in range(self.nSaidas):
            probabilidades.append(exps[_]/somatorio)     
        return probabilidades
    
    def calculoCusto(self, i, prob):
        yReal = self.y[i]
        self.vetor = []
        custo = 0
        for _ in range(self.nSaidas):
            self.vetor.append(0)
        self.vetor[yReal] = 1
        for _ in range(self.nSaidas):   
            custo += self.vetor[_] * math.log(prob[_])
        custo = -(custo)
        return custo
    
    def functionG(self, u):
        r = 1/(1 + math.exp(-u)) 
        return r
    
    def DerivadaReLu(self, u):
        saida = []
        for _ in u:
            if(_ > 0):
                saida.append(1)
            else:
                saida.append(0)
        saida = np.array(saida)        
        return saida        
    
    def calculoGradiente(self, saidas, somas):
        k = 0
        vez = 0
        newTetas = []
        gCamadas = []
        saidasMat = []
        weightsMat = []
        PreWeightsMat = []
        PrePreWeightsMat = []
        somasMat = []
        for _ in range(self.cOc):
            saidasMat.append(np.array(saidas[_]))
            saidasMat[_] = np.array(saidasMat[_]).reshape((self.nCOc[_], 1))
            somasMat.append(np.array(somas[_]))
            somasMat[_] = np.array(somasMat[_]).reshape((self.nCOc[_], 1))
            for _2 in range(self.nCOc[_]):
                for _3 in range(len(self.atributes)):
                    PrePreWeightsMat.append(np.array(self.weights[_2 + k][_3]))
                PreWeightsMat.append(np.array(PrePreWeightsMat))
                PrePreWeightsMat = []    
            weightsMat.append(np.array(PreWeightsMat))
            PreWeightsMat = []
            k += self.nCOc[_]    
        saidasMat.append(np.array(saidas[-1]))
        saidasMat[-1] = np.array(saidasMat[-1]).reshape((self.nSaidas, 1))
        for _ in range(self.nSaidas):
            for _2 in range(len(self.atributes)):
                PrePreWeightsMat.append(np.array(self.weights[_ + k][_2])) 
            PreWeightsMat.append(np.array(PrePreWeightsMat))
            PrePreWeightsMat = []
        weightsMat.append(np.array(PreWeightsMat))
        self.vetor = np.array(self.vetor).reshape((self.nSaidas,1))
        self.atributes = np.array(self.atributes).reshape((len(self.atributes),1))#Transformar em lista
        self.tetas = np.array(self.tetas).reshape((self.nTotais, 1))#Transformar em lista
        for _ in saidas:
            if(vez == self.cOc):
                saidas[vez] = np.array(_).reshape((self.nSaidas, 1)) 
            else:
                saidas[vez] = np.array(_).reshape((self.nCOc[vez], 1))      
            vez +=1
        gSaida = saidas[-1] - self.vetor
        gSaidafWeights = np.dot(gSaida, np.transpose(saidasMat[-2]))
        gSaidaTetas = gSaida
        for _ in range(self.nSaidas):
            for _2 in range(self.nCOc[-1]):
                self.weights[_+k][_2] = self.weights[_+k][_2] - self.tax*gSaidafWeights[_][_2]
            self.tetas[_+k] = self.tetas[_+k] - self.tax*gSaidaTetas[_]
        k-=self.nCOc[-1]        
        for _ in range(self.cOc - 1):
            if(_ == 0):
                gCamadas.append(np.dot(np.transpose(weightsMat[-1 -(_)]),gSaida) * self.DerivadaReLu(somasMat[-1 -(_)]).reshape((self.nCOc[-1 -(_)],1)))
            else:
                gCamadas.append(np.dot(np.transpose(weightsMat[-1 -(_)]),gCamadas[-1]) * self.DerivadaReLu(somasMat[-1 -(_)]).reshape((self.nCOc[-1 -(_)],1)))  
            gCamadaWeights = np.dot(gCamadas[-1], np.transpose(saidas[-1-(_ + 2)]))
            gCamadaTetas = gCamadas[-1]
            for _2 in range(self.nCOc[-1 -(_)]):
                for _3 in range(self.nCOc[-2 - (_)]):
                    self.weights[_2+k][_3] = self.weights[_2+k][_3] - self.tax*gCamadaWeights[_2][_3]
                self.tetas[_2+k] = self.tetas[_2+k] - self.tax*gCamadaTetas[_2]
            k -= self.nCOc[-1 -(_)]
        gCamadas.append(np.dot(np.transpose(weightsMat[0]),gCamadas[-1]) * self.DerivadaReLu(somasMat[0]).reshape((self.nCOc[0],1)))
        gCamadaWeights = np.dot(gCamadas[-1], np.transpose(self.atributes))
        gCamadaTetas = gCamadas[-1]
        for _ in range(self.nCOc[0]):
            for _2 in range(len(self.atributes)):    
                self.weights[_][_2] = self.weights[_][_2] - self.tax*gCamadaWeights[_][_2]
            self.tetas[_+k] = self.tetas[_+k] - self.tax*gCamadaTetas[_]  
        for _ in range(self.nTotais):
            newTetas.append(self.tetas[_][0])
        self.tetas = newTetas  

    def predict(self, X):
        previsoes = []
        for instancia in X:
            self.atributes = np.delete(instancia, 0)
            saidas = []  
            somas = []
            elementos = []
            weights = []
            preSaida = []
            preSoma = []
            k = 0
            vez = 0
        
            for index, n in enumerate(self.nCOc):
                for e in range(n):
                    for el in self.weights[e+k]:
                        if(el != 0):
                            elementos.append(el)   
                    weights.append(elementos)
                    elementos = []    
                for _ in range(n):
                    if(index == 0):                
                        preSoma.append(Perceptron(self.tax, weights[_], self.atributes, self.tetas[_+k], 0).somatório(self.atributes, weights[_]))
                        z = Perceptron(self.tax, weights[_], self.atributes, self.tetas[_+k], 0).rodarPercep()
                        preSaida.append(z)
                    else:
                        preSoma.append(Perceptron(self.tax, weights[_], saidas[-1], self.tetas[_+k], 0).somatório(saidas[-1], weights[_]))
                        z = Perceptron(self.tax, weights[_], saidas[-1], self.tetas[_+k], 0).rodarPercep()
                        preSaida.append(z)
                    if(_ == n - 1):    
                        saidas.append(preSaida)
                        somas.append(preSoma)
                        preSaida = [] 
                        preSoma = []         
                k += n
                weights = [] 
            for e in range(self.nSaidas):
                for el in self.weights[e+k]:
                    if(el != 0):
                        elementos.append(el)
                weights.append(elementos)
                elementos = []
            for _ in range(self.nSaidas):
                preSoma.append(Perceptron(self.tax, weights[_], saidas[-1], self.tetas[_+k], 1).somatório(saidas[-1], weights[_]))
                z = Perceptron(self.tax, weights[_], saidas[-1], self.tetas[_+k], 1).rodarPercep()
                preSaida.append(z)
                if(_ == self.nSaidas - 1):
                    saidas.append(preSaida)
                    somas.append(preSoma)
                    preSaida = [] 
                    preSoma = []   
            weights = []           
            softmax = self.softmax(saidas[-1]) 
            previsao = np.argmax(softmax)
            previsoes.append(previsao)
    
        return previsoes

    def rodarMLP(self):
        custo = []
        custoTotal = 0
        for i,self.atributes in enumerate(self.X):
            self.atributes = np.delete(self.atributes, 0)
            saidas = []  
            somas = []
            elementos = []
            weights = []
            preSaida = []
            preSoma = []
            totalDeExemplos = i
            k = 0
            vez = 0
            for index, n in enumerate(self.nCOc):
                for e in range(n):
                    for el in self.weights[e+k]:
                        if(el != 0):
                            elementos.append(el)   
                    weights.append(elementos)
                    elementos = []    
                for _ in range(n):
                    if(index == 0):                
                        preSoma.append(Perceptron(self.tax, weights[_], self.atributes, self.tetas[_+k], 0).somatório(self.atributes, weights[_]))
                        z = Perceptron(self.tax, weights[_], self.atributes, self.tetas[_+k], 0).rodarPercep()
                        preSaida.append(z)
                    else:
                        preSoma.append(Perceptron(self.tax, weights[_], saidas[-1], self.tetas[_+k], 0).somatório(saidas[-1], weights[_]))
                        z = Perceptron(self.tax, weights[_], saidas[-1], self.tetas[_+k], 0).rodarPercep()
                        preSaida.append(z)
                    if(_ == n - 1):    
                        saidas.append(preSaida)
                        somas.append(preSoma)
                        preSaida = [] 
                        preSoma = []         
                k += n
                weights = [] 
            for e in range(self.nSaidas):
                for el in self.weights[e+k]:
                    if(el != 0):
                        elementos.append(el)
                weights.append(elementos)
                elementos = []
            for _ in range(self.nSaidas):
                preSoma.append(Perceptron(self.tax, weights[_], saidas[-1], self.tetas[_+k], 1).somatório(saidas[-1], weights[_]))
                z = Perceptron(self.tax, weights[_], saidas[-1], self.tetas[_+k], 1).rodarPercep()
                preSaida.append(z)
                if(_ == self.nSaidas - 1):
                    saidas.append(preSaida)
                    somas.append(preSoma)
                    preSaida = [] 
                    preSoma = []   
            weights = []           
            softmax = self.softmax(saidas[-1]) 
            custo.append(self.calculoCusto(i, softmax))
            self.calculoGradiente(saidas, somas)
        for i in range(totalDeExemplos):
            custoTotal += custo[i]
        custoMedio = custoTotal/totalDeExemplos#CUSTO MEDIO JA CALCULADO   
                
             
            
mlp = MLP(2, [4, 4], 11, 10, "Iris.csv", 1, 3, "Species")
previsoes = mlp.predict(mlp.X_teste)
acuracia = metrics.accuracy_score(mlp.y_teste, previsoes)
print(f'Acurácia: {acuracia}')

