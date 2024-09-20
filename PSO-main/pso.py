import random as rd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas
class Particula:
    def __init__(self, v, pA):
        self.v = v
        self.pA = pA
        self.mL = None
        self.Vn = None
        self.c1 = rd.uniform(0, 3)
        self.c2 = rd.uniform(0, 3)
        self.pN = None        

    def melhorLocal(self):     
        if(type(self.mL) == type(None)):
            if(pso.model != 2):
                self.mL = self.pA
                self.mLAccuracy = pso.avaliacao(self.mL)
                return self.mL   
            else:
                listaFlo2 = []
                self.mL = self.pA
                self.mLAccuracy = pso.avaliacao(self.mL)
                listaFlo2 = [self.mLAccuracy, self.mL]
                return listaFlo2
        else:
            if(pso.model != 2):
                if(pso.avaliacao(self.pA) > self.mLAccuracy):
                    self.mL = self.pA
                    self.mLAccuracy = pso.avaliacao(self.mL)  
                return self.mL     
            else:
                listaFlo2 = []
                pAAccuracy = pso.avaliacao(self.pA)
                if(pAAccuracy > self.mLAccuracy):
                    self.mL = self.pA
                    self.mLAccuracy = pAAccuracy
                listaFlo2 = [self.mLAccuracy, self.mL]
                return listaFlo2 
    
    def randomC(self):
        self.c1 = rd.uniform(0, 3)
        self.c2 = rd.uniform(0, 3)                  

    def atualizarVelocidade(self, mG):
        self.Vn = self.v + self.c1*(self.mL - self.pA) + self.c2*(mG - self.pA)

    def posicaoNova(self):
        self.pN = self.Vn + self.pA       

    def __str__(self):
        return f"posicao atual: {self.pA} melhor local: {self.mL}"        
    
class PSO:
    def __init__(self, num, enx, model, dados):
        self.num = num
        self.enx = enx
        self.model = model
        self.dados = pandas.read_csv(dados)

    def gerarPosicoesAleatorias(self):
        self.listaP = []
        i = 0
        if(self.model == 0):
            while(i != self.num):
                pos = [rd.randint(3, 150), rd.randint(0,2), rd.uniform(0.01, 0.9)]
                self.listaP.append(pos)
                self.Pinicial = [76, 1, 0.55]
                i+=1
        elif(self.model == 1):
            while(i != self.num):
                vizinhos = 2
                while(vizinhos % 2 == 0):
                    vizinhos = rd.randint(3, 11)
                pos = [vizinhos, rd.randint(0, 1)]
                self.listaP.append(pos)
                self.Pinicial = [5, 0]
                i+=1 
        elif(self.model == 2):
            while(i != self.num):
                pos = [rd.randint(1, 100), rd.randint(3, 150), rd.randint(0,2), rd.uniform(0.01, 0.9)]
                self.listaP.append(pos)
                self.Pinicial = [50, 76, 1, 0.55]
                i+=1
        else:
            while(i != self.num):
                pos = [rd.uniform(0.1, 1), rd.randint(0, 3)]
                self.listaP.append(pos)
                self.Pinicial = [0.5, 2]
                i+=1                           
        return self.listaP

    def calcularVIniciais(self):
        self.listaV = []
        self.listaP = np.array(self.listaP)
        np.set_printoptions(formatter={'float' : "{: 0.2f}".format})
        print(self.listaP)
        for item in self.listaP:
            sub = item - self.Pinicial                
            self.listaV.append(sub)
        print(self.listaV)

    def gerarXy(self):
        self.X = self.dados.iloc[:, :-1].values #pega todas as colunas menos o target
        self.y = self.dados.iloc[:, -1].values  #pega apenas a coluna do target
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(self.X, self.y, test_size=0.3, random_state=1) 

    def avaliacao(self, posicao):
        if(self.model == 0):
            tradutor = {0.00:"gini", 1.00:"entropy", 2.00:"log_loss"}
            arvore = DecisionTreeClassifier(max_depth=int(posicao[0]), criterion=tradutor[int(posicao[1])], min_samples_split=posicao[2])
            arvore.fit(self.X_treino, self.y_treino)
            previsao = arvore.predict(self.X_teste)
            acuracia = metrics.accuracy_score(self.y_teste, previsao)
        elif(self.model == 1):
            tradutor = {0.00:"uniform", 1.00:"distance"}
            knn = KNeighborsClassifier(n_neighbors=int(posicao[0]), weights=tradutor[int(posicao[1])])
            knn.fit(self.X_treino, self.y_treino)
            previsao = knn.predict(self.X_teste)
            acuracia = metrics.accuracy_score(self.y_teste, previsao)
        elif(self.model == 2):
            tradutor = {0.00:"gini", 1.00:"entropy", 2.00:"log_loss"}
            floresta = RandomForestClassifier(n_estimators=int(posicao[0]), max_depth=int(posicao[1]), criterion=tradutor[int(posicao[2])], min_samples_split=posicao[3])
            floresta.fit(self.X_treino, self.y_treino)
            previsao = floresta.predict(self.X_teste)
            acuracia = metrics.accuracy_score(self.y_teste, previsao)
        else:
            tradutor = {0.00:"linear", 1.00:"poly", 2.00:"rbf", 3.00:"sigmoid"}
            svm = SVC(C=posicao[0], kernel=tradutor[int(posicao[1])])
            svm.fit(self.X_treino, self.y_treino)
            previsao = svm.predict(self.X_teste)
            acuracia = metrics.accuracy_score(self.y_teste, previsao)    
        return acuracia
    def gerarParticula(self, v, pA):
        self.p = Particula(v, pA)
        return self.p
    
    def gerarGlobal(self, mL, f):
        maior = 0
        i = 0
        k = 0
        melhor = 0
        self.acuracia = 0
        if(self.model != 2):
            self.lista = []
            for ml in mL:
                self.lista.append(self.avaliacao(ml))
            print(self.lista)
        else:
            if(f == 0):
                self.lista = []
                self.listaFlo = []
                for ml in mL:
                    self.listaFlo.append(ml[1])
                    self.lista.append(ml[0])
                print(self.lista)    
            else:
                for ml in mL:
                    resultado = (ml[1] == self.listaFlo[k]).all()
                    if(resultado == False):
                        print(1)
                        self.lista[k] = ml[0]    
                    k+=1    
                print(self.lista) 
        if(self.model != 2):              
            for item in self.lista:
                if(item > maior):
                    maior = item
                    melhor = i
                i+=1
            self.acuracia = self.lista[melhor]        
            return mL[melhor]   
        else:
            for item in self.lista:
                if(item > maior):
                    maior = item
                    melhor = i
                i+=1
            self.acuracia = self.lista[melhor]
            return self.listaFlo[melhor]         

    def limitar(self, particula):
        if(self.model == 0):
            if(particula.pN[0] > 150 or particula.pN[0] < 3):
                particula.pN[0] = rd.randint(3, 150)
            if(particula.pN[1] > 2 or particula.pN[1] < 0):    
                particula.pN[1] = rd.randint(0, 2)
            if(particula.pN[2] > 0.9 or particula.pN[2] < 0.01):
                particula.pN[2] = rd.uniform(0.01, 0.9)
        elif(self.model == 1):
            if(particula.pN[0] > 11 or particula.pN[0] < 3):
                vizinhos = 2
                while(vizinhos % 2 == 0):
                    vizinhos = rd.randint(3, 11)
                particula.pN[0] = vizinhos   
            if(particula.pN[1] > 1 or particula.pN[1] < 0):    
                particula.pN[1] = rd.randint(0, 1)
        elif(self.model == 2):
            if(particula.pN[0] > 100 or particula.pN[0] < 1):
                particula.pN[0] = rd.randint(1, 100)
            if(particula.pN[1] > 150 or particula.pN[1] < 3):
                particula.pN[1] = rd.randint(3, 150)
            if(particula.pN[2] > 2 or particula.pN[2] < 0):    
                particula.pN[2] = rd.randint(0, 2)
            if(particula.pN[3] > 0.9 or particula.pN[3] < 0.01):
                particula.pN[3] = rd.uniform(0.01, 0.9)
        else:
            if(particula.pN[0] > 1 or particula.pN[0] < 0.1):
                particula.pN[0] = rd.uniform(0.1, 1)
            if(particula.pN[1] < 0 or particula.pN[1] > 3):
                particula.pN[1] = rd.randint(0, 3)                          
    
    def gerarEnxame(self):  
        listaE = []
        listaML = []
        i = 0
        j = 0
        f = 0
        while(i != self.num):
            listaE.append(self.gerarParticula(self.listaV[i], self.listaP[i]))
            listaML.append(listaE[i].melhorLocal())
            print(listaE[i].mLAccuracy)
            i+=1    
        i = 0    
        mG = self.gerarGlobal(listaML, f)
        f+=1
        while(j != self.enx):
            if(j > 0):
                listaML.clear()
                while(i != self.num):
                    listaML.append(listaE[i].melhorLocal())
                    i+=1                 
            mG = self.gerarGlobal(listaML, f)    
            for particula in listaE:    
                particula.atualizarVelocidade(mG)
                particula.posicaoNova()
                self.limitar(particula)
                particula.v = particula.Vn
                particula.randomC()
                particula.pA = particula.pN
                print(particula," com acuracia: ",particula.mLAccuracy)
            print("melhor global: ", mG, " com acuracia: ", self.acuracia)
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            i=0
            j+=1
           


pso = PSO(10, 50, 3, "diabetes (1).csv")
print(pso.gerarPosicoesAleatorias())
pso.gerarXy()
print("------------------------------\n---------------------------")
pso.calcularVIniciais()
pso.gerarEnxame()
