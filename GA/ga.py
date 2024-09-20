import random as rd
import pandas
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split  
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ga:
    dados = pandas.read_csv("heart (1).csv")
    def __init__(self, tax, num, ger, model):
        self.tax = tax
        self.num = num
        self.ger = ger
        self.model = model

    def individado(self):
        if(self.model == 0):
            prof = rd.randint(3, 150)
            met = rd.randint(0, 2)
            min = rd.uniform(0.001, 0.9)
            exemplo = [prof, met, min]       
        elif(self.model == 1):
            vizinhos = 2
            while(vizinhos % 2 == 0):
                vizinhos = rd.randint(3, 11)
            peso = rd.randint(0,1)
            exemplo = [vizinhos, peso]
        elif(self.model == 2):
            numarvores = rd.randint(1, 100)
            prof = rd.randint(3, 150)
            met = rd.randint(0, 2)
            min = rd.uniform(0.001, 0.9)
            exemplo = [numarvores, prof, met, min]
        else:
            c = rd.uniform(0.1, 1)
            kernel = rd.randint(0, 3)
            exemplo = [c, kernel]      
        return exemplo[:]
    
    def geracao(self):
        self.lista = []
        i = 0
        while(i != self.num):
            self.lista.append(self.individado())
            i+=1        
         
    def selecao(self):
        a = rd.randint(0, (self.num - 1))
        self.ind1 = self.lista[a]

        self.lista.pop(a)
        b = rd.randint(0, (self.num - 2))

        self.ind2 = self.lista[b]


        if(a == (self.num - 1)):
            self.lista.append(self.ind1)
        else:
            self.lista.insert(a, self.ind1)   
              

    def comparar(self, n1, n2):   
        if(n1 > n2):
            return self.ind1
        else:
            return self.ind2

    def gerarXY(self):
        self.X = self.dados.iloc[:, :-1].values #pega todas as colunas menos o target
        self.y = self.dados.iloc[:, -1].values  #pega apenas a coluna do target
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(self.X, self.y, test_size=0.3, random_state=1)

    def fitness(self):
        p = self.ind1
        if(self.model == 0):
            tradutor = {0:"gini", 1:"entropy", 2:"log_loss"}
            arvore = DecisionTreeClassifier(max_depth=p[0], criterion=tradutor[p[1]], min_samples_split=p[2])
            arvore.fit(self.X_treino, self.y_treino)
            previsao = arvore.predict(self.X_teste)
            self.acuracia1 = metrics.accuracy_score(self.y_teste, previsao) 

            p = self.ind2
            arvore = DecisionTreeClassifier(max_depth=p[0], criterion=tradutor[p[1]], min_samples_split=p[2])
            arvore.fit(self.X_treino, self.y_treino)
            previsao = arvore.predict(self.X_teste)
            self.acuracia2 = metrics.accuracy_score(self.y_teste, previsao)
            maior = self.comparar(self.acuracia1, self.acuracia2)
        
        elif(self.model == 1):
            tradutor = {0:"uniform", 1:"distance"}
            knn = KNeighborsClassifier(n_neighbors=p[0], weights=tradutor[p[1]])
            knn.fit(self.X_treino, self.y_treino)
            previsao = knn.predict(self.X_teste)
            self.acuracia1 = metrics.accuracy_score(self.y_teste, previsao)

            p = self.ind2
            knn = KNeighborsClassifier(n_neighbors=p[0], weights=tradutor[p[1]])
            knn.fit(self.X_treino, self.y_treino)
            previsao = knn.predict(self.X_teste)
            self.acuracia2 = metrics.accuracy_score(self.y_teste, previsao)
            maior = self.comparar(self.acuracia1, self.acuracia2)           
        
        elif(self.model == 2):
            tradutor = {0:"gini", 1:"entropy", 2:"log_loss"}
            floresta = RandomForestClassifier(n_estimators=p[0], max_depth=p[1], criterion=tradutor[p[2]], min_samples_split=p[3])
            floresta.fit(self.X_treino, self.y_treino)
            previsao = floresta.predict(self.X_teste)
            self.acuracia1 = metrics.accuracy_score(self.y_teste, previsao)

            p = self.ind2
            floresta = RandomForestClassifier(n_estimators=p[0], max_depth=p[1], criterion=tradutor[p[2]], min_samples_split=p[3])
            floresta.fit(self.X_treino, self.y_treino)
            previsao = floresta.predict(self.X_teste)
            self.acuracia2 = metrics.accuracy_score(self.y_teste, previsao)
            maior = self.comparar(self.acuracia1, self.acuracia2)

        else:
            tradutor = {0:"linear", 1:"poly", 2:"rbf", 3:"sigmoid"}
            svm = SVC(C=p[0], kernel=tradutor[p[1]])
            svm.fit(self.X_treino, self.y_treino)
            previsao = svm.predict(self.X_teste)
            self.acuracia1 = metrics.accuracy_score(self.y_teste, previsao)

            p = self.ind2
            svm = SVC(C=p[0], kernel=tradutor[p[1]])
            svm.fit(self.X_treino, self.y_treino)
            previsao = svm.predict(self.X_teste)
            self.acuracia2 = metrics.accuracy_score(self.y_teste, previsao)
            maior = self.comparar(self.acuracia1, self.acuracia2)
        return maior[:]
    
    def crossing(self, melhores, int):
        
        first = melhores[0 + int]
        second = melhores[1 + int]
        genes1 = []
        genes2 = []
        genes1.append(first[1])
        genes2.append(second[1])
        if(self.model == 0 or self.model == 2):
            genes1.append(first[2])
            genes2.append(second[2])
            if(self.model == 0):
                c = rd.randint(7, 8)
                if(c == 8):
                    first.pop(1)
                    first.pop(1)
                    first.append(genes2[0])
                    first.append(genes2[1])
                    second.pop(1)
                    second.pop(1)
                    second.append(genes1[0])
                    second.append(genes1[1])
                else:
                    first.pop(2) 
                    first.append(genes2[1])
                    second.pop(2)
                    second.append(genes1[1])

            else:
                genes1.append(first[3])
                genes2.append(second[3])
                c = rd.randint(1, 3)
                if(c == 1):
                    first.pop(1)
                    first.pop(1)
                    first.pop(1)
                    first.append(genes2[0])
                    first.append(genes2[1])
                    first.append(genes2[2])     
                    second.pop(1)
                    second.pop(1) 
                    second.pop(1)
                    second.append(genes1[0])
                    second.append(genes1[1])
                    second.append(genes1[2])
                elif(c == 2):
                    first.pop(2)
                    first.pop(2)
                    first.append(genes2[1])
                    first.append(genes2[2])
                    second.pop(2)
                    second.pop(2)
                    second.append(genes1[1])
                    second.append(genes1[2])
                else:
                    first.pop(3)
                    first.append(genes2[2])
                    second.pop(3)
                    second.append(genes1[2])             

        else:
            first.pop(1)
            first.append(genes2[0])
            second.pop(1)
            second.append(genes1[0])

        melhores[0 + int] = first
        melhores[1 + int] = second
        self.mutacao(first)
        self.mutacao(second) 
    
    def mutacao(self, ind):
        z = 0
        if(self.model == 0):    
            while(z != len(ind)):
                mutacao = rd.randint(1, 100)
                if(mutacao <= self.tax):
                    if(z == 0):
                        ind[0] = rd.randint(3, 150)
                    elif(z == 1):
                        ind[1] = rd.randint(0, 2)
                    else:
                        ind[2] = rd.uniform(0.001, 0.9)
                z+=1 

        elif(self.model == 1):
            while(z != len(ind)):
                mutacao = rd.randint(1,100)
                if(mutacao <= self.tax):
                    if(z==0):
                        ind[0] = rd.randint(3, 7)
                        while(ind[0] % 2 == 0):
                            ind[0] = rd.randint(3, 7)
                    else:
                        ind[1] = rd.randint(0, 1)   
                z+=1 

        elif(self.model == 2):
            while(z != len(ind)):
                mutacao = rd.randint(1, 100)
                if(mutacao <= self.tax):
                    if(z == 0):
                        ind[0] = rd.randint(1, 100)
                    elif(z == 1):
                        ind[1] = rd.randint(3, 150)
                    elif(z == 2):
                        ind[2] = rd.randint(0, 2)
                    else:
                        ind[3] = rd.uniform(0.001, 0.9) 
                z+=1 
        else:
            while(z != len(ind)):
                mutacao = rd.randint(1, 100)
                if(mutacao <= self.tax):
                    if(z == 0):
                        ind[0] = rd.uniform(0.1, 1)
                    else:
                        ind[1] = rd.randint(0,3)
                z+=1                                      
                            
    def melhorDaGeracao(self):
        maior = 0
        if(self.model == 0):    
            for item in self.lista:
                tradutor = {0:"gini", 1:"entropy", 2:"log_loss"}
                arvore = DecisionTreeClassifier(max_depth=item[0], criterion=tradutor[item[1]], min_samples_split=item[2])
                arvore.fit(self.X_treino, self.y_treino)
                previsao = arvore.predict(self.X_teste)
                acuracia = metrics.accuracy_score(self.y_teste, previsao)
                if(acuracia > maior):
                    maior = acuracia
            
        elif(self.model == 1):
            for item in self.lista:
                tradutor = {0:"uniform", 1:"distance"}
                knn = KNeighborsClassifier(n_neighbors=item[0], weights=tradutor[item[1]])
                knn.fit(self.X_treino, self.y_treino)
                previsao = knn.predict(self.X_teste)
                acuracia = metrics.accuracy_score(self.y_teste, previsao)
                if(acuracia > maior):
                    maior = acuracia

        elif(self.model == 2):
            for item in self.lista:
                tradutor = {0:"gini", 1:"entropy", 2:"log_loss"}
                floresta = RandomForestClassifier(n_estimators=item[0], max_depth=item[1], criterion=tradutor[item[2]], min_samples_split=item[3])
                floresta.fit(self.X_treino, self.y_treino)
                previsao = floresta.predict(self.X_teste)
                acuracia = metrics.accuracy_score(self.y_teste, previsao)
                if(acuracia > maior):
                    maior = acuracia

        else:
            for item in self.lista:
                tradutor = {0:"linear", 1:"poly", 2:"rbf", 3:"sigmoid"}
                svm = SVC(C=item[0], kernel=tradutor[item[1]])
                svm.fit(self.X_treino, self.y_treino)
                previsao = svm.predict(self.X_teste)
                acuracia = metrics.accuracy_score(self.y_teste, previsao)
                if(acuracia > maior):
                    maior = acuracia  
        return maior                     

metodo = ga(tax = 50, num = 10, ger = 10, model = 3)
melhores = []
selecionados = []
i=0
j=0
k=0
f=0

metodo.gerarXY()
metodo.geracao()
while(f != metodo.num):
    if(metodo.num - f >= 5):
        print(metodo.lista[f],metodo.lista[f+1],metodo.lista[f+2],metodo.lista[f+3],metodo.lista[f+4])
        f+=5
    else:
        if(metodo.num - f == 4):
            print(metodo.lista[f],metodo.lista[f+1],metodo.lista[f+2],metodo.lista[f+3])
            f+=4  
        elif(metodo.num - f == 3):
            print(metodo.lista[f],metodo.lista[f+1],metodo.lista[f+2])
            f+=3
        elif(metodo.num - f == 2):
            print(metodo.lista[f],metodo.lista[f+1])
            f+=2
        else:
            print(metodo.lista[f])
            f+=1             
f=0
print("---------------------------------------------")
print("---------------------------------------------")
while(j != metodo.ger):
    while(i != metodo.num/2):    
        metodo.selecao()
        selecionados.append(metodo.fitness())
        metodo.selecao()
        selecionados.append(metodo.fitness())
        metodo.crossing(selecionados, k)
        k+=2
        i+=1
    metodo.lista = selecionados[:]
    while(f != metodo.num):
        if(metodo.num - f >= 5):
            print(metodo.lista[f],metodo.lista[f+1],metodo.lista[f+2],metodo.lista[f+3],metodo.lista[f+4])
            f+=5
        else:
            if(metodo.num - f == 4):
                print(metodo.lista[f],metodo.lista[f+1],metodo.lista[f+2],metodo.lista[f+3])
                f+=4
            elif(metodo.num - f == 3):
                print(metodo.lista[f],metodo.lista[f+1],metodo.lista[f+2])
                f+=3
            elif(metodo.num - f == 2):
                print(metodo.lista[f],metodo.lista[f+1])
                f+=2
            else:
                print(metodo.lista[f])
                f+=1
    f=0
    print("---------------------------------------------")
    print("---------------------------------------------")   
    selecionados.clear()
    melhores.append(metodo.melhorDaGeracao())
    k=0
    i=0
    j+=1
print("------------LISTA----------------")
print(melhores)
