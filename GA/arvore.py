import pandas 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split  
from sklearn import metrics  
dados = pandas.read_csv("diabetes.csv")
print(dados)
coluna = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
#armazenar os dados das colunas no x
X = dados[coluna]
#armazendar os dados de saida no y
y = dados.Outcome
#test_size(Porcentagem q vai treinar(0.7) e a que vai realizar o teste(0.3), random_state(Aforma q ele vai pegar os dados de forma aleatória))
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=1)
arvore = DecisionTreeClassifier(max_depth=90, criterion='log_loss', min_samples_split=0.1)
arvore.fit(X_treino, y_treino)
previsao = arvore.predict(X_teste)
acuracia = metrics.accuracy_score(y_teste, previsao)
print(acuracia)