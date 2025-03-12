import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Carregar os dados de treino
feat = pd.read_csv(r'C:\Users\bianc\OneDrive\Documentos\train_images.csv', header=None)
resp = pd.read_csv(r'C:\Users\bianc\OneDrive\Documentos\train_labels.csv')

# Visualizar as primeiras linhas
print(feat.head())
print(resp.head())

feat = feat / 255.0 # é um processo de normalização dos dados para o intervalo de 0 e 1, este processo é importante porque ajuda a melhorar a performance e a convergência dos modelos,  fazendo com que o modelo trate os dados de forma mais consistente e eficiente. 

# Este código visualiza a distribuição das classes na variável 'Volcano?'. Utilizei o gráfico de barras para verificar se as classes estão equilibradas ou se há um desbalanceamento entre elas. 
resp['Volcano?'].value_counts().plot(kind='bar')
plt.title('Distribuição de Classes')
plt.xlabel('Classe')
plt.xticks(rotation=0)
plt.ylabel('Número de Exemplos')
plt.show()

# Utilizei diferentes classificadores para resolver o problema de classificação binária (prever se há ou não um vulcão na imagem). O balanceamento das classes foi configurado através do parâmetro 'class_weight="balanced"'. 
# Escolhi os classificadores: Logistic Regression, SVM, Decision Tree e Random Forest, porque eles abrangem métodos lineares, não lineares e baseados em árvores, permitindo avaliar o desempenho de abordagens diversas no conjunto de dados.
# Decidi não incluir alguns dos classificadores, como LDA, QDA e Naive Bayes, porque: 
# - LDA e QDA assumem distribuições gaussianas dos dados, o que pode ser inadequado para este problema com alta dimensionalidade (p > n) e imagens não processadas.
# - Naive Bayes simplifica as relações entre as variáveis (assume independência condicional), o que pode não capturar bem as nuances dos padrões visuais das imagens.
# O objetivo principal foi minimizar os falsos negativos, pois perder uma imagem que contém um vulcão centralizado pode ser mais crítico para o contexto do projeto. 

# Dividir os dados entre treino e teste 
X_train, X_val, y_train, y_val = train_test_split(feat, resp['Volcano?'], test_size=0.2, random_state=42)

# Exemplos de classificadores com `class_weight='balanced'
classifiers = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=500, random_state=42),
    "SVM": SVC(kernel='linear', class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
}

# Treinar e avaliar cada classificador
for name, clf in classifiers.items():
    print(f"\nTreinando {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f"Resultados para {name}:")
    print(classification_report(y_val, y_pred))


# Utilizei os classificadores Linear Regression, Lasso, Ridge, Naive Bayes, LDA, QDA e KNN. Esses modelos incluem métodos lineares, probabilísticos e baseados em vizinhança, permitindo uma comparação ampla de abordagens. O objetivo é avaliar o desempenho de cada um no conjunto de validação e identificar o mais adequado para a classificação binária.

# Dicionário de classificadores restantes
classifiers = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(n_neighbors=5)  # 5 vizinhos como padrão
}

# Treinamento e avaliação dos classificadores
for name, clf in classifiers.items():
    print(f"\nTreinando {name}...")
    clf.fit(X_train, y_train)  # Treina o modelo no conjunto de treinamento
    y_pred = clf.predict(X_val)  # Realiza previsões no conjunto de validação

    # Exibe as métricas de desempenho
    print(f"Resultados para {name}:")
    print(classification_report(y_val, y_pred))
    print("Matriz de confusão:")
    print(confusion_matrix(y_val, y_pred))