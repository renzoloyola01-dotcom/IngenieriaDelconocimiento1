import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore') 
print("=== INICIANDO EL EJERCICIO 1 ===")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelos = {
    'LogisticRegression': (
        LogisticRegression(max_iter=10000), 
        {'C': [0.01, 0.1, 1, 10, 100]}
    ),
    'DecisionTree': (
        DecisionTreeClassifier(random_state=42), 
        {'max_depth': [3, 5, 7, 10, None], 'min_samples_leaf': [1, 5, 10]}
    ),
    'SVM': (
        SVC(random_state=42), 
        {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
    ),
    'KNN': (
        KNeighborsClassifier(), 
        {'n_neighbors': [3, 5, 9, 15, 21], 'weights': ['uniform', 'distance']}
    )
}
for nombre, (modelo, parametros) in modelos.items():
    grid = GridSearchCV(modelo, parametros, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    
    print(f"\nModelo: {nombre}")
    print(f"Mejores parámetros: {grid.best_params_}")
    print(f"Mejor F1-score: {grid.best_score_:.4f}")
    print("-" * 40)
    
    
    
    
    
    
    
#ejercicio 2 

from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

print("\n=== INICIANDO EL EJERCICIO 2: MANEJO DE DESBALANCE ===")

# 1. Aplicar SMOTE a los datos de entrenamiento (crea datos sintéticos)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 2. Definir los modelos base (sin balanceo)
modelos_base = {
    'LogisticRegression': LogisticRegression(max_iter=10000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

# 3. Definir los modelos con class_weight='balanced' (KNN no va aquí)
modelos_balanced = {
    'LogisticRegression': LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42),
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'SVM': SVC(class_weight='balanced', random_state=42)
}


for nombre in modelos_base.keys():
    print(f"\n--- {nombre} ---")
    
    modelo = modelos_base[nombre]
    modelo.fit(X_train, y_train)
    pred_normal = modelo.predict(X_test)
    recall_normal = recall_score(y_test, pred_normal, pos_label=0)
    print(f"Recall (Normal):    {recall_normal:.4f}")
    
    if nombre in modelos_balanced:
        modelo_bal = modelos_balanced[nombre]
        modelo_bal.fit(X_train, y_train)
        pred_bal = modelo_bal.predict(X_test)
        recall_bal = recall_score(y_test, pred_bal, pos_label=0)
        print(f"Recall (Balanced):  {recall_bal:.4f}")
    else:
        print("Recall (Balanced):  No aplica para este algoritmo")
        
    modelo_smote = modelos_base[nombre] 
    modelo_smote.fit(X_train_smote, y_train_smote)
    pred_smote = modelo_smote.predict(X_test)
    recall_smot = recall_score(y_test, pred_smote, pos_label=0)
    print(f"Recall (Con SMOTE): {recall_smot:.4f}")
    
print("-" * 40)


#EJERCICIO 3: 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

print("\n=== INICIANDO EL EJERCICIO 3: DATASET PROPIO (FÚTBOL) ===")

# 1. Cargar el dataset (Ahora sí funcionará porque el archivo está en la carpeta)
df_futbol = pd.read_csv('results.csv')
df_sample = df_futbol.sample(n=3000, random_state=42).copy()

# 2. Crear la variable objetivo: 1 si gana el local, 0 si empata o pierde
df_sample['victoria_local'] = (df_sample['home_score'] > df_sample['away_score']).astype(int)

# 3. Seleccionar features
features = ['home_team', 'away_team', 'tournament', 'city', 'neutral']
X_propio = df_sample[features].copy()
y_propio = df_sample['victoria_local']

# 4. Preprocesamiento: Convertir texto a números para que los modelos lo entiendan
le = LabelEncoder()
for col in ['home_team', 'away_team', 'tournament', 'city']:
    X_propio[col] = le.fit_transform(X_propio[col])
X_propio['neutral'] = X_propio['neutral'].astype(int)

# 5. Dividir y escalar datos
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_propio, y_propio, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_p = scaler.fit_transform(X_train_p)
X_test_p = scaler.transform(X_test_p)

# 6. Definir modelos (Agregando Random Forest como pedía el ejercicio)
modelos_ej3 = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42), 
    'KNN': KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(random_state=42)
}

# 7. Entrenar y preparar gráfica de Curvas ROC
plt.figure(figsize=(10, 8))
print("Resultados de los Modelos (Gana Local vs No Gana):")

for nombre, modelo in modelos_ej3.items():
    modelo.fit(X_train_p, y_train_p)
    y_pred = modelo.predict(X_test_p)
    y_prob = modelo.predict_proba(X_test_p)[:, 1]
    
    acc = accuracy_score(y_test_p, y_pred)
    f1 = f1_score(y_test_p, y_pred)
    print(f"{nombre:<20} -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    
    fpr, tpr, _ = roc_curve(y_test_p, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{nombre} (AUC = {roc_auc:.2f})')

# 8. Dibujar gráfica
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curvas ROC - Comparación de los 5 Modelos')
plt.legend(loc="lower right")
plt.show()