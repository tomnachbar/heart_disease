
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import sklearn           as sk


# Lê o arquivo CSV
df = pd.read_csv('heart_disease_dataset.csv',low_memory=False)

# %%
df.columns

# %% [markdown]
# # **Análise Exploratória**

# %%
df.shape


# %%
df.describe()

# %%
df.info()

# %%
print(df.isnull().sum())

# %%
duplicates = df.duplicated().sum()
print(f'Número de linhas duplicadas: {duplicates}')

# %% [markdown]
# *Verificamos que na coluna 6 - Alcohol Intake temos uma quantidade faltante de dados, então vamos explorá-la melhor a seguir*
# 

# %%
for column in df.columns:
    unique_values = df[column].unique()
    print(f'Unique values in {column}: {unique_values}')

# %%
missing_percentage = df.isnull().mean() * 100
print("Percentage of missing values for each column:")
print(missing_percentage)

# %%
df['Alcohol Intake'].unique()

# %% [markdown]
# *Para sanar os dados faltantes da coluna Alcohol Intake, vamos preencher com "Light" ou "None", somente para dar tratamentos aos dados.*
# 
# 

# %%
df['Alcohol Intake'] = df['Alcohol Intake'].fillna('Light/None')

# %%
df.hist()

# %% [markdown]
# *É possível notar pelos histogramas acima uma boa distribuição em cada variável. Notamos ainda que temos um target que é a coluna Heart Disease*

# %%
print(df['Heart Disease'].value_counts())


# %% [markdown]
# **Vamos plotar um gráfico para entender melhor a coluna Heart Disease e sua distribuição**

# %%
# Distribuição de Heart Disease por Gênero
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', hue='Heart Disease', data=df, palette='Set1')
plt.title('Distribuição de Heart Disease por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Contagem')
plt.legend(title='Heart Disease', loc='upper right', labels=['Não', 'Sim'])

# Contagem exata no topo das barras
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# %%
plt.figure(figsize=(12, 10))
sns.lmplot(x='Cholesterol', y='Blood Pressure', hue='Heart Disease', data=df, fit_reg=False, scatter_kws={'alpha':0.6}, palette='flare')
plt.title('Cholesterol vs Blood Pressure (Hued by Heart Disease)')
plt.xlabel('Cholesterol')
plt.ylabel('Blood Pressure')
plt.legend(title='Heart Disease', loc='upper right', labels=['Não', 'Sim'])

plt.show()

# %% [markdown]
# # **Transformando as colunas categóricas para torná-las numéricas, padronizando as variáveis contínuas.**

# %%
from sklearn.preprocessing import LabelEncoder

categorical_variables = ['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Chest Pain Type']

label_encoder = LabelEncoder()

for col in categorical_variables:
    df[col] = label_encoder.fit_transform(df[col])
    category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"Mapping for {col}: {category_mapping}")


# %% [markdown]
# **Matriz de Correlação**

# %%
corr_matrix = df.corr()

plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='cividis', vmin=-1, vmax=1)
plt.title('Matrix de Correlação')
plt.tight_layout()
plt.show()


# %%
df1 = df.copy()

# %%
from sklearn.preprocessing import StandardScaler

# %%
continuous_vars = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours', 'Stress Level', 'Blood Sugar']


# %%
scaler = StandardScaler()
df[continuous_vars] = scaler.fit_transform(df[continuous_vars])

# %% [markdown]
# # **Importando Modelos de Machine Learning para Teste e Treino**
# 
# 

# %% [markdown]
# **Logistic Regression**

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %%
X = df.drop(['Heart Disease'], axis = 1)
Y = df['Heart Disease']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)

# %%
print(X.shape, X_train.shape, X_test.shape)

# %%
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# **Treino**

# %%
log = LogisticRegression(penalty='l2',solver='lbfgs', max_iter=1000)
log.fit(X_train, y_train)

# %%
y_pred = log.predict(X_train)
train_accuracy = accuracy_score(y_pred, y_train)

# %%
print('Accuracy Score of Training Data:', train_accuracy)

# %% [markdown]
# **Teste**

# %%
y_pred_test = log.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)
class_report = classification_report(y_test, y_pred_test)

# %%
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# %% [markdown]
# ____________________________________________________________

# %% [markdown]
# **Random Forest**

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
random_model = RandomForestClassifier(n_estimators=100,max_depth=5,bootstrap=True,oob_score=True, random_state=42)
random_model.fit(X_train, y_train)

# %% [markdown]
# **Treino**

# %%
y_pred = random_model.predict(X_train)

# %%
train_accuracy = accuracy_score(y_train, y_pred)
print(f'Acurácia no conjunto de treino: {train_accuracy:.2f}')


# %% [markdown]
# **Teste**

# %%
y_pred_test = random_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Acurácia no conjunto de teste: {test_accuracy:.2f}')

# %%
conf_matrix = confusion_matrix(y_test, y_pred_test)
print('Matriz de Confusão:')
print(conf_matrix)

# %%
class_report = classification_report(y_test, y_pred_test)
print('Relatório de Classificação:')
print(class_report)

# %% [markdown]
# ____________________________________________________________

# %% [markdown]
# **Decision Tree Classifier**

# %%
from sklearn.tree import DecisionTreeClassifier

# %% [markdown]
# **Treino**

# %%
decision_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, min_samples_split=10, random_state=42)
decision_model.fit(X_train, y_train)

# %%
y_pred = decision_model.predict(X_train)

# %%
train_accuracy = accuracy_score(y_pred, y_train)

# %%
print('Accuracy Score of Training Data:', train_accuracy)

# %% [markdown]
# **Teste**

# %%
y_pred_test = decision_model.predict(X_test)

# %%
test_accuracy = accuracy_score(y_test, y_pred_test)

# %%
print('Accuracy Score of Test Data:', test_accuracy)

# %%
train_accuracy = accuracy_score(y_train, y_pred)
test_accuracy = accuracy_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)
class_report = classification_report(y_test, y_pred_test)

# %%
print(f"Random Forest - Training Accuracy: {train_accuracy}")
print(f"Random Forest - Testing Accuracy: {test_accuracy}")
print("Random Forest - Confusion Matrix:\n", conf_matrix)
print("Random Forest - Classification Report:\n", class_report)

# %%
models = ['Logistic Regression', 'Random Forest', 'Decision Tree']
training_accuracies = [0.86625, 1.0, 1.0]
testing_accuracies = [0.86, 1.0, 1.0]

# %%
plt.figure(figsize=(10, 6))
plt.bar(models, training_accuracies, width=0.4, align='center', label='Training Accuracy')
plt.bar(models, testing_accuracies, width=0.4, align='edge', label='Testing Accuracy')
plt.xlabel('Modelos')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácias de Treino e Teste')
plt.ylim(0.8, 1.1)  # Adjust y-axis limits if necessary
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# # **Interface Preditiva com Gradio**

# %%
from joblib import dump
import os

# Caminho onde deseja salvar os arquivos dentro do Space
save_path = './'  # Ou você pode usar 'models/' se preferir salvar em uma subpasta

# Salvar o modelo
dump(random_model, os.path.join(save_path, 'random_model.pkl'))

# Salvar o scaler
dump(scaler, os.path.join(save_path, 'scaler.joblib'))
# %%

# %%
import pandas as pd
import gradio as gr
from joblib import load
import os

# Caminho para o diretório onde os arquivos estão salvos dentro do Space
path = './'  # Ou 'models/' se você salvou os arquivos em uma subpasta

# Carregar o modelo e o scaler
random_model = load(os.path.join(path, 'random_model.pkl'))
scaler = load(os.path.join(path, 'scaler.joblib'))



# Função para fazer previsões com novas entradas
def predict_heart_disease(Age, Gender, Cholesterol, BloodPressure, HeartRate, Smoking, AlcoholIntake, ExerciseHours, FamilyHistory, Diabetes, Obesity, StressLevel, BloodSugar, ExerciseInducedAngina, ChestPainType):
    # Criar um dicionário com os dados de entrada
    input_data = {
        'Age': Age,
        'Gender': Gender,
        'Cholesterol': Cholesterol,
        'Blood Pressure': BloodPressure,
        'Heart Rate': HeartRate,
        'Smoking': Smoking,
        'Alcohol Intake': AlcoholIntake,
        'Exercise Hours': ExerciseHours,
        'Family History': FamilyHistory,
        'Diabetes': Diabetes,
        'Obesity': Obesity,
        'Stress Level': StressLevel,
        'Blood Sugar': BloodSugar,
        'Exercise Induced Angina': ExerciseInducedAngina,
        'Chest Pain Type': ChestPainType
    }

    # Criar um DataFrame com os dados de entrada
    input_df = pd.DataFrame([input_data])

    # Padronizar as variáveis contínuas
    input_df[continuous_vars] = scaler.transform(input_df[continuous_vars])

    # Fazer a previsão
    prediction = random_model.predict(input_df)[0]

    # Retornar o resultado
    return "Positive for Heart Disease" if prediction == 1 else "Negative for Heart Disease"

# Configuração da interface Gradio
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Gender: 0: Female, 1: Male"),
        gr.Number(label="Cholesterol"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Heart Rate"),
        gr.Number(label="Smoking: 0: Current, 1: Former, 2: Never"),
        gr.Number(label="Alcohol Intake: 0: Heavy, 1: Light/None, 2: Moderate"),
        gr.Number(label="Exercise Hours"),
        gr.Number(label="Family History: 0: No, 1: Yes"),
        gr.Number(label="Diabetes: 0: No, 1: Yes"),
        gr.Number(label="Obesity: 0: No, 1: Yes"),
        gr.Number(label="Stress Level"),
        gr.Number(label="Blood Sugar"),
        gr.Number(label="Exercise Induced Angina: 0: No, 1: Yes"),
        gr.Number(label="Chest Pain Type: 0: Asymptomatic, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Typical Angina")
    ],
    outputs="text",
    title="Heart Disease Prediction",
    description="Please enter the following details:"
)

iface.launch(share=True, debug=True)

# %%



