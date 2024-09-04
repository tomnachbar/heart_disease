# Projeto de Ciência de Dados: Identificação de Doenças Cardíacas com Machine Learning

![*Heart*](https://github.com/tomnachbar/heart_disease/blob/fb706e346de9b0b41be3f139a1aa3602d7ecd20f/heart.jpeg)

## 1. Compreensão do Negócio

**• Descrição do Problema:** O principal problema é a criação de um método de aprendizado de máquina que seja capaz de prever com qualidade e boa acurácia se determinado paciente terá doença cardíaca, através de entrada de dados, e que isso traduza-se em um modelo que seja aplicável e que poderão vir a ser utilizado como apoio em diagnósticos médicos.

**•	Importância do Problema:**
	A importância do projeto está em contribuir com diagnósticos médicos, e também auxiliar as campanhas médicas com uso de indicadores estatísticos a fim de promover saúde pública e incentivar a população a adotar um estilo de vida mais saudável.

**•	Objetivos de Negócio:**
    Auxiliar nas previsões estatísticas de pessoas que poderão desenvolver doenças cardíacas e com isso incentivar campanhas de conscientização.
    Auxiliar em sistemas médicos para uma abordagem efetiva nos diagnósticos e possíveis pontos de melhoria de vida do paciente.

**•	Critérios de Sucesso:**
    O projeto obterá sucesso se for possível identificar com boa porcentagem de acurácia a entrada de novos dados através de um modelo de classificação testado e avaliado, realizando as devidas previsões corretamente. 



## 2. Compreensão dos Dados

**•	Coleta de Dados Inicial:**
	Os dados utilizados para realizar esse projeto foram adquiridos através do site Kaggle, disponibilizados de maneira pública e gratuita no link: https://www.kaggle.com/datasets/rashadrmammadov/heart-disease-prediction

**•	Descrição dos Dados:**
Os dados estão divididos em 1.000 linhas com 16 colunas, variando em dados categóricos e numéricos. A análise foi realizada com dados gerais sem um prazo pré-determinado. Temos uma coluna alvo, chamada ‘Heart Disease’, o qual será nosso guia para utilizarmos os modelos de previsão e classificação posteriormente. Abaixo exploramos um pouco melhor a coluna e os demais dados presentes no conjunto. 

**• Exploração dos Dados:**
No gráfico abaixo distribuímos  a coluna de Heart Disease por gênero para melhor visualização 


![*Heart Disease by Gender*](https://huggingface.co/spaces/nachbars/heart_disease/resolve/main/distribution_by_gender.jpg)



Também podemos verificar a distribuição das demais colunas numéricas: 

**•	Verificação da Qualidade dos Dados:**


![*Numerical Columns*](https://huggingface.co/spaces/nachbars/heart_disease/resolve/main/numerical_distribuition.jpg)

Na coluna de ‘Alcoohol Intake’ verificamos uma quantidade de dados faltantes conforme ilustrado abaixo, e faremos o tratamento delas posteriormente:


![*Missing Values*](https://huggingface.co/spaces/nachbars/heart_disease/resolve/main/missing_values.jpg)

 


## 3. Preparação dos dados:

**Começamos com um tratamento de dados:**
* Para sanar os dados faltantes da coluna Alcohol Intake, vamos preencher com "Light" ou "None", somente para dar tratamentos aos dados.

df1['Alcohol Intake'] = df1['Alcohol Intake'].fillna('Light/None')

* Em seguida vamos codificar nossas colunas categóricas para que utilizemos nos nossos modelos de Machine Learning: 

from sklearn.preprocessing import LabelEncoder

categoricas= ['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Chest Pain Type']

label_encoder = LabelEncoder()
for col in categoricas:
 df1[col] = label_encoder.fit_transform(df1[col])


* Em seguida demos uma olhada para ver como essas variáveis se correlacionam:

![*Correlation Matrix*](https://huggingface.co/spaces/nachbars/heart_disease/resolve/main/correlation_matrix.jpg)

* Também foi necessário padronizar os valores das colunas contínuas para escalar melhor nos modelos:


continuous_vars = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours', 'Stress Level', 'Blood Sugar']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df1[continuous_vars] = scaler.fit_transform(df1[continuous_vars])


## 4. Modelagem

**•	Seleção da Técnica de Modelagem:**
	Para esse projeto iremos utilizar um treino e teste dos modelos Logistic Regression, Random Forest, e Decision Tree e verificar qual deles melhor se adequa ao conjunto de dados.
	Também iremos separar os modelos com train-test-split, separando 20% do nosso dataset para utilizar nos modelos.
    
**•	Desenvolvimento dos Modelos:**
	Escolha dos Hiperparâmetros: Definimos e ajustamos os principais hiperparâmetros para cada modelo. Por exemplo:
    
o	Logistic Regression: Ajustamos parâmetros como C (regularização) e solver (método de otimização).

o	Random Forest: Ajustamos parâmetros como n_estimators (número de árvores) e max_depth (profundidade máxima das árvores).

o	Decision Tree: Ajustamos parâmetros como max_depth e min_samples_split (número mínimo de amostras necessárias para dividir um nó).

**•  Normalização e Padronização:** Para melhorar a performance dos modelos, aplicamos técnicas de normalização e padronização aos dados quando necessário. Por exemplo, no caso da Logistic Regression, os dados foram padronizados para garantir que todas as variáveis contribuam igualmente para o treinamento do modelo.

**•  Codificação de Variáveis Categóricas:** As variáveis categóricas foram codificadas usando LabelEncoder para transformar valores não numéricos em valores numéricos que podem ser utilizados pelos algoritmos de Machine Learning.

**• Treinamento dos Modelos:**
Divisão dos Dados: Utilizamos o train_test_split para dividir o dataset em conjuntos de treinamento e teste, assegurando que 20% dos dados sejam utilizados para teste e o restante para treinamento.
Treinamento: Cada modelo foi treinado utilizando o conjunto de dados de treinamento, aplicando as técnicas e hiperparâmetros ajustados para obter o melhor desempenho possível.

**•	Teste dos Modelos:**
•	Avaliação dos Modelos:
Após treinar os modelos Logistic Regression, Random Forest e Decision Tree, realizamos a avaliação utilizando o conjunto de dados de teste (20% do dataset).
Utilizamos métricas como Acurácia, Precisão, Recall, e F1-Score para medir o desempenho de cada modelo. A Acurácia fornece a proporção de previsões corretas entre todas as previsões realizadas. A Precisão avalia a proporção de previsões positivas corretas em relação ao total de previsões positivas feitas. O Recall mede a capacidade do modelo de identificar todas as instâncias positivas reais. O F1-Score é a média harmônica da precisão e recall, oferecendo uma visão mais equilibrada do desempenho do modelo.

**•	Matriz de Confusão:**
Analisamos a Matriz de Confusão para cada modelo para entender melhor as previsões corretas e incorretas, identificando padrões de erro, como falsas positivas e falsas negativas.
 


## 5. Avaliação

**•	Avaliação dos Resultados:**
o	Nos modelos testados podemos verificar conforme gráfico abaixo que atingimos bons resultados de acurácia para Logistic Regression algo em torno de 0,88 no Treino e 0,84 para o Teste. No entanto os melhores resultados foram observados para os modelos de Random Forest e Decision Tree, ambos com 100% de acurácia. Foram realizados outros comparativos como Recall, F1 Score, para verificação de possível overfitting, porém os mesmos sempre atingiram a mesma acurácia, devido ao pequeno tamanho do dataset.

**•	Revisão dos Modelos:**
o	Não houve necessidade de revisar ou melhorar os modelos, pois o conjunto de dados utilizado foi pequeno e não houve necessidade de ajustes.

**•	Validação do Modelo:**
o	Para garantir a validação e robustez dos dados foram utilizados os métodos de Holdout validation, que consiste em separar parte dos dados para o teste, garantindo que houvesse avaliação nos dados nunca vistos pelo modelo. Além disso foram avaliadas outras métricas como Recall, F1 Score e possíveis medidas de overfitting. Isso proporcionou uma visão mais abrangente do desempenho do modelo em diferentes aspectos, como precisão e capacidade de generalização. E por fim, no final do projeto fora realizada o modelo preditivo com Gradio para garantir que o modelo está operando de forma coerente com os dados disponibilizados.

![*Accuracies Model Comparation*](https://huggingface.co/spaces/nachbars/heart_disease/resolve/main/accuracies_models.jpg)


O produto final do projeto
Painel online, hospedado em um Cloud e disponível para acesso em qualquer dispositivo conectado à internet.
O painel pode ser acessado através desse link: https://tomnachbar-zomato.streamlit.app/

## 6. Implantação
**Planejamento da Implantação:**

•	Para a implantação do modelo, o plano inclui a criação de uma API que permitirá a integração do modelo com outros sistemas. Esta API será exposta para que aplicações externas possam enviar dados de entrada e receber as previsões em tempo real.

**Implementação Técnica:**
•	O processo de implantação será realizado utilizando Gradio para criar uma interface interativa que facilita a interação com o modelo. A infraestrutura será baseada no Hugging Face para garantir que o ambiente de produção seja consistente com o ambiente de desenvolvimento.

**Monitoramento e Manutenção:**
•	Após a implantação, o modelo será monitorado através de logs e métricas de desempenho, como a precisão ao longo do tempo, para detectar possíveis drifts nos dados. Manutenções periódicas serão planejadas para atualizar o modelo com novos dados, garantindo que ele continue a fornecer previsões precisas.


## 7. Documentação do Projeto

**Relatórios e Visualizações:**
•	Foram criados relatórios que incluem gráficos de importância das variáveis, curvas ROC e matrizes de confusão. Estas visualizações foram desenvolvidas utilizando matplotlib e seaborn, e ajudam a comunicar os resultados de forma clara e compreensível.

**Documentação Completa do Projeto:**
•	Toda a documentação do projeto foi detalhada, desde a coleta e tratamento dos dados, passando pelo desenvolvimento do modelo, até a fase de implantação. A documentação está organizada em um repositório GitHub, permitindo que outros profissionais reproduzam o trabalho e colaborem em melhorias futuras.

## 8. Referências Utilizadas

•	Artigo Gradio no Medium: https://medium.com/data-hackers/gradio-crie-e-compartilhe-seus-machine-learning-apps-88d8b3c5cca4

•	Dataset do Kaggle: https://www.kaggle.com/datasets/rashadrmammadov/heart-disease-prediction

•	Documentação do Scikit-Learn

•	Projeto Heart Disease disponível no Kaggle em:	 https://www.kaggle.com/code/vikramjaswal/heart-disease-prediction-log-regression

•	Deploy de Hugging Face no Youtube: https://www.youtube.com/watch?v=3y3l_rsbbB8

## 9. Conclusão do Projeto

O objetivo do projeto foi criar uma API interativa para que pudesse ser realizado consultas com tipos distintos de dados de pacientes a fim de prever a possibilidade de ter ou não doença cardíaca.
A entrega foi feita através de um link disponível para acesso em: https://huggingface.co/spaces/nachbars/heart_disease

**• Próximo passos**

1. Para um próximo projeto os dados deverão ser revistos, e possivelmente trabalhar com um número maior de volume para que a assertividade do modelo em generalização abranga maiores características de diferentes tipos de pacientes.
2. Estabelecer novas variáveis alvo, e alocar novos parâmetros nos modelos.
3. Trabalhar com novas interfaces de API a fim de que seja possível a expansão do modelo para diferentes programas e usos, com possíveis sugestões de advertências ao usuário para mudar o estilo de vida, e adotar hábitos mais saudáveis.

## 10. Ferramentas Utilizadas para o Projeto

- ![Hugging Face](https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)
- ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)
- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
- ![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
- ![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

