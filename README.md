# Objetivo
O objetivo deste projeto é desenvolver um modelo de regressão preditiva que permita: 

✅ Treinar um modelo de Machine Learning para prever as vendas de sorvete com base na temperatura do dia.  
✅ Registrar e gerenciar o modelo usando o MLflow.  
✅ Implementar o modelo para previsões em tempo real em um ambiente de cloud computing.  
✅ Criar um pipeline estruturado para treinar e testar o modelo, garantindo reprodutibilidade.

# 
# Passo a Passo

## 1. Criando o dataset
Para criar dados simulados referente ao ano de 2024, contendo data, vendas diárias e temperatura média respeitando as estações do ano no hemisfério sul utilizando as bibliotecas pandas e numpy.   
```
import pandas as pd
import numpy as np

# Criar datas de 2024
dates = pd.date_range(start="2024-01-01", end="2024-12-31")

# Simular vendas e temperatura com sazonalidade
np.random.seed(42)
vendas = np.where(
    dates.month.isin([12, 1, 2]),  # Verão
    np.random.randint(100, 150, len(dates)),
    np.where(
        dates.month.isin([9, 10, 11]),  # Primavera
        np.random.randint(90, 130, len(dates)),
        np.where(
            dates.month.isin([6, 7, 8]),  # Inverno
            np.random.randint(40, 70, len(dates)),
            np.random.randint(60, 90, len(dates))  # Outono
        )
    )
)

# Simular temperatura (valores médios por estação)
temp = np.where(
    dates.month.isin([12, 1, 2]), 28 - (dates.day / 10),
    np.where(
        dates.month.isin([3, 4, 5]), 22 - (dates.month - 3),
        np.where(
            dates.month.isin([6, 7, 8]), 12 + (dates.month - 6),
            18 + (dates.month - 9)  # Primavera
        )
    )
)

# Criar DataFrame
df = pd.DataFrame({
    "Data": dates,
    "Vendas": vendas,
    "Temperatura": np.round(temp, 1),
    "Estação": np.select(
        [dates.month.isin([12, 1, 2]), dates.month.isin([3, 4, 5]), dates.month.isin([6, 7, 8])],
        ["Verão", "Outono", "Inverno"],
        default="Primavera"
    )
})

# Salvar em CSV
df.to_csv("vendas_temperatura_2024.csv", index=False)
```


## 2. Criar o grupo de recursos para o projeto
- Acesse o **[Portal do Azure](https://portal.azure.com/)**;
- Faça _login_ com sua conta **Azure**;
- Após o _login_ acesse as opções **Create a resource** e selecione **Resource group**;
- Insira o nome que desejar e em "**Review & Create**".  
![Captura de tela 2025-03-26 224208](https://github.com/user-attachments/assets/6dd841d2-8579-4cce-9a03-6445ab1e3e05)  
<p align="center">Imagem com o resource group criado</p>  

## 3. Acessar o Machine Learning Studio
- Acesse o **[Portal do Azure AI](https://ml.azure.com/)**;
- Selecione o workspace criado na etapa anterior;
![Captura de tela 2025-03-26 224239](https://github.com/user-attachments/assets/865885a3-c157-4576-b2e4-b08b1146e80c)
<p align="center">Imagem com o workspace do Machine Learning Studio criado</p>  

## 4. Importar o dataset
### Para importar o dataset criado na etapa 1:
- Acesse o menu lateral **Data** -> **Create**: informe o nome do _dataset_,  o tipo "**Tabular**" e clique em **Next**;  
- **Data source**: selecione **From local files**, deixe o **datastore** padrão, clique em **Upload files** e selecione o *dataset* criado na etapa 1;
- **Settings**: verique se as configurações da tabulação estão corretas para o *dataset*;
- **Schema**: para esse experimento remova as colunas "Data" e "Estação", por fim, clique em **Create**.  
![Captura de tela 2025-03-26 224312](https://github.com/user-attachments/assets/eae34756-3a08-40e8-942c-5297dbecda58)  
<p align="center">Imagem com o dataset criado</p>  

## 5. Criar a instância e cluster
### 5.1Compute instance  
- Acesse o menu lateral **Compute** -> **Compute instance** -> **New** para criar uma nova instância de computação;
- **Required setting**: informe o nome da instância e selecione a capacidade da VM;
- **Scheduling**: configure um agendamento para o desligamento automatico da VM caso fique sem uso;
- **Security**: selecione as configurações de segurança, para esse experimento serão utilizada as configurações padrão;
- **Applications**: Adicione aplicativos personalizados que você pode querer usar na sua instância de computação;
- **Tags**: adicione tags para categorizar a instância;
- **Review + Create**: por fim, revise as configurações e cria a instância.
![image](https://github.com/user-attachments/assets/250e1b2a-7239-4de0-9c2c-8f45314b4f39)
<p align="center">Imagem com a instância de computação criada</p>  

### 5.2 Compute cluster
- Acesse o menu lateral **Compute** -> **Compute cluster** -> **New** para criar um novo *cluster* de computação;
- **Virtual machine**: Selecione o tamanho da máquina virtual que você gostaria de usar para seu cluster de computação;
- **Advanced settings**: Configure as definições do cluster de computação para o tamanho da máquina virtual selecionada.
![Captura de tela 2025-03-27 003254](https://github.com/user-attachments/assets/06cd0b9d-c60b-41ca-ac6b-c149fc929d65)
<p align="center">Imagem com o cluster de computação criado</p>  

## 6. Iniciar os treinamentos
### 6.1 Autometed ML
- Acesse o menu lateral **Autometed ML** -> **New Autometed ML**;
- **Traning method**: escolha o método de treinamento que você gostaria de usar;
- **Basic settings**: informe as configurações básicas (nome do job, nome do experimento);
- **Task type & data**: escolha o tipo de tarefa que você gostaria que seu modelo executasse e os dados a serem usados ​​para o treinamento. Nesse caso iremos utilizar o tipo **Regression** e o dataset que foi importado na **Etapa 4**;
- **Task settings**: para esse experimento defina as configurações para a **Target column** como **Vendas**. Clique **View additional configuration settings** e bloqueie todos os modelos com exceção do **XGBoostRegressor**. Para **Limits** defina as opções **Experiment timeout (minutes)** e **Iteration timeout (minutes)** como **15**, marque a opção **Enable early termination** e para as opções **Validade and test**: deixe as opções padrões.
- **Compute**: selecione a opção **Compute cluster** e selecione o cluster que foi criado na **Etapa 5.2**.
![Captura de tela 2025-03-26 225412](https://github.com/user-attachments/assets/40a826d6-e158-4109-9e5e-9b903411b839)
<p align="center">Imagem com o job criado pelo Autometed ML</p>  

### 6.2 Designer
- Acesse o menu lateral **Designer** -> **New pipeline** -> **Create a new pipeline**;
- Estruture o pipeline, exemplo: selecionar o dataset, selecionar colunas, dividir o dataset para treinamento, selecionar o modelo, selecionar o treinamento, pontuação do modelo e avaliar do modelo;
- **Configure & Submit**: após terminar de estruturar o pipeline, submeta o modelo para executar.  
![Captura de tela 2025-03-27 011000](https://github.com/user-attachments/assets/ff352412-5129-4707-97b6-efcb580f3cca)
<p align="center">Imagem com o job criado pelo Autometed ML</p>  


## 7. Resultados
### 7.1 Autometed ML
Para acessar os resultados da execução clique em **Jobs** -> **Nome do job criado na Etapa 6.1** -> **Models + child jobs**. Os modelos estarão ordenados dos melhores para os com piores desempenhos.
![Captura de tela 2025-03-27 011253](https://github.com/user-attachments/assets/bba6f01b-7e56-4a9b-9b54-1704be561198)
<p align="center">Desempenho dos modelos executados pelo Autometed ML</p>  

![Captura de tela 2025-03-26 225627](https://github.com/user-attachments/assets/c51c8049-a8b5-43dd-b3d8-942966a438de)  
<p align="center">Detalhes do modelo VotingEnsemble que obteve o melhor desempenho</p>  

![Captura de tela 2025-03-26 225607](https://github.com/user-attachments/assets/73f2ff81-2bdb-4490-8e11-f5f5b3a65c79)
<p align="center">Detalhes do modelo VotingEnsemble que obteve o melhor desempenho</p>  

## 7. Resultados
### 7.2 Designer pípeline
Para acessar os resultados da execução clique em **Jobs** -> **Nome do job criado na Etapa 6.2**.
- **Score Model** -> **Preview data** -> **Scored dataset**.
![Captura de tela 2025-03-26 224645](https://github.com/user-attachments/assets/ebb85ee4-9632-475b-81be-3bb0c5577d77)
<p align="center">Detalhes do Scored dataset</p>  

- **Evaluate Model** -> **Preview data** -> **Evaluation results**.
![Captura de tela 2025-03-26 225322](https://github.com/user-attachments/assets/9ab94640-32d7-41fb-8104-f196e7bc1736)
<p align="center">Detalhes do Evaluation results</p>  

