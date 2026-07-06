# Predição de Doença Cardiovascular com Aprendizado de Máquina

Comparação entre três algoritmos de aprendizado supervisionado — **Regressão Logística**, **Random Forest** e **SVM** — aplicados à predição de doença cardiovascular, com redução de dimensionalidade (PCA), validação cruzada estratificada e otimização de hiperparâmetros via GridSearchCV.

## Escopo do projeto

- [x] **Dataset:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (Kaggle, ~70.000 registros)
- [x] **Pelo menos 3 algoritmos supervisionados:** Regressão Logística, Random Forest e SVM
- [x] **Validação cruzada + busca de hiperparâmetros:** `StratifiedKFold` (k=5) + `GridSearchCV`
- [x] **Redução de dimensionalidade:** PCA (seleção automática de componentes para ≥ 95% de variância explicada)
- [x] **Comparação entre modelos:** notebook dedicado consolidando os resultados dos 3 algoritmos
- [x] **Métricas adequadas ao problema:** acurácia, precisão, recall, F1-score, ROC-AUC, matriz de confusão e curva ROC

## Estrutura do repositório
├── 1 - Regressao Logistica.ipynb      # Pipeline completo + modelo de Regressão Logística
├── 2 - Random Forest.ipynb            # Pipeline completo + modelo de Random Forest
├── 3 - SVM.ipynb                      # Pipeline completo + modelo de SVM
├── 4 - Codigo Final Comparacao.ipynb  # Consolidação e comparação dos 3 modelos
└── README.md

Cada um dos três notebooks de algoritmo é **autocontido** e segue exatamente o mesmo pipeline de dados, garantindo que os três modelos sejam avaliados sob condições idênticas.

## Dataset

- **Fonte:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (sulianova, Kaggle)
- **Tamanho:** ~70.000 registros, 11 atributos de entrada + 1 variável-alvo (`cardio`)
- **Atributos:** idade, gênero, altura, peso, pressão arterial sistólica/diastólica, colesterol, glicose, tabagismo, consumo de álcool, atividade física

## Pipeline de pré-processamento

Aplicado de forma idêntica nos três notebooks de algoritmo:

1. Conversão da idade de dias para anos
2. Engenharia de atributos: **IMC**, **pressão de pulso**, **pressão arterial média (MAP)**, **razão sistólica/diastólica**, **indicador de obesidade** e **escore de risco composto**
3. Remoção de outliers fisiologicamente inválidos (ex.: pressão sistólica menor que a diastólica, alturas/pesos fora de faixas plausíveis)
4. Padronização das variáveis numéricas com `StandardScaler`
5. Redução de dimensionalidade com `PCA` (≥ 95% de variância explicada)
6. Divisão treino/teste fixa (`random_state=42`), idêntica entre os três notebooks

## Modelos e otimização

| Etapa | Ferramenta |
|---|---|
| Validação cruzada | `StratifiedKFold(n_splits=5)` |
| Busca de hiperparâmetros | `GridSearchCV` |
| Algoritmos | `LogisticRegression`, `RandomForestClassifier`, `SVC` |

> **Nota sobre o SVM:** devido ao custo computacional em ambiente Colab, o `GridSearchCV` foi executado sobre uma subamostra estratificada de ~6.000 exemplos. O modelo final foi retreinado sobre o conjunto de treino completo, já projetado pelo PCA, garantindo comparabilidade com os demais algoritmos.

## Resultados

| Modelo | Acurácia | Precisão | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0,7326 | 0,7502 | 0,6891 | 0,7183 | 0,8001 |
| Regressão Logística | 0,7281 | 0,7532 | 0,6699 | 0,7091 | 0,7918 |
| SVM | 0,7296 | 0,7750 | 0,6388 | 0,7004 | 0,7916 |

*Métricas calculadas sobre o mesmo conjunto de teste (13.723 registros) para os três modelos.*

O **Random Forest** apresentou o melhor equilíbrio geral entre as métricas, com o maior recall e o maior ROC-AUC. O **SVM** obteve a maior precisão, às custas de um recall mais baixo. A **Regressão Logística** teve desempenho intermediário e competitivo em quase todas as métricas, indicando que a fronteira de decisão entre pacientes com e sem doença cardiovascular é predominantemente próxima de linear neste dataset.

A comparação completa (tabela, gráfico de barras e heatmap) está disponível no notebook `4 - Codigo Final Comparacao.ipynb`.

## Como executar

1. Abra qualquer um dos notebooks no [Google Colab](https://colab.research.google.com/)
2. Execute as células em ordem — o dataset é baixado automaticamente via `kagglehub` (ou pode ser feito upload manual do CSV)
3. Para reproduzir a comparação final, execute primeiro os três notebooks de algoritmo e depois o notebook `4 - Codigo Final Comparacao.ipynb`

## Tecnologias

- Python
- scikit-learn (`GridSearchCV`, `StratifiedKFold`, `PCA`, `StandardScaler`)
- pandas / numpy
- matplotlib / seaborn
- kagglehub

## Autor

**WesdleyR**
