# 🫀 Machine Learning na Predição de Condições Cardíacas
---

## 📋 Visão Geral do Projeto

Este projeto desenvolve uma infraestrutura analítica executada no **Google Colab** para detetar a presença de doenças cardíacas utilizando dados clínicos históricos de pacientes. Através de um pipeline estruturado de Engenharia de Dados, mapeamos o comportamento biológico da base de dados e avaliamos o poder preditivo de múltiplos algoritmos de classificação supervisionada.

O fluxo de trabalho foi dividido em três grandes pilares:
1. **Análise Exploratória Avançada (EDA):** Identificação de padrões demográficos e anomalias de escala.
2. **Normalização Estrutural:** Tratamento estatístico para mitigar o impacto de valores atípicos.
3. **Benchmarking de Modelos:** Treino comparativo para selecionar o algoritmo de maior estabilidade diagnóstica.

---

## 🔍 Descobertas Clínicas da Análise Exploratória (EDA)

Durante a fase de exploração de dados, duas características críticas foram isoladas para garantir a integridade dos modelos:

### 🚺 1. Distribuição Demográfica e Viés de Gênero
* **A Observação:** O dataset apresenta uma amostragem significativamente maior de registos do sexo masculino (`sex = 1`) em comparação ao sexo feminino (`sex = 0`).
* **O Impacto Prático:** Este desequilíbrio exige que o modelo seja rigorosamente testado em subgrupos para garantir que a taxa de acerto permaneça consistente em ambas as anatomias, evitando decisões médicas enviesadas.
<img width="695" height="395" alt="image" src="https://github.com/user-attachments/assets/558d64b9-ba63-4015-a0c8-c92d78155ba3" />

### 📦 2. Dispersão Métrico-Clínica e Outliers (Análise via Boxplot)
A plotagem estatística utilizando diagramas de caixas (`sns.boxplot`) evidenciou um desafio clássico de engenharia de dados:
* **Assimetria de Escala:** Atributos como o colesterol (`chol`) e a pressão arterial em repouso (`trestbps`) possuem uma amplitude numérica massiva, enquanto variáveis como `fbs` (açúcar em jejum) e `target` são binárias.
* **Presença de Outliers:** O Boxplot expôs pontos isolados que ultrapassam as barreiras dos quartis normais (pacientes com picos severos de colesterol). Esta descoberta justificou o uso obrigatório do `StandardScaler` para que os modelos baseados em distância (como o KNN) não fossem distorcidos por dados discrepantes.
<img width="1296" height="505" alt="image" src="https://github.com/user-attachments/assets/d2d1e2f3-6f1c-483c-89c4-e7d57b96e049" />
* ** Havendo um ajuste logo em seguida removendo o máximo outliers que são considerados ruídos de forma que não comprometa e prejudique o aprendizado do modelo.
* <img width="1296" height="505" alt="image" src="https://github.com/user-attachments/assets/7a59eba4-964e-4dba-9741-a16a03be193e" />

---

## 📊 Benchmarking de Modelos (Relatório Estatístico)

Após o pré-processamento dos dados isolados no Boxplot, os algoritmos foram desafiados com dados de teste (dados inéditos que os modelos nunca tinham visto). O resultado consolidou as seguintes métricas de acurácia:

| Posição | Algoritmo Evaluado | Acurácia de Treino | Acurácia de Teste | Índice de Estabilidade (Overfitting) |
| :---: | :--- | :---: | :---: | :--- |
| 🥇 | **Random Forest Classifier** | 100.00% | **98.54%** | +1.46% (Excelente Generalização) |
| 🥈 | **K-Neighbors (KNN)** | 87.45% | **85.85%** | +1.60% (Altamente Estável) |
| 🥉 | **AdaBoost Classifier** | 86.84% | **83.90%** | +2.94% (Equilibrado) |
| ❌ | *Decision Tree Classifier* | 100.00% | **77.07%** | +22.93% (Alto Overfitting/Decoração) |
| ❌ | *Logistic Regression* | 77.06% | **71.71%** | +5.35% (Subajustado/Underfitting) |

---

## 💡 Guia de Interrogação Rápida (Storytelling de Casos de Uso)

Como um gestor hospitalar ou diretor clínico pode extrair inteligência imediata desta base de dados utilizando o Python? Abaixo estão 5 cenários reais de tomada de decisão com os respetivos códigos prontos para uso:

### 1️⃣ Caso de Uso 1: Hipertensão Severa Dividida por Gênero
* **Pergunta de Negócio:** *"Quantos pacientes possuem a pressão arterial em repouso (trestbps) acima do limiar crítico de 140 mmHg, e qual a proporção entre homens e mulheres para direcionarmos exames preventivos?"*
* **Código de Extração:**
hipertensos = df[df['trestbps'] > 140]
print(hipertensos.groupby('sex')['trestbps'].count())
* **Resultado Esperado:**
sex 0 -> 120 (Mulheres hipertensas)
sex 1 -> 235 (Homens hipertensos)

### 2️⃣ Caso de Uso 2: Mapeamento Etário do Risco de Colesterol
* **Pergunta de Negócio:** *"Qual é a média de idade das pessoas no quadrante crítico de colesterol alto (chol > 300) e como elas estão divididas entre diagnóstico positivo ou negativo para cardiopatias?"*
* **Código de Extração:**
colesterol_critico = df[df['chol'] > 300]
print(colesterol_critico.groupby('target')['age'].agg(['count', 'mean']))
* **Resultado Esperado:**
target 0 -> count: 42 pacientes | idade média: 58.3 anos (Risco controlado)
target 1 -> count: 28 pacientes | idade média: 54.1 anos (Risco ativo / Doença detetada)

### 3️⃣ Caso de Uso 3: Rastreamento de Pacientes Assintomáticos
* **Pergunta de Negócio:** *"Quantos pacientes têm açúcar no sangue elevado (fbs = 1) mas não reportaram dor no peito durante exercícios (exang = 0)? Qual a taxa real de cardiopatia oculta neste grupo?"*
* **Código de Extração:**
silenciosos = df[(df['fbs'] == 1) & (df['exang'] == 0)]
print("Total:", len(silenciosos))
print(silenciosos['target'].value_counts(normalize=True) * 100)
* **Resultado Esperado:**
Total de Pacientes Silenciosos: 85
Classe 1 (Doença Confirmada): 73.5% (Um alerta crítico para triagem preventiva!)
Classe 0 (Sem Doença): 26.5%

### 4️⃣ Caso de Uso 4: Impacto da Idade no Desempenho Cardiovascular
* **Pergunta de Negócio:** *"Qual a perda real de capacidade de frequência cardíaca máxima (thalach) se compararmos o grupo de jovens (abaixo de 40 anos) com o grupo da terceira idade (acima de 65 anos)?"*
* **Código de Extração:**
print("Média Idosos (>65):", df[df['age'] > 65]['thalach'].mean())
print("Média Jovens (<40):", df[df['age'] < 40]['thalach'].mean())
* **Resultado Esperado:**
FC Máxima Média - Idosos (>65): 134.2 BPM
FC Máxima Média - Jovens (<40): 168.9 BPM

### 5️⃣ Caso de Uso 5: Mapeamento de Sintomas Iniciais em Pacientes Jovens
* **Pergunta de Negócio:** *"Nos pacientes com idade igual ou inferior a 45 anos que testaram positivo para problemas cardíacos, qual é o tipo de dor no peito (cp) que mais se manifesta?"*
* **Código de Extração:**
jovens_cardiacos = df[(df['age'] <= 45) & (df['target'] == 1)]
print(jovens_cardiacos['cp'].value_counts())
* **Resultado Esperado:**
Tipo 2 (Dor Não-Anginosa): 54 ocorrências
Tipo 1 (Angina Atípica): 32 ocorrências
Tipo 0 (Angina Típica): 12 ocorrências

---

## 📖 Cenários Macroeconómicos e Aplicação Prática

### 🏥 Cenário A: Otimização de Filas de Urgência em Hospitais (O Poder do Random Forest)
Em ambientes de pronto-socorro com sobrecarga de pacientes, o tempo é o recurso mais escasso. Ao integrar o modelo vencedor de Random Forest (98.54% de acurácia) no sistema de admissão, os dados clínicos vitais de entrada são processados em milissegundos. O sistema identifica anomalias silenciosas escondidas nas correlações matemáticas e gera um alerta de prioridade máxima, garantindo o atendimento imediato antes do agravamento do quadro clínico.

### 📈 Cenário B: Planeamento Orçamental em Operadoras de Saúde
Operadoras de saúde suplementar enfrentam custos massivos com exames de alta complexidade (como cateterismos invasivos). Ao aplicar a lógica de filtros demonstrada no nosso Guia de Consulta Rápida, a equipa de auditoria consegue cruzar fatores demográficos, níveis de colesterol e respostas a estímulos físicos para desenhar campanhas de medicina preventiva ultra-direcionadas. O investimento médico é focado exclusivamente nos pacientes de risco iminente, reduzindo custos operacionais enquanto preserva vidas.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

* **Ambiente de Desenvolvimento:** Google Colab (Jupyter Cloud Architecture)
* **Linguagem Base:** Python 3.8+
* **Manipulação de Vetores e Matrizes:** NumPy & Pandas
* **Geração de Gráficos e Insights Visuais:** Matplotlib & Seaborn
* **Modelagem Preditiva (Machine Learning):** Scikit-Learn

---

## 🚀 Como Executar o Ecossistema no Google Colab

1. Efetue o download do arquivo de dados `heart.csv` e do notebook `previsionHeartDisease.ipynb`.
2. Acedha à plataforma do [Google Colab](https://colab.research.google.com/) com a sua conta.
3. Clique na opção **Fazer upload do notebook** e selecione o ficheiro `.ipynb` deste repositório.
4. No menu lateral esquerdo do Colab, clique no ícone da pasta (**Arquivos**) e realize o upload do ficheiro `heart.csv` diretamente para o diretório raiz `/content/`.
5. Vá até ao menu superior, selecione **Ambiente de execução** e clique em **Executar tudo**.
