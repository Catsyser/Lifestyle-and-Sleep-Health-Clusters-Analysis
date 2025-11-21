# Lifestyle-and-Sleep-Heath-Clusters-Analysis

## 🌙 Análise de Qualidade do Sono e Estilo de Vida (Clustering)
  
  Este projeto aplica técnicas de **Machine Learning Não Supervisionado** para identificar padrões comportamentais e de saúde em dados de qualidade do sono e estilo de vida. O objetivo principal foi utilizar o algoritmo **K-Means** para segmentar indivíduos em grupos (clusters) baseados em seus hábitos, permitindo inferir perfis de risco e estilos de vida sem a necessidade de rótulos prévios.

## 🎯 Objetivos do Projeto

1. Realizar a limpeza e pré-processamento de dados brutos de saúde;
2. Aplicar Engenharia de Atributos (ex: tratamento de pressão arterial e One-Hot Encoding);
3. Normalizar dados utilizando StandardScaler;
4. Encontrar o número ideal de grupos com o Método do Cotovelo (Elbow Method);
5. Segmentar os dados utilizando o algoritmo K-Means.Visualizar os resultados através de PCA (Redução de Dimensionalidade) e Gráficos de Radar
6. Gerar relatórios automáticos em Markdown.

## 📊 Dataset

  Os dados utilizados foram obtidos no Kaggle: SITE. O conjunto de dados abrange variáveis como: duração e qualidade do sono, nível de atividade física e passos diários, nível de estresse, categoria de IMC e pressão arterial e distúrbios do sono (insônia, apneia).
  
## 🛠️ Tecnologias Utilizadas

1. Python (Linguagem principal);
2. Pandas & Numpy (Manipulação de dados);
3. Scikit-learn (K-Means, PCA, StandardScaler);
4. Matplotlib & Seaborn (Visualização de dados)Tabulate (Formatação de tabelas para relatórios).

## 🚀 Como Executar

1. Clone este repositório:
```bash
git clone [https://github.com/SEU_USUARIO/analise-sono-clustering.git](https://github.com/SEU_USUARIO/analise-sono-clustering.git)
```
3. Instale as dependências necessárias:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tabulate
```
4. Execute o script principal:
```bash
python Analise_Sono_KMeans_Export.py
```
5. Verifique a pasta resultados_projeto/ gerada, que conterá:

5.1 relatorio_analise.md: Relatório completo com os insights.

5.2 Imagens (.png) com os gráficos gerados.
  
## 📈 Resultados e Insights

O algoritmo identificou **4 perfis principais** (Clusters) nesta população:

| Cluster | Perfil Sugerido | Características Marcantes |
| :--- | :--- | :--- |
| **0** | **Os Saudáveis** | Baixo estresse, IMC normal, ótima qualidade de sono. |
| **1** | **Ativos com Risco** | Alta atividade física, mas presença marcante de Apneia do Sono. |
| **2** | **Jovens Normais** | Grupo majoritário, jovens com bons indicadores gerais. |
| **3** | **Grupo de Alerta** | Sedentários, alto estresse, sobrepeso e forte presença de Insônia. |

### Visualizações Geradas

**Comparativo de Perfis (Radar Chart):**
Permite visualizar rapidamente as forças e fraquezas de cada cluster.

** Gráfico de Colunas:**

Garante uma visualização dos disturbios do sono e IMC.

**Análise PCA:**
Visualização da separação matemática dos grupos em 2D.

**Tabela de Cluster x Categoria:**

Mostra os valores médios de cada cluster, em cada uma das categorias analisadas.

---

Desenvolvido para fins de estudo em Data Science e Machine Learning, para a disciplina de Inteligência Artificial na Universidade Federal do ABC (UFABC)
