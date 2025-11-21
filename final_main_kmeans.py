#----------IMPORTANDO BIBLIOTECAS----------#
# bibliotecas para plotagem de gráficos,   #
# fazer o processo de clusterização, tra-  #
# tamento de dados, tabelas, entre outros. #
#------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import warnings

# Configurações; deixando o output mais limpo
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
pd.set_option('display.max_columns', None)

#------------NOTAS IMPORTANTES!------------#
# O relatório final será exibido em uma    #
# pasta chamada 'resultados_projeto', con- #
# tendo gráficos e tabelas informativos.   #
#------------------------------------------#

OUTPUT_DIR = "resultados_projeto" # Criando a pasta onde os resultados serão encontrados
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Mensagem no terminal para indicar o início do script
print(f"--- Iniciando Script de Análise ---")
print(f"Os resultados serão salvos na pasta: {OUTPUT_DIR}/")

# --- PASSO 1. Carregando os dados do datset --- #
df_raw = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# --- PASSO 2. Pré-processamento dos dados, separando a pressão arterial em duas colunas --- #
print("Processando dados...") # Retorno visual do andamento no terminal
df_processed = df_raw.copy()

# 2.1 Convertendo Pressão Arterial (String "120/80" -> Int 120 e Int 80)
df_processed[['Systolic Pressure', 'Diastolic Pressure']] = df_processed['Blood Pressure'].str.split('/', expand=True).astype(int)
df_processed = df_processed.drop(['Blood Pressure', 'Person ID'], axis=1)

# 2.2 One-Hot Encoding (Transformando Gênero, Profissão, etc. em 0 e 1)
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False)
final_features = df_processed.columns

# --- PASSO 3. Normalização dos dados, para que todos estejam em formato numérico e na mesma escala --- #
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_processed)

# --- PASSO 4. Execução do K-Means ---
print("Executando K-Means...") # Retorno visual do andamento no terminal
K_IDEAL = 4
kmeans_final = KMeans(n_clusters=K_IDEAL, init='k-means++', n_init=10, max_iter=300, random_state=42)
clusters = kmeans_final.fit_predict(data_scaled)
df_raw['Cluster'] = clusters
df_processed['Cluster'] = clusters

# --- PASSO 5. Gerando os gráficos ---
print("Gerando e salvando gráficos...") # Retorno visual do andamento no terminal

# Gráfico 1:  Método Elbow - para comprovar o número de clusters escolhido
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, n_init=10, random_state=42).fit(data_scaled)
    wcss.append(km.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/1_metodo_cotovelo.png")
plt.close()

# Gráfico 2: Radar Chart - é um gráfico estilo "teia de aranha" para visualizar os dados
centroids_scaled = kmeans_final.cluster_centers_
features_to_plot = [
    'Age', 
    'Sleep Duration', 
    'Quality of Sleep', 
    'Physical Activity Level', 
    'Stress Level', 
    'Heart Rate', 
    'Daily Steps',
    'Systolic Pressure',
    'Diastolic Pressure'
]

# Mapear índices
indices = [list(final_features).index(col) for col in features_to_plot]
num_vars = len(features_to_plot)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
colors = ['red', 'green', 'blue', 'orange']
for i, (centroid, color) in enumerate(zip(centroids_scaled, colors)):
    values = centroid[indices].tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, label=f"Cluster {i}")
    ax.fill(angles, values, color=color, alpha=0.15)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features_to_plot)
plt.title('Perfil dos Clusters (Normalizado)')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig(f"{OUTPUT_DIR}/2_radar_perfis.png")
plt.close()

# Gráfico 3: Categorias - disturbios do sono e IMC
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(ax=axes[0], data=df_raw, x='Cluster', hue='Sleep Disorder', palette='viridis')
axes[0].set_title('Distúrbios do Sono por Cluster')
sns.countplot(ax=axes[1], data=df_raw, x='Cluster', hue='BMI Category', palette='plasma')
axes[1].set_title('IMC por Cluster')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_distribuicao_categorias.png")
plt.close()

# Gráfico 4: Gênero e idade entre os clusters
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(ax=axes[0], data=df_raw, x='Cluster', hue='Gender', palette='pastel')
axes[0].set_title('Distribuição de Gênero por Cluster')
sns.barplot(ax=axes[1], data=df_raw, x='Cluster', y='Age', palette='viridis', hue='Cluster', legend=False)
axes[1].set_title('Média de Idade por Cluster')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_genero_idade.png")
plt.close()


# Gráfico 5: Distribuição de profissão
plt.figure(figsize=(12, 8))
# Usamos y='Occupation' para fazer barras horizontais (melhor leitura)
sns.countplot(data=df_raw, y='Occupation', hue='Cluster', palette='viridis')
plt.title('Profissões por Cluster')
plt.xlabel('Quantidade')
plt.ylabel('Profissão')
plt.legend(title='Cluster', loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/5_profissoes_cluster.png")
plt.close()


# Gráfico 6: PCA - visualizar os clusters encontrados em um gráfico
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
plt.figure(figsize=(10, 7))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)
plt.scatter(pca.transform(kmeans_final.cluster_centers_)[:, 0], 
            pca.transform(kmeans_final.cluster_centers_)[:, 1], 
            c='red', s=200, marker='X', label='Centroides')
plt.title('Visualização PCA 2D')
plt.savefig(f"{OUTPUT_DIR}/6_visualizacao_pca.png")
plt.close()

# --- PASSO 6. GERAÇÃO DO RELATÓRIO MARKDOWN --- #
print("Escrevendo relatório Markdown...") # Retorno visual do andamento no terminal

numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                'Stress Level', 'Heart Rate', 'Daily Steps']

df_temp = df_raw.copy()
df_temp[['Systolic Pressure', 'Diastolic Pressure']] = df_processed[['Systolic Pressure', 'Diastolic Pressure']]
numeric_cols += ['Systolic Pressure', 'Diastolic Pressure']

medias = df_temp.groupby('Cluster')[numeric_cols].mean().round(2)
contagem = df_raw['Cluster'].value_counts().sort_index()

md_content = f"""
# Relatório Detalhado: Clustering de Sono e Estilo de Vida

## 1. Visão Geral dos Grupos
Total de pessoas analisadas: {len(df_raw)}

### Quantidade por Cluster
{contagem.to_markdown()}

### Estatísticas Médias (Vital + Estilo de Vida)
{medias.to_markdown()}

---

## 2. Análise Demográfica e Social

### Quem são essas pessoas?

Abaixo vemos a divisão entre Homens e Mulheres em cada grupo, bem com a média de faixa etária em cada cluster.

![Idade](4_genero_idade.png)

### O que elas fazem? (Profissão)

Distribuição das ocupações profissionais dentro de cada cluster.

![Profissão](5_profissoes_cluster.png)

---

## 3. Análise de Saúde e Sono

### Perfil Geral (Radar Chart)

Comparativo visual das variáveis numéricas normalizadas.

![Radar](2_radar_perfis.png)

### Riscos de Saúde (Distúrbios e IMC)

Relação entre peso e distúrbios do sono.

![Saúde](3_distribuicao_categorias.png)

### Separação Matemática (PCA)

![PCA](6_visualizacao_pca.png)

---
## 3. Metodologia
- **Algoritmo:** K-Means Clustering
- **K Ideal:** {K_IDEAL} (definido pelo Método do Cotovelo - confira saída nos arquivos)
- **Pré-processamento:** Padronização Z-Score e One-Hot Encoding para variáveis categóricas.
*Gerado em {pd.Timestamp.now().strftime('%d/%m/%Y')}*
"""

with open(f"{OUTPUT_DIR}/relatorio_analise.md", "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"--- Sucesso! ---")
print(f"Verifique a pasta '{OUTPUT_DIR}' para ver o relatório e as imagens.")