import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

#Coleta de dados
db = pd.read_csv('matches.csv')

print(db.head())
print(db.info())


# Contar o número de vitórias e derrotas por time
db['win'] = db['result'].apply(lambda x: 1 if x == 'W' else 0)
db['loss'] = db['result'].apply(lambda x: 1 if x == 'L' else 0)

status = db.groupby('team').agg({
    'win': 'sum',
    'loss': 'sum'
}).reset_index()

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(status[['win', 'loss']])


# KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
status['kmeans_cluster'] = kmeans.fit_predict(X)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
status['agg_cluster'] = agg_clustering.fit_predict(X)


# KMeans
silhouette_kmeans = silhouette_score(X, status['kmeans_cluster'])
calinski_kmeans = calinski_harabasz_score(X, status['kmeans_cluster'])

# Agglomerative Clustering
silhouette_agg = silhouette_score(X, status['agg_cluster'])
calinski_agg = calinski_harabasz_score(X, status['agg_cluster'])

print("KMeans Metricas:")
print(f"Qualidade kmeans: {silhouette_kmeans:.4f}")
print(f"Qualidade por Calinski: {calinski_kmeans:.4f}")

print("\nAgglomerative Clustering Metrics:")
print(f"Qualide do Agglomerative: {silhouette_agg:.4f}")
print(f"Qualidade por Calinski: {calinski_agg:.4f}")

# 5. Escolher o melhor algoritmo
if silhouette_kmeans > silhouette_agg and calinski_kmeans > calinski_agg:
    melhor = 'KMeans'
else:
    melhor = 'Agglomerative Clustering'

print(f"\nMelhor: {melhor}")

# Reaplicar o melhor algoritmo
if melhor == 'KMeans':
    final_clusters = status['kmeans_cluster']
else:
    final_clusters = status['agg_cluster']

status['final_cluster'] = final_clusters


# Tratamento de dados - Filtrar apenas as colunas numéricas para agregação
colunas = ['win', 'loss']
cluster_stats = status.groupby('final_cluster')[colunas].agg(['mean', 'std', 'min', 'max'])
print("\nCluster Statistics:")
print(cluster_stats)

# 7. Identificar o time que mais venceu e o que mais perdeu
maisVenceu = status.loc[status['win'].idxmax()]
maisDerrotado = status.loc[status['loss'].idxmax()]

print("\nTime que mais venceu:")
print(maisVenceu)

print("\nTime que mais perdeu:")
print(maisDerrotado)

# Salvar o DataFrame com clusters para referência
status.to_csv('status_clustered.csv', index=False)
