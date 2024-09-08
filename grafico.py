import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

status = pd.read_csv('status_clustered.csv')

sns.set(style='whitegrid')

plt.figure(figsize=(16,8))

#kmeans grafico

plt.subplot(1,2,1)
sns.scatterplot(data=status, x='win',y='loss',hue='kmeans_cluster',palette = 'viridis',s=100, edgecolor='w')
plt.title('KMeans')
plt.xlabel('Vitorias')
plt.ylabel('derrotas')
plt.legend(title='cluster',loc='best')

#Agglomerative Clustering grafico
plt.subplot(1,2,2)
sns.scatterplot(data=status, x='win',y='loss',hue='agg_cluster',palette = 'viridis',s=100, edgecolor='w')
plt.title('Agglomerative')
plt.xlabel('Vitorias')
plt.ylabel('derrotas')
plt.legend(title='cluster',loc='best')

plt.tight_layout
plt.show()
