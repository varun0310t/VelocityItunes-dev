import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity  # Add this import
import gc  # Add garbage collector import
from tqdm import tqdm  # Add for progress bar

"""Importing Dataset"""
dataset_path = './data.csv'

df = pd.read_csv(dataset_path)

df_copy = df.copy()

"""DATA PREPROCESSING"""

df.head()

df.info()

df.shape

df.describe(include = 'all')

plt.figure(figsize=(8, 6))
sns.histplot(df['tempo'], kde=True, color='blue')
plt.title('Distribution of Tempo')
plt.xlabel('Tempo')
plt.ylabel('Frequency')
plt.show()

df.isnull().sum()

df.duplicated().sum()

x = df[['tempo', 'valence', 'instrumentalness', 'energy', 'mode', 
        'acousticness', 'speechiness', 'key', 'loudness', 'artists']]
y = df['name']

x = pd.DataFrame(x)
x.head()

y = pd.DataFrame(y)
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train.head()

x_train.shape

x_test.shape

y_train.head()

y_train.shape

y_test.shape

#Scaling

scaler = MinMaxScaler()
numeric_features = ['tempo', 'valence', 'energy', 'mode', 'acousticness', 
                   'danceability', 'instrumentalness', 'liveness', 'loudness']
df_scaled = scaler.fit_transform(df[numeric_features])
df_scaled = pd.DataFrame(df_scaled, columns=[f'{col}_scaled' for col in numeric_features])

df_scaled

c = df_scaled

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(c)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
clusters = kmeans.fit_predict(c)
print(clusters)

df['Clusters'] = clusters
df.tail()

df_sample = df.sample(n=100)
df_sample = pd.DataFrame(df_sample)
df_sample

cluster_centroids = c.groupby(clusters).mean()
cluster_centroids = pd.DataFrame(cluster_centroids)
cluster_centroids

plt.figure(figsize=(10, 6))
sns.scatterplot(data=c, x='tempo_scaled', y='valence_scaled')
plt.title('Clusters of Songs')
plt.xlabel('Tempo')
plt.ylabel('Valence')
plt.show()
sns.scatterplot(data=cluster_centroids, x='tempo_scaled', y='valence_scaled', s=100, hue=cluster_centroids.index, palette='deep', edgecolor='black')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=c, x='tempo_scaled', y='valence_scaled', hue=df['Clusters'], palette='deep')
sns.scatterplot(data=cluster_centroids, x='tempo_scaled', y='valence_scaled', s=100, color='red', edgecolor='black', label='Centroids')
plt.title('Clusters of Songs')
plt.xlabel('Tempo')
plt.ylabel('Valence')
plt.legend()
plt.show()

x_train.head()

x['Cluster'] = kmeans.labels_
x.head(5)

x_train['Cluster'] = x['Cluster']
x_train.head(5)

x_test.head(5)

y_train.head(5)

x_train_encoded = x_train.drop(['artists'], axis=1)  # Remove artists column instead of encoding it

columns_to_scale = ['tempo', 'key', 'loudness', 'valence',
                    'instrumentalness', 'energy', 'mode', 'acousticness',
                    'speechiness']

scaler = MinMaxScaler()
x_train_encoded[columns_to_scale] = scaler.fit_transform(x_train_encoded[columns_to_scale])
print(x_train_encoded[columns_to_scale].head())


x_train_encoded.head(5)

print(list(x_train_encoded.columns))

features = list(x_train_encoded.columns)

from sklearn.decomposition import PCA

# Calculate number of components based on available features
n_components = min(len(features), 10)  # Use minimum of feature count or 10
pca = PCA(n_components=n_components)
x_train_reduced = pca.fit_transform(x_train_encoded)

x_train_reduced = pd.DataFrame(x_train_reduced, columns=[f'PC{i+1}' for i in range(n_components)])

def calculate_similarity_in_batches(features_matrix, batch_size=200):  # Increased batch size
    n_samples = features_matrix.shape[0]
    filename = 'temp_similarity.npy'
    
    # Create memory-mapped file
    similarity_matrix = np.memmap(filename, dtype='float32', mode='w+', shape=(n_samples, n_samples))
    
    try:
        # Add progress bar
        pbar = tqdm(total=n_samples, desc="Processing rows")
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch_i = features_matrix[i:end_i]
            
            for j in range(0, n_samples, batch_size):
                end_j = min(j + batch_size, n_samples)
                batch_j = features_matrix[j:end_j]
                
                similarity_batch = cosine_similarity(batch_i, batch_j)
                # Fix the syntax error here - remove keyword argument
                similarity_matrix[i:end_i, j:end_j] = similarity_batch
                
                similarity_matrix.flush()
                gc.collect()
            
            pbar.update(end_i - i)
        
        pbar.close()
        return np.array(similarity_matrix)
    
    finally:
        # Ensure proper cleanup
        del similarity_matrix
        gc.collect()
        
        # Wait a bit before trying to remove file
        import time
        time.sleep(1)
        
        try:
            os.remove(filename)
        except:
            print(f"Note: Could not remove {filename}. Please delete it manually.")

print("Calculating similarity matrix for full dataset...")
similarity_matrix = calculate_similarity_in_batches(x_train_encoded[features].values, batch_size=450)

def recommend_songs(track_name, x_train_encoded, y_train, similarity_matrix, top_n=5):
    try:
        train_data = pd.concat([x_train_encoded.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        song_index = train_data[train_data['name'] == track_name].index[0]
        
        similarity_scores = list(enumerate(similarity_matrix[song_index]))
        sorted_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_songs = sorted_songs[1:top_n+1]
        
        recommendations = [train_data.iloc[i[0]]['name'] for i in top_songs]
        return recommendations
    except IndexError:
        return "Track not found in the training dataset."

# Use full dataset for recommendations
query_song = y_train.iloc[2]['name']
recommendations = recommend_songs(query_song, x_train_encoded, y_train, similarity_matrix)

# Output recommendations
print(f"Recommended songs for '{query_song}': {recommendations}")
