
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
"""Importing Dataset"""
dataset_path = './spotify_songs.csv'

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

plt.figure(figsize=(8, 6))
sns.histplot(df['playlist_genre'], kde=True, color='blue')
plt.title('Distribution of playlist genre')
plt.xlabel('Playlist Genre')
plt.ylabel('Frequency')
plt.show()

df.isnull().sum()

df['lyrics'] = df['lyrics'].fillna("No lyrics available")

df['language'] = df['language'].fillna(df['language'].mode()[0])

df.isnull().sum()

df.duplicated().sum()

df['playlist_genre'].unique()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_playlist_genre = encoder.fit_transform(df[['playlist_genre']])

encoded_df = pd.DataFrame(encoded_playlist_genre, columns=encoder.get_feature_names_out())
encoded_df.head()

#Splitting the Dataset

x = df[['tempo','valence','instrumentalness','energy','mode','acousticness','speechiness','key','loudness','playlist_genre','playlist_subgenre','playlist_name','track_album_name','track_artist','lyrics']]
y = df['track_name']

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['tempo', 'valence', 'energy','mode']])
df_scaled = pd.DataFrame(df_scaled, columns=['tempo_scaled', 'valence_scaled', 'energy_scaled','mode_scaled'])

df_scaled

c = pd.concat([df_scaled,encoded_df], axis=1)
c.head()

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

x_train_encoded =x_train

categorical_columns = ['playlist_genre', 'playlist_subgenre', 'playlist_name', 'track_album_name', 'track_artist']
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(x_train_encoded[categorical_columns])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
x_train_encoded = x_train_encoded.drop(categorical_columns, axis=1)
x_train_encoded= pd.concat([x_train_encoded.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

x_train_encoded.head()

df_lyrics= x_train['lyrics']
df_lyrics

df_lyrics.str.lower().replace(r'^\w\s', '', regex= True).replace(r'\n', '', regex = True).replace(r'\r', '', regex = True)

import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt_tab')

stemmer = PorterStemmer()
def token(txt):
  token= nltk.word_tokenize(txt)
  a = [stemmer.stem(w) for w in token]
  return " ".join(a)

df_lyrics.apply(lambda x: token(x))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

tfid= TfidfVectorizer(analyzer='word')
lyrics_matrix = tfid.fit_transform(df_lyrics)

similar = cosine_similarity(lyrics_matrix)

similar[0]

x_train_encoded['lyrics']= similar
x_train_encoded.head(5)

x_train_encoded['lyrics'].head()

x_train_encoded.shape

print(x_train_encoded.columns)

columns_to_scale = ['tempo','key', 'loudness','valence','instrumentalness','energy','mode','acousticness','speechiness']

scaler = MinMaxScaler()
x_train_encoded[columns_to_scale] = scaler.fit_transform(x_train_encoded[columns_to_scale])
print(x_train_encoded[columns_to_scale].head())


x_train_encoded.head(5)

print(list(x_train_encoded.columns))

features = list(x_train_encoded.columns)

from sklearn.decomposition import PCA

pca = PCA(n_components=50)
x_train_reduced = pca.fit_transform(x_train_encoded)

x_train_reduced = pd.DataFrame(x_train_reduced, columns=[f'PC{i+1}' for i in range(50)])




similarity_matrix = cosine_similarity(x_train_encoded[features])

#uncomment if u wanna save the model 

#x_test.to_csv("x_test.csv",index=False)
#y_test.to_csv("y_test.csv",index=False)
#x_train_encoded.to_csv("x_train_encoded.csv",index=False)
#y_train.to_csv("y_train.csv",index=False)
#joblib.dump(pca, 'pca_model.joblib')
#joblib.dump(scaler, 'scaler_model.joblib')
#joblib.dump(similarity_matrix, 'similarity_matrix.joblib')

def recommend_songs(track_name, x_train_encoded, y_train, similarity_matrix, top_n=5):
    try:
        train_data = pd.concat([x_train_encoded.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)

        song_index = train_data[train_data['track_name'] == track_name].index[0]

        similarity_scores = list(enumerate(similarity_matrix[song_index]))

        sorted_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        top_songs = sorted_songs[1:top_n+1]

        recommendations = [train_data.iloc[i[0]]['track_name'] for i in top_songs]
        return recommendations
    except IndexError:
        return "Track not found in the training dataset."
query_song = y_train.iloc[2]['track_name']  
recommendations = recommend_songs(query_song, x_train_encoded, y_train, similarity_matrix)

# Output recommendations
print(f"Recommended songs for '{query_song}': {recommendations}")
