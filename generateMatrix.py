import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity  # Add this import
from sklearn.preprocessing import StandardScaler
import gc  # Add garbage collector import
from tqdm import tqdm  # Add for progress bar
from sklearn.cluster import KMeans
import scipy.sparse as sp  # Add sparse matrix import
from joblib import Parallel, delayed  # Add for parallel processing

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

# Use scaled features directly
x_train_scaled = df_scaled
x_train_encoded = x_train_scaled

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(c)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
clusters = kmeans.fit_predict(x_train_scaled)
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

# Add size tracking after data loading
print("\nData size tracking:")
print(f"Original dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Before similarity matrix calculation
print("\nBefore similarity matrix:")
print(f"x_train_scaled shape: {x_train_scaled.shape}")
print(f"Memory usage: {x_train_scaled.memory_usage().sum() / 1024**2:.2f} MB")

def process_batch(i, j, batch_i, batch_j, threshold, n_samples):
    """Process a single batch of similarity calculations"""
    similarity_batch = cosine_similarity(batch_i, batch_j)
    significant_similarities = np.where(similarity_batch > threshold)
    batch_rows = significant_similarities[0] + i
    batch_cols = significant_similarities[1] + j
    batch_data = similarity_batch[significant_similarities].astype(np.float16)
    
    # Filter diagonal and lower triangle entries
    valid_indices = batch_cols >= batch_rows
    return (batch_rows[valid_indices], 
            batch_cols[valid_indices], 
            batch_data[valid_indices])

def calculate_similarity_in_batches(features_matrix, batch_size=1000, n_jobs=4):
    n_samples = features_matrix.shape[0]
    filename = 'similarity_matrix.npy'
    
    # Calculate size for upper triangular matrix (excluding diagonal)
    triu_elements = (n_samples * (n_samples - 1)) // 2
    matrix_size_gb = (triu_elements * 2) / (1024**3)
    
    print(f"\nSimilarity matrix details:")
    print(f"Number of samples: {n_samples}")
    print(f"Upper triangular elements: {triu_elements}")
    print(f"Expected size: {matrix_size_gb:.2f} GB")
    
    proceed = input("Continue with matrix calculation? (y/n): ")
    if proceed.lower() != 'y':
        return None
    
    try:
        # Create memory-mapped file for upper triangle
        similarity_matrix = np.memmap(filename, dtype='float16', mode='w+', 
                                    shape=(triu_elements,))
        
        idx = 0  # Global index for storing in 1D array
        total_rows = (n_samples + batch_size - 1) // batch_size
        
        with tqdm(total=total_rows, desc="Processing rows") as pbar:
            for i in range(0, n_samples, batch_size):
                end_i = min(i + batch_size, n_samples)
                rows = end_i - i
                batch_i = features_matrix[i:end_i]
                
                for j in range(i + 1, n_samples, batch_size):
                    end_j = min(j + batch_size, n_samples)
                    cols = end_j - j
                    batch_j = features_matrix[j:end_j]
                    
                    # Calculate similarities
                    sim_batch = cosine_similarity(batch_i, batch_j).astype(np.float16)
                    
                    # Calculate exact number of elements to store
                    elements_to_store = rows * cols
                    
                    # Ensure we don't exceed array bounds
                    if idx + elements_to_store <= triu_elements:
                        similarity_matrix[idx:idx + elements_to_store] = sim_batch.ravel()
                        idx += elements_to_store
                    
                    similarity_matrix.flush()
                    gc.collect()
                
                pbar.update(1)
        
        print("\nMatrix calculation complete!")
        print(f"Total elements stored: {idx}")
        return similarity_matrix
    
    except Exception as e:
        print(f"Error in similarity calculation: {str(e)}")
        if 'similarity_matrix' in locals():
            del similarity_matrix
        return None

# Add cluster column to x_train_encoded
x_train_encoded['Cluster'] = clusters  # Add this line before calculating similarity matrix

def recommend_songs(track_name, x_train_encoded, y_train, similarity_matrix, speed_kmh=None, top_n=5):
    try:
        if similarity_matrix is None:
            raise ValueError("Similarity matrix is not available")
            
        train_data = pd.concat([x_train_encoded.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        song_index = train_data[train_data['name'] == track_name].index[0]
        n_samples = len(train_data)
        
        # Get similarity scores
        def get_similarity_score(similarity_matrix, index1, index2):
            """Retrieve similarity score from the similarity matrix"""
            if index1 == index2:
                return 1.0  # Similarity with itself is always 1
            elif index1 < index2:
                index1, index2 = index2, index1  # Ensure index1 > index2 for upper triangle
            return similarity_matrix[index1 * (index1 - 1) // 2 + index2]
        
        similarity_scores = [(i, get_similarity_score(similarity_matrix, song_index, i)) 
                             for i in range(n_samples)]
        
        if speed_kmh is not None:
            # Get preferred cluster for current speed
            target_cluster = map_speed_to_cluster(speed_kmh)
            
            # Adjust similarity scores based on cluster matching
            weighted_scores = []
            for idx, score in similarity_scores:
                song_cluster = train_data.iloc[idx]['Cluster']
                # Give higher weight to songs in the target cluster
                cluster_weight = 1.5 if song_cluster == target_cluster else 1.0
                weighted_scores.append((idx, score * cluster_weight))
            
            sorted_songs = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
        else:
            sorted_songs = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        top_songs = sorted_songs[1:top_n+1]
        recommendations = []
        for idx, score in top_songs:
            song_name = train_data.iloc[idx]['name']
            song_cluster = train_data.iloc[idx]['Cluster']
            recommendations.append((song_name, song_cluster))
        return recommendations
    except IndexError:
        return "Track not found in the training dataset."

# Add cluster analysis here
def analyze_clusters(df, clusters, numeric_features):
    """Analyze characteristics of each cluster"""
    # Add cluster labels to the dataframe
    df_analysis = df.copy()
    df_analysis['Cluster'] = clusters
    
    # Calculate mean values for each feature in each cluster
    cluster_means = df_analysis.groupby('Cluster')[numeric_features].mean()
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Heatmap of feature means by cluster
    sns.heatmap(cluster_means, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Cluster Characteristics Heatmap')
    plt.show()
    
    # Box plots for each feature across clusters
    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_analysis, x='Cluster', y=feature)
        plt.title(f'Distribution of {feature} across Clusters')
        plt.show()
    
    # Print cluster characteristics summary
    print("\nCluster Characteristics Summary:")
    for cluster in range(len(cluster_means)):
        print(f"\nCluster {cluster}:")
        characteristics = []
        
        # Get top 3 defining features for this cluster
        cluster_features = cluster_means.loc[cluster]
        top_features = cluster_features.nlargest(3)
        
        for feature, value in top_features.items():
            if value > cluster_means[feature].mean():
                characteristics.append(f"High {feature} ({value:.2f})")
            else:
                characteristics.append(f"Low {feature} ({value:.2f})")
        
        print("Main characteristics:", ", ".join(characteristics))
        
        # Get sample songs from this cluster
        sample_songs = df_analysis[df_analysis['Cluster'] == cluster][['name', 'artists']].head(3)
        print("Example songs:")
        for _, song in sample_songs.iterrows():
            print(f"- {song['name']} by {song['artists']}")

    # Add cluster labels to summary
    cluster_labels = {
        0: "Classical/Instrumental",
        1: "Traditional/Classical Vocal",
        2: "Modern Mixed",
        3: "High Energy Contemporary"
    }
    
    print("\nCluster Labels and Characteristics Summary:")
    for cluster in range(len(cluster_means)):
        print(f"\nCluster {cluster} - {cluster_labels[cluster]}:")
        # ...rest of existing analysis code...

# Run cluster analysis immediately
print("\nAnalyzing cluster characteristics...")
numeric_features_for_analysis = ['tempo', 'valence', 'energy', 'mode', 'acousticness', 
                               'instrumentalness', 'liveness', 'loudness']
analyze_clusters(df, clusters, numeric_features_for_analysis)

# Ask user if they want to continue with similarity matrix calculation
input("\nPress Enter to continue with similarity matrix calculation (this may take a while)...")

# Replace feature matrix calculation with x_train_scaled
print("Calculating similarity matrix for full dataset...")
x_train_scaled_float16 = x_train_scaled.values.astype(np.float16)  # Convert to float16
similarity_matrix = calculate_similarity_in_batches(x_train_scaled_float16, batch_size=500, n_jobs=4)

def map_speed_to_cluster(speed_kmh):
    """
    Maps speed ranges to appropriate song clusters based on analysis:
    0: Classical/Instrumental (Low tempo, High acousticness, Instrumental)
    1: Traditional/Classical Vocal (Low tempo, More major key, Acoustic)
    2: Modern Mixed (Medium-high tempo, Upbeat, Moderate energy)
    3: High Energy Contemporary (Highest tempo, Major key, High energy)
    """
    if speed_kmh < 30:  # Slow city driving/traffic
        return 0  # Classical/Instrumental - calming
    elif speed_kmh < 60:  # City driving
        return 1  # Traditional/Classical Vocal - relaxed but engaging
    elif speed_kmh < 90:  # Highway cruising
        return 2  # Modern Mixed - moderate energy
    else:  # Fast driving
        return 3  # High Energy Contemporary - most energetic

# Example usage with speed
# ...existing code...

# Test recommendations at different speeds
query_song=y_train.iloc[2]["name"]
speeds = [30, 60, 90, 120]
for speed in speeds:
    recommendations = recommend_songs(query_song, x_train_encoded, y_train, similarity_matrix, speed_kmh=speed)
    print(f"\nRecommended songs for '{query_song}' at {speed} km/h:")
    for song, cluster in recommendations:
        print(f"- {song} (Cluster {cluster})")

def save_model_components(x_train_encoded, y_train, similarity_matrix_file, kmeans_model, scaler_model, save_dir='./saved_models'):
    """Save all model components"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the data
    x_train_encoded.to_csv(f"{save_dir}/x_train_encoded.csv", index=False)
    y_train.to_csv(f"{save_dir}/y_train.csv", index=False)
    
    # Copy similarity matrix file instead of loading it
    import shutil
    shutil.copy2(similarity_matrix_file, f"{save_dir}/similarity_matrix.npy")
    
    # Save sklearn models
    joblib.dump(kmeans_model, f"{save_dir}/kmeans_model.joblib")
    joblib.dump(scaler_model, f"{save_dir}/scaler_model.joblib")
    
    print("All model components saved successfully!")


# After training, save all components
if similarity_matrix is not None:
    # Save components using the file path
    save_model_components(x_train_encoded, y_train, 'similarity_matrix.npy', kmeans, scaler)
    
    # Test recommendations
    speeds = [30, 60, 90, 120]
    for speed in speeds:
        recommendations = recommend_songs(query_song, x_train_encoded, y_train, similarity_matrix, speed_kmh=speed)
        print(f"\nRecommended songs for '{query_song}' at {speed} km/h:")
        for song, cluster in recommendations:
            print(f"- {song} (Cluster {cluster})")
    
    # Clean up at the end
    del similarity_matrix
    gc.collect()

# Example of how to load and use the saved models:
"""
# Load the models
loaded_components = load_model_components()
if loaded_components is not None:
    x_train_encoded, y_train, similarity_matrix, kmeans, scaler = loaded_components
    
    # Use loaded models for predictions
    speeds = [30, 60, 90, 120]
    query_song = y_train.iloc[2]["name"]
    for speed in speeds:
        recommendations = recommend_songs(query_song, x_train_encoded, y_train, 
                                       similarity_matrix, speed_kmh=speed)
        print(f"\nRecommended songs for '{query_song}' at {speed} km/h:")
        for song, cluster in recommendations:
            print(f"- {song} (Cluster {cluster})")
"""

# ...rest of existing code...
