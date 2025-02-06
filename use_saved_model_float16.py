import os
import numpy as np
import pandas as pd
import joblib
import speed_recommender as sr
import time

def get_memory_preference():
    print("\nChoose memory management strategy:")
    print("1. Load entire matrix in RAM (Faster but requires more memory)")
    print("2. Use memory mapping (Slower but memory efficient)")
    while True:
        try:
            choice = int(input("Enter your choice (1 or 2): "))
            if choice in [1, 2]:
                return choice == 1
            print("Please enter 1 or 2")
        except ValueError:
            print("Invalid input. Please enter 1 or 2")

def load_saved_components(load_dir='./saved_models'):
    try:
        use_ram = get_memory_preference()
        print(f"\nUsing {'RAM' if use_ram else 'memory mapping'} for matrix operations")
        
        x_train_encoded = pd.read_csv(f"{load_dir}/x_train_encoded.csv")
        y_train = pd.read_csv(f"{load_dir}/y_train.csv")

        n_samples = len(x_train_encoded)
        triu_elements = (n_samples * (n_samples - 1)) // 2
        matrix_size_gb = (triu_elements * 2) / (1024**3)  # 2 bytes for float16
        
        print(f"Matrix size: {matrix_size_gb:.2f} GB")
        
        if use_ram:
            similarity_matrix = np.fromfile(f"{load_dir}/upperTriangle.npy", 
                                         dtype=np.float16)
        else:
            similarity_matrix = np.memmap(f"{load_dir}/upperTriangle.npy",
                                       dtype=np.float16,
                                       mode='r',
                                       shape=(triu_elements,))

        kmeans_model = joblib.load(f"{load_dir}/kmeans_model.joblib")
        scaler_model = joblib.load(f"{load_dir}/scaler_model.joblib")
        
        print("All model components loaded successfully!")
        return x_train_encoded, y_train, similarity_matrix, kmeans_model, scaler_model
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        return None

def get_similarity_scores_vectorized(similarity_matrix, song_index, n_samples, use_ram):
    if use_ram:
        scores = np.ones(n_samples, dtype=np.float32)
        batch_size = 10000
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = np.arange(start, end)
            
            row_indices = np.minimum(batch_indices, song_index)
            col_indices = np.maximum(batch_indices, song_index)
            idx_matrix = (row_indices * (2 * n_samples - row_indices - 1)) // 2 + (col_indices - row_indices - 1)
            
            batch_mask = batch_indices != song_index
            # Direct float16 values - no scaling needed
            scores[start:end][batch_mask] = similarity_matrix[idx_matrix[batch_mask]]
        
        return list(enumerate(scores))
    else:
        # Similar change for memmap version
        indices = np.arange(n_samples)
        scores = np.ones(n_samples, dtype=np.float32)
        mask = indices != song_index
        
        chunk_size = 5000
        for i in range(0, n_samples, chunk_size):
            chunk_end = min(i + chunk_size, n_samples)
            chunk_mask = mask[i:chunk_end]
            if np.any(chunk_mask):
                chunk_indices = indices[i:chunk_end][chunk_mask]
                min_indices = np.minimum(chunk_indices, song_index)
                max_indices = np.maximum(chunk_indices, song_index)
                idx = (min_indices * (2 * n_samples - min_indices - 1)) // 2 + (max_indices - min_indices - 1)
                # Direct float16 values
                scores[chunk_indices] = similarity_matrix[idx]
        
        return list(enumerate(scores))

def recommend_songs(track_name, x_train_encoded, y_train, similarity_matrix, speed_kmh=None, top_n=50, use_ram=False):
    """Use the same recommendation logic as standalone version for natural diversity"""
    try:
        start_time = time.time()
        
        train_data = pd.concat([x_train_encoded.reset_index(drop=True),
                              y_train.reset_index(drop=True)], axis=1)
        train_data = train_data.dropna(subset=['name'])
        
        song_index = train_data[train_data['name'] == track_name].index[0]
        n_samples = len(train_data)
        
        # Get similarity scores
        similarity_scores = get_similarity_scores_vectorized(
            similarity_matrix, song_index, n_samples, use_ram)
        
        if speed_kmh is not None:
            target_cluster = sr.map_speed_to_cluster(speed_kmh)
            clusters = train_data['Cluster'].values
            scores = np.array([score for _, score in similarity_scores])
            
            # Simple cluster weighting
            cluster_weights = np.where(clusters == target_cluster, 1.5, 1.0)
            weighted_scores = scores * cluster_weights
            
            sorted_indices = np.argsort(weighted_scores)[::-1][1:top_n*2]
            valid_recommendations = []
            
            # Get valid recommendations
            for idx in sorted_indices:
                if len(valid_recommendations) >= top_n:
                    break
                name = train_data.iloc[idx]['name']
                cluster = train_data.iloc[idx]['Cluster']
                if pd.notna(name):
                    valid_recommendations.append((name, cluster))
            
            recommendations = valid_recommendations[:top_n]
        else:
            sorted_indices = np.argsort([score for _, score in similarity_scores])[::-1][1:top_n+1]
            recommendations = [(train_data.iloc[idx]['name'], 
                             train_data.iloc[idx]['Cluster']) 
                             for idx in sorted_indices]
        
        return recommendations
    except IndexError:
        return "Track not found in the training dataset.", 0
def get_random_songs_by_speed(speed_kmh, y_train, x_train_encoded, top_n=5):
    """Get random songs based on speed only"""
    try:
        # Map speed to appropriate cluster
        target_cluster = sr.map_speed_to_cluster(speed_kmh)
        
        # Combine song data with cluster info
        song_data = pd.concat([y_train, x_train_encoded['Cluster']], axis=1)
        
        # Filter songs from target cluster
        cluster_songs = song_data[song_data['Cluster'] == target_cluster]
        
        if cluster_songs.empty:
            return "No songs found for this speed range"
            
        # Get random songs
        random_songs = cluster_songs.sample(
            n=min(top_n, len(cluster_songs)),
            replace=False
        )
        
        # Format output
        recommendations = [
            (str(row['name']), int(row['Cluster'])) 
            for _, row in random_songs.iterrows()
        ]
        
        return recommendations
        
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"
def main():
    components = load_saved_components()
    if components:
        x_train_encoded, y_train, similarity_matrix, kmeans, scaler = components
        use_ram = isinstance(similarity_matrix, np.ndarray)

        print("\nFirst 5 available songs:")
        print(y_train['name'].head())

        while True:
            try:
                idx = int(input("\nEnter the index of a song to recommend (or -1 to exit): "))
                if idx == -1:
                    break
                query_song = y_train.iloc[idx]['name']
            except:
                print("Invalid input. Using default song at index 0.")
                query_song = y_train.iloc[0]['name']

            speed = float(input("Enter current speed in km/h: "))

            start_time = time.time()
            recommendations, proc_time = recommend_songs(
                query_song, x_train_encoded, y_train,
                similarity_matrix, speed_kmh=speed, use_ram=use_ram
            )
            random_recommendation = sr.random_recommendation_by_speed(
                speed, y_train, x_train_encoded
            )
            total_time = time.time() - start_time
            
            print(f"\nRecommended songs for '{query_song}' at {speed} km/h:")
            print(f"Processing time: {proc_time:.3f} seconds")
            print(f"Total response time: {total_time:.3f} seconds")
            
            for song, cluster in recommendations:
                print(f"- {song} (Cluster {cluster})")
                
            print("\nRandom recommendations:")
            for song in random_recommendation:
                print(f"- {song}")
    else:
        print("No model components available. Please ensure saved files exist.")

if __name__ == "__main__":
    main()
