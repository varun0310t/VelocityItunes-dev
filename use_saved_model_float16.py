import os
import numpy  as np
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
        
        # Load both features and artist information
        x_train_encoded = pd.read_csv(f"{load_dir}/x_train_encoded.csv")
        y_train = pd.read_csv(f"{load_dir}/y_train.csv")
        
        # Load original dataset to get artist information and IDs
        df = pd.read_csv('./data.csv')
        # Merge artist and ID information with y_train
        y_train['artists'] = df['artists']
        y_train['spotify_id'] = df['id']  # Use original Spotify ID from dataset

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

def clean_artist_string(artist_str):
    """Convert artist string to array of artist names"""
    if isinstance(artist_str, str):
        # Remove outer brackets and split by comma
        # Handle both string representation of list and direct string
        if artist_str.startswith('[') and artist_str.endswith(']'):
            # Handle list representation
            artists = artist_str.strip('[]').split(',')
        else:
            # Handle direct string
            artists = [artist_str]
        
        # Clean each artist name
        return [
            artist.strip().strip('\'\"') 
            for artist in artists 
            if artist.strip()
        ]
    return [str(artist_str)]

def recommend_songs(track_name, x_train_encoded, y_train, similarity_matrix, speed_kmh=None, top_n=50, use_ram=False, CustomMode=False, Mode=-1):
    """Use recommendation logic with original IDs"""
    try:
        start_time = time.time()
        
        # Include artists and ID in train_data
        train_data = pd.concat([
            x_train_encoded.reset_index(drop=True),
            y_train.reset_index(drop=True)
        ], axis=1)
        train_data = train_data.dropna(subset=['name'])
        train_data['id'] = train_data.index  # Add ID column
        
        # Create a lower-cased column for more flexible matching
        train_data['name_lower'] = train_data['name'].str.lower()
        track_name_lower = track_name.strip().lower()
        
        matches = train_data[train_data['name_lower'].str.contains(track_name_lower)]
        if matches.empty:
            raise IndexError("Track not found in the training dataset")
        
        song_index = matches.index[0]
        n_samples = len(train_data)
        
        similarity_scores = get_similarity_scores_vectorized(
            similarity_matrix, song_index, n_samples, use_ram)
        
        if speed_kmh is not None:
            # Use CustomMode and Mode if provided
            target_cluster = Mode if CustomMode else sr.map_speed_to_cluster(speed_kmh)
            clusters = train_data['Cluster'].values
            scores = np.array([score for _, score in similarity_scores])
            
            cluster_weights = np.where(clusters == target_cluster, 1.5, 1.0)
            weighted_scores = scores * cluster_weights
            
            sorted_indices = np.argsort(weighted_scores)[::-1][1:top_n*2]
            valid_recommendations = []
            
            for idx in sorted_indices:
                if len(valid_recommendations) >= top_n:
                    break
                name = train_data.iloc[idx]['name']
                cluster = train_data.iloc[idx]['Cluster']
                artists = clean_artist_string(train_data.iloc[idx]['artists'])
                spotify_id = train_data.iloc[idx]['spotify_id']  # Use Spotify ID
                
                if pd.notna(name):
                    valid_recommendations.append((name, cluster, artists, spotify_id))
            
            recommendations = valid_recommendations[:top_n]
        else:
            sorted_indices = np.argsort([score for _, score in similarity_scores])[::-1][1:top_n+1]
            recommendations = [
                (train_data.iloc[idx]['name'],
                 train_data.iloc[idx]['Cluster'],
                 clean_artist_string(train_data.iloc[idx]['artists']),
                 train_data.iloc[idx]['spotify_id'])  # Use Spotify ID
                for idx in sorted_indices
            ]
        
        return recommendations
    except IndexError:
        raise
    except Exception as e:
        raise e

def get_random_songs_by_speed(speed_kmh, y_train, x_train_encoded, top_n=5):
    """Get random songs with cleaned artist info"""
    try:
        target_cluster = sr.map_speed_to_cluster(speed_kmh)
        
        # Combine all necessary information
        song_data = pd.concat([
            y_train[['name', 'artists', 'spotify_id']],  # Include Spotify ID
            x_train_encoded['Cluster']
        ], axis=1)
        song_data['id'] = song_data.index
        
        cluster_songs = song_data[song_data['Cluster'] == target_cluster]
        
        if cluster_songs.empty:
            return "No songs found for this speed range"
            
        random_songs = cluster_songs.sample(n=min(top_n, len(cluster_songs)), replace=False)
        
        recommendations = [
            (str(row['name']), 
             int(row['Cluster']),
             clean_artist_string(row['artists']),
             str(row['spotify_id']))  # Use Spotify ID
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
