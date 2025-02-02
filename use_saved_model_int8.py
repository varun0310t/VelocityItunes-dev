import os
import numpy as np
import pandas as pd
import joblib
import speed_recommender as sr
import time  # Add this import

def get_memory_preference():
    print("\nChoose memory management strategy:")
    print("1. Load entire matrix in RAM (Faster but requires more memory)")
    print("2. Use memory mapping (Slower but memory efficient)")
    while True:
        try:
            choice = int(input("Enter your choice (1 or 2): "))
            if choice in [1, 2]:
                return choice == 1  # Returns True for RAM, False for memmap
            print("Please enter 1 or 2")
        except ValueError:
            print("Invalid input. Please enter 1 or 2")

def load_saved_components(load_dir='./saved_models'):
    """
    Load all model components from saved files.
    """
    try:
        # Get user preference for memory management
        use_ram = get_memory_preference()
        print(f"\nUsing {'RAM' if use_ram else 'memory mapping'} for matrix operations")
        
        # Load encoded training data and song names
        x_train_encoded = pd.read_csv(f"{load_dir}/x_train_encoded.csv")
        y_train = pd.read_csv(f"{load_dir}/y_train.csv")

        # Calculate matrix size
        n_samples = len(x_train_encoded)
        triu_elements = (n_samples * (n_samples - 1)) // 2
        matrix_size_gb = (triu_elements * 1) / (1024**3)
        
        print(f"Matrix size: {matrix_size_gb:.2f} GB")
        
        # Load similarity matrix based on user preference
        if use_ram:
            # Load entire matrix into RAM
            similarity_matrix = np.fromfile(f"{load_dir}/similarity_matrix.npy", 
                                         dtype='int8')
        else:
            # Use memory mapping
            similarity_matrix = np.memmap(f"{load_dir}/similarity_matrix.npy",
                                       dtype='int8',
                                       mode='r',
                                       shape=(triu_elements,))

        # Load sklearn models
        kmeans_model = joblib.load(f"{load_dir}/kmeans_model.joblib")
        scaler_model = joblib.load(f"{load_dir}/scaler_model.joblib")
        
        print("All model components loaded successfully!")
        return x_train_encoded, y_train, similarity_matrix, kmeans_model, scaler_model
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        return None

def get_similarity_score(similarity_matrix, index1, index2, n_samples):
    """Get similarity score from int8 matrix and convert back to float range"""
    if index1 == index2:
        return 1.0
    if index1 > index2:
        index1, index2 = index2, index1
    
    # Calculate position in upper triangular matrix
    idx = (index1 * (2 * n_samples - index1 - 1)) // 2 + (index2 - index1 - 1)
    
    # Convert back from int8 to float range [-1,1]
    return float(similarity_matrix[idx]) / 127.5 - 1.0

def get_similarity_scores_vectorized(similarity_matrix, song_index, n_samples, use_ram):
    """Optimized vectorized similarity score calculation"""
    if use_ram:
        # Pre-allocate scores array
        scores = np.ones(n_samples, dtype=np.float32)
        
        # Calculate indices in batches to avoid memory issues
        batch_size = 10000
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = np.arange(start, end)
            
            # Vectorized index calculation for batch
            row_indices = np.minimum(batch_indices, song_index)
            col_indices = np.maximum(batch_indices, song_index)
            idx_matrix = (row_indices * (2 * n_samples - row_indices - 1)) // 2 + (col_indices - row_indices - 1)
            
            # Update scores for batch
            batch_mask = batch_indices != song_index
            scores[start:end][batch_mask] = similarity_matrix[idx_matrix[batch_mask]].astype(np.float32) / 127.5 - 1.0
        
        # Add numerical stability
        scores = np.clip(scores, -1.0, 1.0)  # Ensure scores are in valid range
        return list(enumerate(scores))
    else:
        # Use numpy's advanced indexing for memmap
        indices = np.arange(n_samples)
        scores = np.ones(n_samples, dtype=np.float32)
        mask = indices != song_index
        
        # Process in smaller chunks for memory efficiency
        chunk_size = 5000
        for i in range(0, n_samples, chunk_size):
            chunk_end = min(i + chunk_size, n_samples)
            chunk_mask = mask[i:chunk_end]
            if np.any(chunk_mask):
                chunk_indices = indices[i:chunk_end][chunk_mask]
                min_indices = np.minimum(chunk_indices, song_index)
                max_indices = np.maximum(chunk_indices, song_index)
                idx = (min_indices * (2 * n_samples - min_indices - 1)) // 2 + (max_indices - min_indices - 1)
                scores[chunk_indices] = similarity_matrix[idx].astype(np.float32) / 127.5 - 1.0
        
        # Add numerical stability
        scores = np.clip(scores, -1.0, 1.0)  # Ensure scores are in valid range
        return list(enumerate(scores))

def recommend_songs(track_name, x_train_encoded, y_train, similarity_matrix, speed_kmh=None, top_n=50, use_ram=False):
    """Optimized song recommendation with validation"""
    try:
        start_time = time.time()
        
        train_data = pd.concat([x_train_encoded.reset_index(drop=True),
                              y_train.reset_index(drop=True)], axis=1)
        
        # Filter out nan values
        train_data = train_data.dropna(subset=['name'])
        
        song_index = train_data[train_data['name'] == track_name].index[0]
        n_samples = len(train_data)
        
        # Get similarity scores using optimized vectorization
        similarity_scores = get_similarity_scores_vectorized(
            similarity_matrix, song_index, n_samples, use_ram)
        
        if speed_kmh is not None:
            target_cluster = sr.map_speed_to_cluster(speed_kmh)
            clusters = train_data['Cluster'].values
            scores = np.array([score for _, score in similarity_scores])
            
            # Vectorized cluster weighting with validation
            cluster_weights = np.where(clusters == target_cluster, 10, 1.0)
            weighted_scores = scores * cluster_weights
            
            # Filter out nan and invalid scores
            valid_indices = ~np.isnan(weighted_scores)
            weighted_scores[~valid_indices] = -np.inf
            
            sorted_indices = np.argsort(weighted_scores)[::-1]
            # Get more recommendations than needed to filter nans
            candidate_indices = sorted_indices[1:top_n*2]
            
            # Filter recommendations
            valid_recommendations = []
            for idx in candidate_indices:
                if len(valid_recommendations) >= top_n:
                    break
                name = train_data.iloc[idx]['name']
                cluster = train_data.iloc[idx]['Cluster']
                if pd.notna(name):
                    valid_recommendations.append((name, cluster))
            
            recommendations = valid_recommendations[:top_n]
        else:
            sorted_indices = np.argsort([score for _, score in similarity_scores])[::-1][1:top_n+1]
        
            # Get recommendations
            recommendations = [(train_data.iloc[idx]['name'], 
                          train_data.iloc[idx]['Cluster']) 
                         for idx in sorted_indices]
        
        return recommendations, time.time() - start_time
    except IndexError:
        return "Track not found in the training dataset.", 0

def main():
    # Load the models
    components = load_saved_components()
    if components:
        x_train_encoded, y_train, similarity_matrix, kmeans, scaler = components
        # Store RAM mode preference
        use_ram = isinstance(similarity_matrix, np.ndarray)

        # Show some available songs
        print("\nFirst 5 available songs:")
        print(y_train['name'].head())

        # Get user input (song index for demonstration)
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

            # Get recommendations with timing
            start_time = time.time()
            recommendations, proc_time = recommend_songs(query_song, x_train_encoded, y_train,
                                                          similarity_matrix, speed_kmh=speed, use_ram=use_ram)
            random_recommendation = sr.random_recommendation_by_speed(speed, y_train,x_train_encoded)   
            total_time = time.time() - start_time
            
            print(f"\nRecommended songs for '{query_song}' at {speed} km/h:")
            print(f"Processing time: {proc_time:.3f} seconds")
            print(f"Total response time: {total_time:.3f} seconds")
            
            for song, cluster in recommendations:
                print(f"- {song} (Cluster {cluster})")
                
            print("random_recommendation_by_speed:")
            for song in random_recommendation:
                print(f"- {song}")    
    else:
        print("No model components available. Please ensure saved files exist.")

if __name__ == "__main__":
    main()