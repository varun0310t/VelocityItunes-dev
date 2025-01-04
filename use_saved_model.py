import os
import numpy as np
import pandas as pd
import joblib
import speed_recommender as sr

def load_saved_components(load_dir='./saved_models'):
    """
    Load all model components from saved files.
    """
    try:
        # Load encoded training data and song names
        x_train_encoded = pd.read_csv(f"{load_dir}/x_train_encoded.csv")
        y_train = pd.read_csv(f"{load_dir}/y_train.csv")

        # Load similarity matrix
        similarity_matrix = np.memmap(f"{load_dir}/similarity_matrix.npy",dtype='float32',mode='r',shape=(x_train_encoded.shape[0], x_train_encoded.shape[0]))

        # Load sklearn models
        kmeans_model = joblib.load(f"{load_dir}/kmeans_model.joblib")
        scaler_model = joblib.load(f"{load_dir}/scaler_model.joblib")
        pca_model = joblib.load(f"{load_dir}/pca_model.joblib")
        
        
        print("All model components loaded successfully!")
        return x_train_encoded, y_train, similarity_matrix, kmeans_model, scaler_model, pca_model
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        return None

def recommend_songs(track_name, x_train_encoded, y_train, similarity_matrix, speed_kmh=None, top_n=5):
    """
    Recommend songs based on similarity matrix, with optional weighting by cluster preference.
    """
    try:
        train_data = pd.concat([x_train_encoded.reset_index(drop=True),
                                y_train.reset_index(drop=True)], axis=1)
        song_index = train_data[train_data['name'] == track_name].index[0]
        similarity_scores = list(enumerate(similarity_matrix[song_index]))

        if speed_kmh is not None:
            target_cluster = sr.map_speed_to_cluster(speed_kmh)
            weighted_scores = []
            for idx, score in similarity_scores:
                song_cluster = train_data.iloc[idx]['Cluster']
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

def main():
    # Load the models
    components = load_saved_components()
    if components:
        x_train_encoded, y_train, similarity_matrix, kmeans, scaler, pca = components

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

            # Get recommendations
            recommendations = recommend_songs(query_song, x_train_encoded, y_train,
                                              similarity_matrix, speed_kmh=speed)
            random_recommendation = sr.random_recommendation_by_speed(speed, y_train,x_train_encoded)   
        
            print(f"\nRecommended songs for '{query_song}' at {speed} km/h:")
            for song, cluster in recommendations:
                print(f"- {song} (Cluster {cluster})")
                
            print("random_recommendation_by_speed:")
            for song in random_recommendation:
                print(f"- {song}")    
    else:
        print("No model components available. Please ensure saved files exist.")

if __name__ == "__main__":
    main()