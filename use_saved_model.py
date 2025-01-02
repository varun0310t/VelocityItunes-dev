from generateMatrix import load_model_components, recommend_songs

def main():
    # Load the models
    loaded_components = load_model_components()
    if loaded_components is not None:
        x_train_encoded, y_train, similarity_matrix, kmeans, scaler, pca = loaded_components
        
        # Get available songs
        print("\nFirst 5 available songs:")
        print(y_train['name'].head())
        
        # Get song name from user
        query_song = input("\nEnter a song name from the dataset: ")
        speed = float(input("Enter current speed in km/h: "))
        
        # Get recommendations
        recommendations = recommend_songs(query_song, x_train_encoded, y_train, 
                                       similarity_matrix, speed_kmh=speed)
        
        print(f"\nRecommended songs for '{query_song}' at {speed} km/h:")
        for song, cluster in recommendations:
            print(f"- {song} (Cluster {cluster})")

if __name__ == "__main__":
    main()
