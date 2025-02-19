import random
import pandas as pd


def map_speed_to_cluster(speed_kmh):
    """
    Maps speed ranges to appropriate song clusters based on analysis:
    0: Classical/Instrumental (Low tempo, High acousticness, Instrumental)
    1: Traditional/Classical Vocal (Low tempo, More major key, Acoustic)
    2: Modern Mixed (Medium-high tempo, Upbeat, Moderate energy)
    3: High Energy Contemporary (Highest tempo, Major key, High energy)
    """
    if speed_kmh < 30:      # Slow city driving
        return 0
    elif speed_kmh < 60:    # City driving
        return 1
    elif speed_kmh < 90:    # Highway cruising
        return 2
    else:                   # Fast driving
        return 3

def random_recommendation_by_speed(speed_kmh, y_train, x_train_encoded, count=5):
    """Get random song recommendations with artist and ID information"""
    cluster_id = map_speed_to_cluster(speed_kmh)
    train_data = pd.concat([x_train_encoded.reset_index(drop=True),
                          y_train.reset_index(drop=True)], axis=1)
    
    # Filter out nan values
    train_data = train_data.dropna(subset=['name'])
    cluster_songs = train_data[train_data['Cluster'] == cluster_id]
    
    if cluster_songs.empty:
        return "No songs found for this speed range"
        
    # Get random songs with full information
    random_songs = cluster_songs.sample(n=min(count, len(cluster_songs)), replace=False)
    
    # Format output with all required fields
    recommendations = []
    for _, row in random_songs.iterrows():
        if pd.notna(row['name']):
            recommendations.append((
                str(row['name']),
                int(row['Cluster']),
                clean_artist_string(row['artists']),
                str(row['spotify_id'])
            ))
    
    return recommendations if recommendations else "No valid songs found in this cluster"