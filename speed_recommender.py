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
    """Get random song recommendations for given speed, filtering out nan values"""
    cluster_id = map_speed_to_cluster(speed_kmh)
    train_data = pd.concat([x_train_encoded.reset_index(drop=True),
                          y_train.reset_index(drop=True)], axis=1)
    
    # Filter out nan values before getting cluster songs
    train_data = train_data.dropna(subset=['name'])
    cluster_songs = train_data[train_data['Cluster'] == cluster_id]['name']
    
    if cluster_songs.empty:
        return ["No songs available for this cluster."]
    
    # Convert to list and filter out any remaining nans
    available_songs = [song for song in cluster_songs.tolist() if pd.notna(song)]
    
    if not available_songs:
        return ["No valid songs found in this cluster."]
    
    # Get random sample or all available songs if less than count
    if len(available_songs) <= count:
        return available_songs
    return random.sample(available_songs, count)