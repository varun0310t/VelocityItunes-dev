from flask import Flask, request, jsonify
from flask_cors import CORS
import use_saved_model_float16 as flt16
import numpy as np
import os 
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return obj  # Return lists of strings as-is
        return super().default(obj)
    
    
app=Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins in development
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})
app.json_encoder = NumpyEncoder
try:
    print("loading model components...")
    model_components=flt16.load_saved_components()
    if model_components is not None:
        x_train_encoded,y_train,similarity_matrix, kmeans, scaler = model_components
        print("Model loaded successfully!")
    else:
        raise ValueError("Failed to load model components")
except Exception as e:
    print(f"Error loading model:{str(e)}")
    exit(1)
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/random-by-speed', methods=['POST'])
def get_random_by_speed():
    try:
        data = request.get_json()
        speed = float(data.get('speed', 60))
        n_songs = int(data.get('n_songs', 5))
        
        if not isinstance(speed, (int, float)) or speed < 0:
            return jsonify({"error": "Invalid speed value"}), 400
            
        recommendations = flt16.get_random_songs_by_speed(
            speed,
            y_train,
            x_train_encoded,
            top_n=n_songs
        )
        
        if isinstance(recommendations, str):
            return jsonify({"error": recommendations}), 404
            
        return jsonify({
            "speed": speed,
            "recommendations": [
                {
                    "name": str(name),
                    "cluster": int(cluster),
                    "artist": artist,  # Now a list of artists
                    "id": str(spotify_id)
                }
                for name, cluster, artist, spotify_id in recommendations
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        song_name = data.get('song_name')
        speed = int(data.get('speed', 60))
        n_songs = int(data.get('n_songs', 5))
        CustomMode = bool(data.get('CustomMode', False))
        Mode = int(data.get('Mode', -1))
        
        if not song_name:
            return jsonify({"error": "song_name is required"}), 400
        
        try:
            recommendations = flt16.recommend_songs(
                song_name,
                x_train_encoded,
                y_train,
                similarity_matrix,
                speed_kmh=speed,
                top_n=n_songs,
                CustomMode=CustomMode,
                Mode=Mode
            )
        except IndexError as idx_err:
            fallback_recs = flt16.get_random_songs_by_speed(speed, y_train, x_train_encoded, top_n=n_songs)
            return jsonify({
                "song": song_name,
                "speed": speed,
                "recommendation_type": "random",
                "recommendations": [
                    {
                        "name": name,
                        "cluster": cluster,
                        "artist": artist,
                        "id": song_id
                    } 
                    for name, cluster, artist, song_id in fallback_recs
                ]
            })
        
        return jsonify({
            "song": song_name,
            "speed": speed,
            "recommendation_type": "match",
            "recommendations": [
                {
                    "name": str(name),
                    "cluster": int(cluster),
                    "artist": artist,  # artist is already a list from clean_artist_string
                    "id": str(song_id)
                }
                for name, cluster, artist, song_id in recommendations
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
                
if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)