import joblib
import pandas as pd

# Load the scaler, PCA model, and similarity matrix
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')
similarity_matrix = joblib.load('similarity_matrix.pkl')

# Load the encoded training data
x_train_encoded = pd.read_csv('x_train_encoded.csv')
#x_test=pd.read_csv("x_test.csv")
#y_test=pd.read_csv("y_test.csv")
# Load y_train
y_train = pd.read_csv('y_train.csv')

# Function to recommend songs
def recommend_songs(track_name, x_train_encoded, y_train, similarity_matrix, top_n=5):
    try:
        train_data = pd.concat([x_train_encoded.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)

        song_index = train_data[train_data['track_name'] == track_name].index[0]
        similarity_scores = list(enumerate(similarity_matrix[song_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:top_n+1]

        recommended_songs = [train_data.iloc[i[0]]['track_name'] for i in similarity_scores]

        return recommended_songs
    except IndexError:
        return f"Track '{track_name}' not found in the dataset."
    except ValueError as e:
        return str(e)

# Loop to continuously get recommendations
while True:
    index = int(input("Enter a track name (or type 'exit' to quit): "))
    query_song=y_train.iloc[index]['track_name']
    if query_song.lower() == 'exit':
        break
    recommendations = recommend_songs(query_song, x_train_encoded, y_train, similarity_matrix)
    print(f"Recommended songs for '{query_song}': {recommendations}")