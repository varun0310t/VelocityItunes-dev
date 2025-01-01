import yt_dlp
import librosa
import pandas as pd
import os
import time


# Set up a function to extract features using librosa
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio file
        valence = librosa.feature.spectral_centroid(y=y, sr=sr).mean()  # Valence (approximation)
        energy = librosa.feature.rms(y=y).mean()  # Energy
        mode = librosa.beat.tempo(y=y, sr=sr)[0]  # Mode (will use tempo for now)
        acousticness = librosa.feature.spectral_flatness(y=y).mean()  # Acousticness
        speechiness = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()  # Speechiness
        loudness = librosa.core.amplitude_to_db(y).mean()  # Loudness
        key = librosa.feature.zero_crossing_rate(y).mean()  # Key (approximated by zero-crossing)

        return {
            'valence': valence,
            'energy': energy,
            'mode': mode,
            'acousticness': acousticness,
            'speechiness': speechiness,
            'loudness': loudness,
            'key': key
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


# Function to download a song using yt_dlp
def download_song(song_name, download_path="downloads"):
    try:
        # Construct the URL search query
        search_url = f"ytsearch:{song_name}"

        # Download the first result from YouTube
        ydl_opts = {
            'format': 'bestaudio/best',  # Download the best audio quality
            'outtmpl': os.path.join(download_path, f"{song_name}.%(ext)s"),  # Save to the download path with song name
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_url, download=False)  # Get video info without downloading
            video_url = result['entries'][0]['url']  # Get the URL of the first video

            # Now download the song
            ydl_opts['outtmpl'] = os.path.join(download_path, f"{song_name}.mp3")  # Save as mp3
            ydl_opts['format'] = 'bestaudio/best'  # Download the best audio quality

            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([video_url])

            return os.path.join(download_path, f"{song_name}.mp3")
    except Exception as e:
        print(f"Error downloading song {song_name}: {e}")
        return None  

# Function to save the checkpoint
def save_checkpoint(song_name, checkpoint_file):
    with open(checkpoint_file, 'a') as f:
        f.write(f"{song_name}\n")


# Main processing function
def process_songs(input_csv, output_csv, checkpoint_file):
    # Load song names from the input CSV
    song_names = pd.read_csv(input_csv)['song'].tolist()

    # Load already processed songs from the checkpoint file
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_songs = f.read().splitlines()
    else:
        processed_songs = []

    # Load the existing output CSV, if it exists
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
    else:
        existing_df = pd.DataFrame()

    features_list = []
    for song in song_names:
        if song in processed_songs:
            print(f"Skipping {song}: already processed.")
            continue

        print(f"Processing: {song}")
        start_time = time.time()

        # Download the song
        song_path = download_song(song)
        if song_path:
            # Extract audio features
            features = extract_audio_features(song_path)
            if features:
                features['name'] = song
                features_list.append(features)

                # Append new features to the existing DataFrame
                new_entry = pd.DataFrame([features])
                updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
                updated_df.to_csv(output_csv, index=False)

            # Clean up by removing the downloaded song file
            os.remove(song_path)

        # Save the checkpoint
        save_checkpoint(song, checkpoint_file)

        end_time = time.time()
        print(f"Time taken for {song}: {end_time - start_time:.2f} seconds")


# Example usage
if __name__ == "__main__":
    input_csv = "spotify_millsongdata.csv"  # CSV file containing song names
    output_csv = "output.csv"
    checkpoint_file = "checkpoint.txt"

    process_songs(input_csv, output_csv, checkpoint_file)
