import axios from 'axios';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import * as csvWriter from 'csv-writer';
import csvParser from 'csv-parser';

dotenv.config();

const clientId = process.env.SPOTIFY_CLIENT_ID;
const clientSecret = process.env.SPOTIFY_CLIENT_SECRET;

const getToken = async (): Promise<string> => {
    const response = await axios.post('https://accounts.spotify.com/api/token', 
        new URLSearchParams({
            'grant_type': 'client_credentials'
        }), {
        headers: {
            'Authorization': 'Basic ' + Buffer.from(clientId + ':' + clientSecret).toString('base64'),
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    });
    return response.data.access_token;
};

const getTrackDetails = async (token: string, trackId: string): Promise<any> => {
    const response = await axios.get(`https://api.spotify.com/v1/tracks/${trackId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    return response.data;
};

const getAudioFeatures = async (token: string, trackId: string): Promise<any> => {
    console.log("here 2")
    const response = await axios.get(`https://api.spotify.com/v1/audio-features/${trackId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    return response.data;
};

const getArtistDetails = async (token: string, artistId: string): Promise<any> => {
    const response = await axios.get(`https://api.spotify.com/v1/artists/${artistId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    return response.data;
};

const writeToCSV = (data: any[], filePath: string) => {
    const csvWriterInstance = csvWriter.createObjectCsvWriter({
        path: filePath,
        header: [
            {id: 'track_id', title: 'Track ID'},
            {id: 'track_name', title: 'Track Name'},
            {id: 'track_artist', title: 'Track Artist'},
            {id: 'lyrics', title: 'Lyrics'},
            {id: 'track_popularity', title: 'Track Popularity'},
            {id: 'track_album_id', title: 'Track Album ID'},
            {id: 'track_album_name', title: 'Track Album Name'},
            {id: 'track_album_release_date', title: 'Track Album Release Date'},
            {id: 'playlist_name', title: 'Playlist Name'},
            {id: 'playlist_id', title: 'Playlist ID'},
            {id: 'playlist_genre', title: 'Playlist Genre'},
            {id: 'playlist_subgenre', title: 'Playlist Subgenre'},
            {id: 'danceability', title: 'Danceability'},
            {id: 'energy', title: 'Energy'},
            {id: 'key', title: 'Key'},
            {id: 'loudness', title: 'Loudness'},
            {id: 'mode', title: 'Mode'},
            {id: 'speechiness', title: 'Speechiness'},
            {id: 'acousticness', title: 'Acousticness'},
            {id: 'instrumentalness', title: 'Instrumentalness'},
            {id: 'liveness', title: 'Liveness'},
            {id: 'valence', title: 'Valence'},
            {id: 'tempo', title: 'Tempo'},
            {id: 'duration_ms', title: 'Duration (ms)'},
            {id: 'language', title: 'Language'},
            {id: 'artist_genres', title: 'Artist Genres'}
        ],
        append: true
    });

    csvWriterInstance.writeRecords(data)
        .then(() => console.log('Data written to CSV file successfully.'));
};

const readCheckpoint = (checkpointFilePath: string): number => {
    if (fs.existsSync(checkpointFilePath)) {
        const checkpoint = fs.readFileSync(checkpointFilePath, 'utf8');
        return parseInt(checkpoint, 10);
    }
    return 0;
};

const writeCheckpoint = (checkpointFilePath: string, index: number) => {
    fs.writeFileSync(checkpointFilePath, index.toString(), 'utf8');
};

const main = async () => {
    const inputFilePath = path.join(__dirname, 'spotify_millsongdata.csv'); // CSV file with song names
    const outputFilePath = path.join(__dirname, 'SongDataset.csv'); // Output CSV file
    const checkpointFilePath = path.join(__dirname, 'checkpoint.txt'); // Checkpoint file

    if (!fs.existsSync(outputFilePath)) {
        fs.writeFileSync(outputFilePath, 'Track ID,Track Name,Track Artist,Lyrics,Track Popularity,Track Album ID,Track Album Name,Track Album Release Date,Playlist Name,Playlist ID,Playlist Genre,Playlist Subgenre,Danceability,Energy,Key,Loudness,Mode,Speechiness,Acousticness,Instrumentalness,Liveness,Valence,Tempo,Duration (ms),Language,Artist Genres\n');
    }

    try {
        const token = await getToken();
        const songNames: string[] = [];

        // Read song names from the input CSV file
        fs.createReadStream(inputFilePath)
            .pipe(csvParser())
            .on('data', (row) => {
                songNames.push(row.song); // Assuming the column name is 'song'
            })
            .on('end', async () => {
                console.log('CSV file successfully processed');
                
                let startIndex = readCheckpoint(checkpointFilePath);

                for (let i = startIndex; i < songNames.length; i++) {
                    const songName = songNames[i];
                    const searchResponse = await axios.get('https://api.spotify.com/v1/search', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        },
                        params: {
                            q: songName,
                            type: 'track',
                            limit: 1
                        }
                    });

                    if (searchResponse.data.tracks.items.length > 0) {
                        const track = searchResponse.data.tracks.items[0];
                        //const trackDetails = await getTrackDetails(token, track.id);
                        console.log("here")
                        const audioFeatures = await getAudioFeatures(token, track.id);
                       // const artistDetails = await getArtistDetails(token, track.artists[0].id);

                        const trackData = {
                            track_id: track.id,
                            track_name: track.name,
                            track_artist: track.artists[0].name,
                            lyrics: '', // You will need to fetch lyrics from another source
                            track_popularity: track.popularity,
                            track_album_id: track.album.id,
                            track_album_name: track.album.name,
                            track_album_release_date: track.album.release_date,
                            playlist_name: '', // You will need to fetch playlist info from another source
                            playlist_id: '', // You will need to fetch playlist info from another source
                            playlist_genre: '', // You will need to fetch playlist info from another source
                            playlist_subgenre: '', // You will need to fetch playlist info from another source
                            danceability: audioFeatures.danceability,
                            energy: audioFeatures.energy,
                            key: audioFeatures.key,
                            loudness: audioFeatures.loudness,
                            mode: audioFeatures.mode,
                            speechiness: audioFeatures.speechiness,
                            acousticness: audioFeatures.acousticness,
                            instrumentalness: audioFeatures.instrumentalness,
                            liveness: audioFeatures.liveness,
                            valence: audioFeatures.valence,
                            tempo: audioFeatures.tempo,
                            duration_ms: audioFeatures.duration_ms,
                            language: '', // You will need to fetch language info from another source
                           // artist_genres: artistDetails.genres.join(', ')
                        };

                        writeToCSV([trackData], outputFilePath);
                        writeCheckpoint(checkpointFilePath, i + 1);
                    }

                    await new Promise(resolve => setTimeout(resolve, 60000)); // Wait for 1 minute between requests to avoid rate limiting
                }
            });
    } catch (error) {
        console.error('Error fetching data from Spotify API:', error);
    }
};

main();