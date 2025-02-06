# VelocityItunes-dev

## setup 

pip install -r requirements.txt

# To install packages 

python generateMatrix.py

## To generate the file model once the saved_models is generated u dont need to run this again

python server.py

# To run the server

API Endpoints
1. Song Recommendations
Get song recommendations based on a song and speed.

Endpoint: /recommend Method: POST

Request Body:

{
    "song_name": "Shape of You",
    "speed": 90
}

Response:

{
    "song": "Shape of You",
    "speed": 90,
    "recommendations": [
        {
            "name": "Dance Monkey",
            "cluster": 3
        },
        // ... more songs
    ]
}

2. Random Songs by Speed
Get random songs suitable for current speed.

Endpoint: /random-by-speed Method: POST

{
    "speed": 90,
    "n_songs": 5
}

Response:

{
    "speed": 90,
    "recommendations": [
        {
            "name": "Thunder",
            "cluster": 3
        },
        // ... more songs
    ]
}