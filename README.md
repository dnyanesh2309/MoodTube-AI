# MoodTube AI – Intelligent Music Recommendation System

MoodTube AI is a music recommendation application that suggests songs based on audio features and user preferences.

## Features
- Personalized music recommendations
- Mood-based song discovery
- Song search
- Spotify authentication using OAuth
- Spotify playlist creation and export
- Interactive web interface

## Technologies Used
- Python
- Flask
- Pandas and NumPy
- Spotify Web API
- HTML, CSS and JavaScript
- Cosine Similarity

## Installation

1. Clone this repository:
   git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate into the project:
   cd MoodTune_AI

3. Install dependencies:
   pip install -r requirements.txt

4. Create a `.env` file using `.env.example` as a template.

5. Add your Spotify API credentials to `.env`.

6. Run the application:
   python MoodTube_AI.py

7. Open your browser:
   http://127.0.0.1:5000

## Project Structure

- MoodTube_AI.py – Flask application and recommendation logic
- index.html – Standalone HTML interface
- dataset.csv – Music dataset
- requirements.txt – Python dependencies
- .env.example – Spotify credentials template
- .gitignore – Files excluded from Git

## Security
Never upload your `.env` file or expose your Spotify API credentials.

## Author
Your Name
