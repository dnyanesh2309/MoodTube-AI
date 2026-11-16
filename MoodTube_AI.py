from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)
app.secret_key = os.urandom(24)

# -------------------------------------------------------------------
# SPOTIFY DEVELOPER CREDENTIALS
# -------------------------------------------------------------------
SPOTIPY_CLIENT_ID = '524499af3e484683b4fe9bb7b5041873'
SPOTIPY_CLIENT_SECRET = '75b2f4e027814ceb81e264ab112600eb'
SPOTIPY_REDIRECT_URI = 'http://127.0.0.1:5000/callback'
SCOPE = 'playlist-modify-public playlist-modify-private'

def create_spotify_oauth():
    return SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SCOPE,
        cache_handler=spotipy.cache_handler.FlaskSessionCacheHandler(session)
    )

# -------------------------------------------------------------------
# 1. LOAD DATASET & NUMPY KNN ENGINE
# -------------------------------------------------------------------
DATASET_PATH = 'dataset.csv' # Adjust if it's inside a subfolder

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Could not find '{DATASET_PATH}'. Place it in the same folder as MoodTube_AI.py.")

df = pd.read_csv(DATASET_PATH)

# Optional: Clean up standard index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Ensure required acoustic columns exist or are properly handled
feature_cols = ['energy', 'danceability', 'valence', 'acousticness', 'tempo']

# If your dataset already has these columns pre-calculated from Spotify API, 
# you can skip df.apply() or adapt it safely using 'track_genre' and 'artists':
X = df[feature_cols].values
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1.0

X_scaled = (X - X_mean) / X_std
X_norms = np.linalg.norm(X_scaled, axis=1)

def compute_knn(query_vector, top_k=9):
    q_scaled = (query_vector - X_mean) / X_std
    q_norm = np.linalg.norm(q_scaled)
    if q_norm == 0: q_norm = 1.0
    
    dot_products = np.dot(X_scaled, q_scaled)
    cosine_sim = dot_products / (X_norms * q_norm + 1e-9)
    top_indices = np.argsort(cosine_sim)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        sim_pct = round(float(cosine_sim[idx]) * 100, 1)
        results.append({
            'track_id': str(row['track_id']),
            'track_name': str(row['track_name']),
            'artist_name': str(row['artists']),          # Fixed column name 'artists'
            'genres': str(row['track_genre']) if pd.notnull(row['track_genre']) else "pop", # Fixed column name 'track_genre'
            'match': max(0.0, min(100.0, sim_pct))
        })
    return results

# -------------------------------------------------------------------
# 2. HIGH-END CYBERPUNK UI WITH AUDIO PLAYER
# -------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOODTUBE AI // Neural Cyber-Audio Matrix</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        :root {
            --cyan: #00f3ff;
            --pink: #ff0055;
            --purple: #9d00ff;
            --green: #1db954;
            --dark-bg: #03030b;
            --glass-bg: rgba(10, 11, 28, 0.75);
            --border-neon: rgba(0, 243, 255, 0.3);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Rajdhani', sans-serif; }
        
        body {
            background-color: var(--dark-bg);
            color: #fff;
            min-height: 100vh;
            padding-bottom: 120px;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 243, 255, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(255, 0, 85, 0.12) 0%, transparent 45%),
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
        }

        canvas#visualizer-canvas {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            z-index: 0; pointer-events: none; opacity: 0.6;
        }

        .wrapper {
            position: relative; z-index: 10; max-width: 1400px; margin: 0 auto; padding: 25px;
        }

        /* Top HUD Header */
        header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 20px 30px; background: var(--glass-bg);
            border: 1px solid var(--border-neon); border-radius: 20px;
            backdrop-filter: blur(20px); margin-bottom: 30px;
            box-shadow: 0 0 30px rgba(0, 243, 255, 0.15);
        }

        .brand-box { display: flex; align-items: center; gap: 15px; }
        .brand-logo {
            width: 48px; height: 48px; background: linear-gradient(135deg, var(--cyan), var(--pink));
            border-radius: 12px; display: flex; align-items: center; justify-content: center;
            font-family: 'Orbitron', sans-serif; font-weight: 900; color: #000; font-size: 1.5rem;
            box-shadow: 0 0 20px var(--cyan); animation: pulseLogo 3s infinite alternate;
        }
        @keyframes pulseLogo { 0% { box-shadow: 0 0 15px var(--cyan); } 100% { box-shadow: 0 0 25px var(--pink); } }

        header h1 {
            font-family: 'Orbitron', sans-serif; font-size: 2.2rem; font-weight: 900; letter-spacing: 4px;
            background: linear-gradient(90deg, #fff, var(--cyan), var(--pink));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }

        .sys-status {
            font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: var(--cyan);
            letter-spacing: 2px; text-transform: uppercase; display: flex; align-items: center; gap: 8px;
        }
        .status-dot { width: 8px; height: 8px; background: var(--green); border-radius: 50%; box-shadow: 0 0 10px var(--green); animation: blink 1.5s infinite; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        .btn-spotify-login {
            background: var(--green); color: #000; font-family: 'Orbitron', sans-serif; font-weight: 800;
            padding: 10px 22px; border-radius: 30px; text-decoration: none; font-size: 0.85rem;
            box-shadow: 0 0 20px rgba(29, 185, 84, 0.4); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .btn-spotify-login:hover { transform: translateY(-2px) scale(1.05); box-shadow: 0 0 30px var(--green); }

        /* Search HUD Box */
        .search-container { position: relative; margin-bottom: 30px; }

        .search-input {
            width: 100%; padding: 18px 28px; background: rgba(14, 16, 42, 0.85);
            border: 1px solid var(--cyan); border-radius: 40px; color: #fff; font-size: 1.1rem;
            font-family: 'Rajdhani', sans-serif; font-weight: 600; outline: none;
            box-shadow: 0 0 25px rgba(0, 243, 255, 0.15), inset 0 0 15px rgba(0, 243, 255, 0.05);
            backdrop-filter: blur(15px); transition: all 0.4s ease;
        }
        .search-input:focus {
            box-shadow: 0 0 35px rgba(255, 0, 85, 0.3), inset 0 0 20px rgba(0, 243, 255, 0.1);
            border-color: var(--pink);
        }

        .dropdown-menu {
            position: absolute; top: 68px; left: 0; right: 0; z-index: 100;
            background: rgba(8, 10, 25, 0.95); border: 1px solid var(--cyan);
            border-radius: 20px; max-height: 320px; overflow-y: auto; display: none;
            backdrop-filter: blur(25px); box-shadow: 0 15px 40px rgba(0,0,0,0.8);
        }

        .dropdown-item {
            padding: 14px 25px; border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            cursor: pointer; transition: all 0.2s ease; display: flex; justify-content: space-between; align-items: center;
        }
        .dropdown-item:hover { background: rgba(0, 243, 255, 0.15); transform: translateX(5px); }

        .item-action-btn {
            background: rgba(0, 243, 255, 0.15); border: 1px solid var(--cyan); color: var(--cyan);
            padding: 4px 12px; border-radius: 12px; font-family: 'Orbitron', sans-serif; font-size: 0.7rem;
            transition: all 0.2s;
        }
        .dropdown-item:hover .item-action-btn { background: var(--cyan); color: #000; font-weight: 800; }

        /* Presets Grid */
        .preset-container {
            display: flex; justify-content: center; gap: 14px; flex-wrap: wrap; margin-bottom: 35px;
        }

        .btn-preset {
            background: rgba(14, 16, 40, 0.7); border: 1px solid rgba(0, 243, 255, 0.3); color: #d1d5db;
            padding: 10px 22px; border-radius: 30px; font-family: 'Orbitron', sans-serif; font-size: 0.8rem;
            font-weight: 700; cursor: pointer; backdrop-filter: blur(10px); transition: all 0.3s ease;
        }
        .btn-preset:hover {
            background: var(--cyan); color: #000; border-color: var(--cyan);
            box-shadow: 0 0 20px var(--cyan); transform: translateY(-3px);
        }

        /* Layout Grid */
        .layout-grid { display: grid; grid-template-columns: 370px 1fr; gap: 30px; }

        /* Control Panel HUD */
        .control-panel {
            background: var(--glass-bg); border: 1px solid var(--border-neon);
            border-radius: 24px; padding: 28px; backdrop-filter: blur(20px);
            box-shadow: 0 0 40px rgba(0,0,0,0.6); position: sticky; top: 25px; height: fit-content;
        }

        .panel-heading {
            font-family: 'Orbitron', sans-serif; color: var(--cyan); font-size: 1.15rem; font-weight: 800;
            border-bottom: 1px solid var(--border-neon); padding-bottom: 12px; margin-bottom: 24px;
            display: flex; justify-content: space-between; align-items: center;
        }

        .slider-box { margin-bottom: 22px; }
        .slider-box label {
            display: flex; justify-content: space-between; font-weight: 700; font-size: 0.95rem;
            color: #e2e8f0; margin-bottom: 8px; letter-spacing: 1px;
        }
        .slider-box label span { color: var(--pink); font-family: 'Share Tech Mono', monospace; font-size: 1rem; }

        input[type=range] { width: 100%; height: 6px; appearance: none; background: #161838; border-radius: 4px; outline: none; }
        input[type=range]::-webkit-slider-thumb {
            appearance: none; width: 20px; height: 20px; border-radius: 50%;
            background: var(--cyan); cursor: pointer; box-shadow: 0 0 15px var(--cyan); transition: all 0.2s;
        }
        input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.3); background: var(--pink); box-shadow: 0 0 20px var(--pink); }

        .btn-synth {
            width: 100%; padding: 16px; font-family: 'Orbitron', sans-serif; font-weight: 900;
            font-size: 1rem; letter-spacing: 2px; color: #03030b;
            background: linear-gradient(90deg, var(--cyan), var(--pink));
            border: none; border-radius: 14px; cursor: pointer; transition: all 0.3s ease; margin-bottom: 14px;
            box-shadow: 0 0 25px rgba(0, 243, 255, 0.4);
        }
        .btn-synth:hover { transform: scale(1.02); box-shadow: 0 0 35px rgba(255, 0, 85, 0.7); }

        .btn-save-spotify {
            width: 100%; padding: 16px; font-family: 'Orbitron', sans-serif; font-weight: 800;
            font-size: 0.95rem; letter-spacing: 1px; color: #fff; background: var(--green);
            border: none; border-radius: 14px; cursor: pointer; box-shadow: 0 0 20px rgba(29, 185, 84, 0.4);
            transition: all 0.3s ease; display: flex; justify-content: center; align-items: center; gap: 10px;
        }
        .btn-save-spotify:hover { transform: scale(1.02); box-shadow: 0 0 30px var(--green); }

        /* Recommendations Cards Grid */
        .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 24px; }

        .song-card {
            background: var(--glass-bg); border: 1px solid rgba(0, 243, 255, 0.2);
            border-radius: 20px; padding: 20px; position: relative; backdrop-filter: blur(15px);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: fadeInCard 0.5s ease-out forwards; opacity: 0; transform: translateY(20px);
        }
        @keyframes fadeInCard { to { opacity: 1; transform: translateY(0); } }

        .song-card:hover {
            border-color: var(--cyan); transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0, 243, 255, 0.25);
        }

        .match-badge {
            position: absolute; top: 16px; right: 16px; background: rgba(0, 243, 255, 0.15);
            border: 1px solid var(--cyan); color: var(--cyan); padding: 5px 12px;
            border-radius: 20px; font-family: 'Orbitron', sans-serif; font-size: 0.75rem; font-weight: 800;
        }

        .song-title {
            font-family: 'Orbitron', sans-serif; font-size: 1.05rem; font-weight: 700; color: #fff;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 80px; margin-bottom: 6px;
        }

        .artist-name { color: var(--pink); font-weight: 700; font-size: 0.95rem; margin-bottom: 12px; }

        .play-trigger-btn {
            width: 100%; padding: 10px; background: rgba(0, 243, 255, 0.1); border: 1px solid var(--cyan);
            border-radius: 12px; color: var(--cyan); font-family: 'Orbitron', sans-serif; font-weight: 700;
            font-size: 0.8rem; cursor: pointer; transition: all 0.3s ease; display: flex; align-items: center; justify-content: center; gap: 8px;
            margin-top: 10px;
        }
        .play-trigger-btn:hover { background: var(--cyan); color: #000; box-shadow: 0 0 15px var(--cyan); }

        /* Floating Cyber Player Bar */
        .player-bar {
            position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
            width: 90%; max-width: 1000px; background: rgba(8, 10, 26, 0.92);
            border: 1px solid var(--cyan); border-radius: 25px; padding: 12px 25px;
            display: flex; align-items: center; justify-content: space-between; gap: 20px;
            backdrop-filter: blur(25px); box-shadow: 0 0 40px rgba(0, 243, 255, 0.3);
            z-index: 1000; transition: all 0.4s ease;
        }

        .player-track-info { display: flex; align-items: center; gap: 15px; width: 30%; }
        .track-icon { width: 42px; height: 42px; background: var(--pink); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; animation: rotateDisc 4s linear infinite; }
        @keyframes rotateDisc { 100% { transform: rotate(360deg); } }

        .player-details h4 { font-family: 'Orbitron', sans-serif; font-size: 0.95rem; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .player-details p { font-size: 0.8rem; color: var(--cyan); }

        .player-iframe-box { width: 65%; height: 80px; border-radius: 12px; overflow: hidden; }
        .player-iframe-box iframe { width: 100%; height: 100%; border: none; }
    </style>
</head>
<body>

    <canvas id="visualizer-canvas"></canvas>

    <div class="wrapper">
        <!-- Top HUD Header -->
        <header>
            <div class="brand-box">
                <div class="brand-logo">M</div>
                <div>
                    <h1>MOODTUBE AI</h1>
                    <div class="sys-status">
                        <span class="status-dot"></span> NEURAL MATRIX ACTIVE // 8,582 TRACKS
                    </div>
                </div>
            </div>
            <div>
                {% if user_display_name %}
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span style="color:var(--green); font-family:'Orbitron'; font-size:0.85rem; border:1px solid var(--green); padding:6px 16px; border-radius:20px;">🟢 {{ user_display_name }}</span>
                        <a href="/logout" style="color:var(--pink); text-decoration:none; font-size:0.85rem; font-weight:700;">LOGOUT</a>
                    </div>
                {% else %}
                    <a href="/login" class="btn-spotify-login">⚡ CONNECT SPOTIFY</a>
                {% endif %}
            </div>
        </header>

        <!-- Search Engine HUD -->
        <div class="search-container">
            <input type="text" id="track-search" class="search-input" placeholder="🔍 Search track or artist in library (e.g. Diplo, Yelawolf, Riff Raff)..." autocomplete="off">
            <div id="search-dropdown" class="dropdown-menu"></div>
        </div>

        <!-- Preset Mood Matrix -->
        <div class="preset-container">
            <button class="btn-preset" onclick="setPreset(0.95, 0.85, 0.80, 0.05, 145)">🔥 CYBER HYPE</button>
            <button class="btn-preset" onclick="setPreset(0.30, 0.20, 0.40, 0.70, 85)">🌙 NEON SADNESS</button>
            <button class="btn-preset" onclick="setPreset(0.50, 0.45, 0.45, 0.50, 105)">🧠 DEEP FOCUS</button>
            <button class="btn-preset" onclick="setPreset(0.85, 0.25, 0.75, 0.10, 140)">⚡ DARK SYNTH</button>
            <button class="btn-preset" onclick="setPreset(0.70, 0.90, 0.85, 0.20, 120)">✨ EUPHORIA</button>
        </div>

        <!-- Main Workspace Grid -->
        <div class="layout-grid">
            <!-- Controls Sidebar -->
            <div class="control-panel">
                <div class="panel-heading">
                    <span>ACOUSTIC MATRIX</span>
                    <span style="font-size:0.8rem; color:var(--pink);">KNN V2.5</span>
                </div>

                <div class="slider-box">
                    <label>ENERGY <span id="val-energy">85%</span></label>
                    <input type="range" id="energy" min="0" max="100" value="85">
                </div>

                <div class="slider-box">
                    <label>VALENCE (HAPPY/SAD) <span id="val-valence">65%</span></label>
                    <input type="range" id="valence" min="0" max="100" value="65">
                </div>

                <div class="slider-box">
                    <label>DANCEABILITY <span id="val-dance">80%</span></label>
                    <input type="range" id="danceability" min="0" max="100" value="80">
                </div>

                <div class="slider-box">
                    <label>ACOUSTICNESS <span id="val-acoustic">20%</span></label>
                    <input type="range" id="acousticness" min="0" max="100" value="20">
                </div>

                <div class="slider-box">
                    <label>TEMPO (BPM) <span id="val-tempo">135</span></label>
                    <input type="range" id="tempo" min="60" max="180" value="135">
                </div>

                <button class="btn-synth" onclick="fetchRecommendations()">RUN NEURAL SEARCH</button>
                <button class="btn-save-spotify" onclick="exportToSpotify()">💾 EXPORT TO SPOTIFY</button>
            </div>

            <!-- Output Cards Grid -->
            <div class="results-grid" id="results"></div>
        </div>
    </div>

    <!-- Active Floating Audio Player Bar -->
    <div class="player-bar" id="player-bar">
        <div class="player-track-info">
            <div class="track-icon">🎵</div>
            <div class="player-details">
                <h4 id="player-title">Select a Track</h4>
                <p id="player-artist">MoodTube Cyber Engine</p>
            </div>
        </div>
        <div class="player-iframe-box" id="player-iframe-box">
            <iframe id="active-spotify-iframe" src="https://open.spotify.com/embed/track/3EJS5LyekDim1Tf5rBFmZl?utm_source=generator&theme=0" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>
        </div>
    </div>

    <script>
        let currentTrackIds = [];
        let searchDebounce = null;

        // Search Engine Autocomplete
        const searchInput = document.getElementById('track-search');
        const searchDropdown = document.getElementById('search-dropdown');

        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchDebounce);
            const query = e.target.value.trim();
            if (query.length < 2) { searchDropdown.style.display = 'none'; return; }

            searchDebounce = setTimeout(async () => {
                const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
                const matches = await response.json();

                if (!matches.length) {
                    searchDropdown.innerHTML = `<div class="dropdown-item">No matching tracks found</div>`;
                } else {
                    searchDropdown.innerHTML = matches.map(m => `
                        <div class="dropdown-item" onclick="selectTrack(${m.energy}, ${m.valence}, ${m.danceability}, ${m.acousticness}, ${m.tempo}, '${escapeHtml(m.track_name)}', '${m.track_id}', '${escapeHtml(m.artist_name)}')">
                            <div>
                                <strong style="color:#fff;">${escapeHtml(m.track_name)}</strong>
                                <span style="color:#8c9eb4; font-size:0.85rem; margin-left:8px;">by ${escapeHtml(m.artist_name)}</span>
                            </div>
                            <button class="item-action-btn">▶ SEED & PLAY</button>
                        </div>
                    `).join('');
                }
                searchDropdown.style.display = 'block';
            }, 250);
        });

        function escapeHtml(str) { return str.replace(/'/g, "\\'").replace(/"/g, "&quot;"); }

        function selectTrack(energy, valence, dance, acoustic, tempo, trackName, trackId, artistName) {
            searchDropdown.style.display = 'none';
            searchInput.value = `Seed: ${trackName}`;
            setPreset(energy, valence, dance, acoustic, tempo);
            playTrack(trackId, trackName, artistName);
        }

        function playTrack(trackId, title, artist) {
            document.getElementById('player-title').innerText = title;
            document.getElementById('player-artist').innerText = artist;
            document.getElementById('active-spotify-iframe').src = `https://open.spotify.com/embed/track/${trackId}?utm_source=generator&theme=0&autoplay=1`;
        }

        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !searchDropdown.contains(e.target)) {
                searchDropdown.style.display = 'none';
            }
        });

        // Sliders & Presets
        const sliders = ['energy', 'valence', 'danceability', 'acousticness', 'tempo'];
        sliders.forEach(id => {
            const el = document.getElementById(id);
            el.addEventListener('input', () => {
                const label = document.getElementById('val-' + (id === 'danceability' ? 'dance' : id === 'acousticness' ? 'acoustic' : id));
                label.innerText = id === 'tempo' ? el.value : el.value + '%';
            });
        });

        function setPreset(energy, valence, dance, acoustic, tempo) {
            document.getElementById('energy').value = Math.round(energy * 100);
            document.getElementById('valence').value = Math.round(valence * 100);
            document.getElementById('danceability').value = Math.round(dance * 100);
            document.getElementById('acousticness').value = Math.round(acoustic * 100);
            document.getElementById('tempo').value = Math.round(tempo);

            document.getElementById('val-energy').innerText = Math.round(energy * 100) + '%';
            document.getElementById('val-valence').innerText = Math.round(valence * 100) + '%';
            document.getElementById('val-dance').innerText = Math.round(dance * 100) + '%';
            document.getElementById('val-acoustic').innerText = Math.round(acoustic * 100) + '%';
            document.getElementById('val-tempo').innerText = Math.round(tempo);

            fetchRecommendations();
        }

        async function fetchRecommendations() {
            const payload = {
                energy: parseFloat(document.getElementById('energy').value) / 100,
                valence: parseFloat(document.getElementById('valence').value) / 100,
                danceability: parseFloat(document.getElementById('danceability').value) / 100,
                acousticness: parseFloat(document.getElementById('acousticness').value) / 100,
                tempo: parseFloat(document.getElementById('tempo').value)
            };

            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const songs = await response.json();
            currentTrackIds = songs.map(s => s.track_id);
            const grid = document.getElementById('results');

            grid.innerHTML = songs.map((song, i) => `
                <div class="song-card" style="animation-delay: ${i * 0.08}s;">
                    <div class="match-badge">${song.match}% MATCH</div>
                    <div class="song-title">${song.track_name}</div>
                    <div class="artist-name">${song.artist_name}</div>
                    <div style="font-size:0.8rem; color:#8c9eb4; margin-bottom:8px;">GENRE: ${song.genres}</div>
                    <button class="play-trigger-btn" onclick="playTrack('${song.track_id}', '${escapeHtml(song.track_name)}', '${escapeHtml(song.artist_name)}')">
                        ▶ PLAY IN PLAYER HUD
                    </button>
                </div>
            `).join('');

            if (songs.length > 0) {
                playTrack(songs[0].track_id, songs[0].track_name, songs[0].artist_name);
            }
        }

        async function exportToSpotify() {
            if (!currentTrackIds.length) { alert("Please generate recommendations first!"); return; }

            const response = await fetch('/create_playlist', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ track_ids: currentTrackIds })
            });

            const data = await response.json();
            if (data.status === 'unauthorized') {
                alert("Please click 'CONNECT SPOTIFY' at the top to log in!");
                window.location.href = '/login';
            } else if (data.status === 'success') {
                alert(`🎉 Playlist "${data.name}" created on your Spotify account!`);
                window.open(data.url, '_blank');
            } else { alert("Error: " + data.error); }
        }

        // Animated Audio Spectrum Canvas Visualizer
        const canvas = document.getElementById('visualizer-canvas');
        const ctx = canvas.getContext('2d');
        let w, h, bars = 70, audioTick = 0;

        function resize() { w = canvas.width = window.innerWidth; h = canvas.height = window.innerHeight; }
        window.addEventListener('resize', resize);
        resize();

        function drawVisualizer() {
            ctx.clearRect(0, 0, w, h);
            audioTick += 0.04;
            const barWidth = w / bars;

            for (let i = 0; i < bars; i++) {
                const heightVal = Math.abs(Math.sin(audioTick + i * 0.18) * Math.cos(audioTick * 0.4 + i * 0.08)) * (h * 0.35);
                const x = i * barWidth;
                const y = h - heightVal;

                const grad = ctx.createLinearGradient(0, h, 0, y);
                grad.addColorStop(0, 'rgba(0, 243, 255, 0.02)');
                grad.addColorStop(0.6, 'rgba(0, 243, 255, 0.25)');
                grad.addColorStop(1, 'rgba(255, 0, 85, 0.7)');

                ctx.fillStyle = grad;
                ctx.shadowBlur = 12;
                ctx.shadowColor = '#00f3ff';
                ctx.fillRect(x, y, barWidth - 4, heightVal);
            }
            requestAnimationFrame(drawVisualizer);
        }
        drawVisualizer();
        fetchRecommendations();
    </script>
</body>
</html>
"""

# -------------------------------------------------------------------
# 3. FLASK ROUTES
# -------------------------------------------------------------------
@app.route('/')
def home():
    sp_oauth = create_spotify_oauth()
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    user_display_name = None

    if token_info:
        sp = spotipy.Spotify(auth=token_info['access_token'])
        try:
            user_profile = sp.current_user()
            user_display_name = user_profile.get('display_name', 'Spotify User')
        except Exception:
            pass

    return render_template_string(HTML_TEMPLATE, user_display_name=user_display_name)

@app.route('/search', methods=['GET'])
def search_tracks():
    query = request.args.get('q', '').lower().strip()
    if not query or len(query) < 2:
        return jsonify([])
    
    mask = df['track_name'].str.lower().str.contains(query, na=False) | \
           df['artist_name'].str.lower().str.contains(query, na=False)
    
    matches = df[mask].head(8)
    results = []
    for _, row in matches.iterrows():
        results.append({
            'track_id': str(row['track_id']),
            'track_name': str(row['track_name']),
            'artist_name': str(row['artist_name']),
            'energy': float(row['energy']),
            'valence': float(row['valence']),
            'danceability': float(row['danceability']),
            'acousticness': float(row['acousticness']),
            'tempo': float(row['tempo'])
        })
    return jsonify(results)

@app.route('/login')
def login():
    sp_oauth = create_spotify_oauth()
    return redirect(sp_oauth.get_authorize_url())

@app.route('/callback')
def callback():
    sp_oauth = create_spotify_oauth()
    session.clear()
    code = request.args.get('code')
    sp_oauth.get_access_token(code)
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    query_vector = np.array([
        float(data.get('energy', 0.8)),
        float(data.get('danceability', 0.8)),
        float(data.get('valence', 0.6)),
        float(data.get('acousticness', 0.2)),
        float(data.get('tempo', 135))
    ])
    results = compute_knn(query_vector, top_k=9)
    return jsonify(results)

@app.route('/create_playlist', methods=['POST'])
def create_playlist():
    sp_oauth = create_spotify_oauth()
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    
    if not token_info:
        return jsonify({'status': 'unauthorized'})

    data = request.json
    track_ids = data.get('track_ids', [])
    if not track_ids:
        return jsonify({'status': 'error', 'error': 'No track IDs provided'})

    sp = spotipy.Spotify(auth=token_info['access_token'])
    user_id = sp.current_user()['id']
    
    playlist_name = "MoodTube AI Matrix Mix"
    playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True, description="Generated with MoodTube AI KNN Recommender")
    
    track_uris = [f"spotify:track:{tid}" for tid in track_ids]
    sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)
    
    return jsonify({
        'status': 'success',
        'name': playlist_name,
        'url': playlist['external_urls']['spotify']
    })

if __name__ == '__main__':
    print("\n=======================================================")
    print("🚀 MOODTUBE AI (CYBER HUD PLAYBACK) LAUNCHED!")
    print("👉 Open your browser at: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(debug=True, port=5000)