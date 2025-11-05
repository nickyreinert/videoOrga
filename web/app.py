from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
import os
import json
import sqlite3
import humanize
from datetime import datetime
from pathlib import Path
from functools import lru_cache
import mmap
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration from environment or defaults
VIDEO_BASE_PATH = os.environ.get('VIDEO_BASE_PATH', '/videos')
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "video_metadata.db")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
THUMBNAIL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "thumbnails")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_video_path(relative_path, for_windows=False):
    """Convert relative path to absolute path based on VIDEO_BASE_PATH.

    This simplified helper returns a path under the configured
    `VIDEO_BASE_PATH` mount. For opening videos on the host the UI
    expects a path that points into the host-mounted folder.
    """
    return os.path.join(VIDEO_BASE_PATH, relative_path)

def check_file_exists(wsl_path):
    """Check if a file exists using WSL path"""
    try:
        return os.path.exists(wsl_path)
    except Exception:
        return False

@app.route('/')
def index():
    db = get_db()
    cursor = db.cursor()
    
    # Get all videos with their metadata (adapted to db_handler schema)
    cursor.execute("""
        SELECT v.id,
               v.file_name AS filename,
               v.file_path,
               v.file_created_date AS creation_date,
               v.duration_seconds AS duration,
               (
                   SELECT image_data FROM thumbnails th WHERE th.video_id = v.id LIMIT 1
               ) AS thumbnail_data,
               GROUP_CONCAT(t.tag) AS tags
        FROM videos v
        LEFT JOIN tags t ON v.id = t.video_id
        GROUP BY v.id
        ORDER BY v.file_created_date DESC
    """)
    videos = cursor.fetchall()
    
    processed_videos = []
    for video in videos:
        # Convert video data to dictionary
        video_data = dict(video)
        video_data['tags'] = video_data['tags'].split(',') if video_data['tags'] else []
        video_data['friendly_date'] = humanize.naturaldate(datetime.fromisoformat(video_data['creation_date']))
        video_data['friendly_duration'] = humanize.precisedelta(video_data['duration'])
        video_data['file_exists'] = os.path.exists(get_video_path(video_data['file_path']))
        processed_videos.append(video_data)

    return render_template('index.html', videos=processed_videos)

@app.route('/api/videos')
def get_videos():
    tags = request.args.getlist('tags')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    search = request.args.get('search')
    
    db = get_db()
    cursor = db.cursor()
    
    query = """
        SELECT DISTINCT v.id, v.file_name AS filename, v.file_path,
               v.file_created_date AS creation_date, v.duration_seconds AS duration
        FROM videos v
    """
    
    conditions = []
    params = []
    
    if tags:
        # Our tags table uses a per-video tag record (video_id, tag, confidence)
        placeholders = ','.join('?' * len(tags))
        conditions.append(f"t.tag IN ({placeholders})")
        params.extend(tags)
        query += " JOIN tags t ON v.id = t.video_id "
    
    if start_date:
        conditions.append("v.creation_date >= ?")
        params.append(start_date)
    
    if end_date:
        conditions.append("v.creation_date <= ?")
        params.append(end_date)
    
    if search:
        conditions.append("(v.filename LIKE ? OR v.file_path LIKE ?)")
        search_pattern = f"%{search}%"
        params.extend([search_pattern, search_pattern])
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY v.creation_date DESC"
    
    cursor.execute(query, params)
    videos = cursor.fetchall()
    
    return jsonify([dict(video) for video in videos])

@app.route('/api/tags')
def get_tags():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT t.tag as tag_name, COUNT(*) as count
        FROM tags t
        GROUP BY t.tag
        ORDER BY count DESC
    """)
    tags = cursor.fetchall()
    return jsonify([dict(tag) for tag in tags])

@app.route('/video/<path:video_path>')
def open_video(video_path):
    """Handle video opening through Windows default player"""
    # Get both WSL and Windows paths
    wsl_path = get_video_path(video_path, for_windows=False)
    windows_path = get_video_path(video_path, for_windows=True)
    
    if check_file_exists(wsl_path):
        try:
            # Use cmd.exe through WSL to open the file in Windows
            windows_path_escaped = windows_path.replace('/', '\\')
            cmd = f'cmd.exe /C start "" "{windows_path_escaped}"'
            subprocess.run(['wslpath', '-w', wsl_path], check=True, capture_output=True, text=True)
            subprocess.run(cmd, shell=True)
            return jsonify({"status": "opened", "path": windows_path})
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Error opening video: {str(e)}"}), 500
    else:
        return jsonify({"error": "Video file not found"}), 404

@app.route('/api/video/<int:video_id>/tags', methods=['POST'])
def update_tags(video_id):
    data = request.get_json()
    new_tags = data.get('tags', [])
    
    db = get_db()
    cursor = db.cursor()
    
    try:
        # Remove existing tags for this video
        cursor.execute("DELETE FROM tags WHERE video_id = ?", (video_id,))

        # Insert new tags (tags table stores tag per video in this schema)
        for tag in new_tags:
            cursor.execute(
                "INSERT INTO tags (video_id, tag, confidence) VALUES (?, ?, ?)",
                (video_id, tag, 1.0)
            )

        db.commit()
        return jsonify({"status": "success", "tags": new_tags})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

if __name__ == '__main__':
    # Bind to 0.0.0.0 so the Flask dev server is reachable from outside the container
    app.run(host='0.0.0.0', debug=True, port=5000)