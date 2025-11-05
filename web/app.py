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

def windows_to_wsl_path(windows_path):
    """Convert Windows path to WSL path"""
    # Convert to PureWindowsPath to handle Windows-style paths
    win_path = PureWindowsPath(windows_path)
    # Get drive letter and convert to lowercase
    drive = win_path.drive.lower().rstrip(':')
    # Convert path to posix and combine with WSL mount point
    rest_of_path = str(win_path).replace(win_path.drive, '').replace('\\', '/')
    return f"/mnt/{drive}{rest_of_path}"

def wsl_to_windows_path(wsl_path):
    """Convert WSL path to Windows path"""
    if wsl_path.startswith('/mnt/'):
        # Extract drive letter and rest of path
        parts = wsl_path.split('/')
        drive = parts[2].upper()
        rest_of_path = '/'.join(parts[3:])
        return f"{drive}:/{rest_of_path}"
    return wsl_path

def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_video_path(relative_path, for_windows=False):
    """Convert relative path to absolute path in either WSL or Windows format"""
    if for_windows:
        return os.path.join(WINDOWS_VIDEO_PATH, relative_path)
    return os.path.join(WSL_VIDEO_PATH, relative_path)

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
    
    # Get all videos with their metadata
    cursor.execute("""
        SELECT v.id, v.filename, v.file_path, v.creation_date, v.duration,
               v.thumbnail_data, GROUP_CONCAT(t.tag_name) as tags
        FROM videos v
        LEFT JOIN video_tags vt ON v.id = vt.video_id
        LEFT JOIN tags t ON vt.tag_id = t.id
        GROUP BY v.id
        ORDER BY v.creation_date DESC
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
        SELECT DISTINCT v.id, v.filename, v.file_path, v.creation_date, 
               v.duration, v.thumbnail_data
        FROM videos v
    """
    
    conditions = []
    params = []
    
    if tags:
        query += """
            JOIN video_tags vt ON v.id = vt.video_id
            JOIN tags t ON vt.tag_id = t.id
        """
        placeholders = ','.join('?' * len(tags))
        conditions.append(f"t.tag_name IN ({placeholders})")
        params.extend(tags)
    
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
        SELECT t.tag_name, COUNT(vt.video_id) as count
        FROM tags t
        LEFT JOIN video_tags vt ON t.id = vt.tag_id
        GROUP BY t.tag_name
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
        # Remove existing tags
        cursor.execute("DELETE FROM video_tags WHERE video_id = ?", (video_id,))
        
        # Add new tags
        for tag in new_tags:
            # First ensure the tag exists
            cursor.execute(
                "INSERT OR IGNORE INTO tags (tag_name) VALUES (?)",
                (tag,)
            )
            cursor.execute(
                """
                INSERT INTO video_tags (video_id, tag_id)
                SELECT ?, id FROM tags WHERE tag_name = ?
                """,
                (video_id, tag)
            )
        
        db.commit()
        return jsonify({"status": "success", "tags": new_tags})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

if __name__ == '__main__':
    app.run(debug=True, port=5000)