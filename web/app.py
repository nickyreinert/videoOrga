from flask import Flask, render_template, request, jsonify, send_file
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
    # Increase the group_concat_max_len to avoid truncated thumbnail data
    conn.execute("PRAGMA group_concat_max_len = 1000000")
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

    return render_template('index.html')

@app.route('/api/videos')
def get_videos():
    tags = request.args.getlist('tags')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    search = request.args.get('search')
    
    db = get_db()
    cursor = db.cursor()
    
    query = """
        SELECT v.id,
               v.file_name AS filename,
               v.file_path,
               v.file_created_date AS creation_date,
               v.duration_seconds AS duration,
               (
                   SELECT GROUP_CONCAT(image_data) FROM thumbnails th WHERE th.video_id = v.id
               ) AS thumbnail_data,
               GROUP_CONCAT(t.tag) AS tags
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
    
    query += " GROUP BY v.id ORDER BY v.file_created_date DESC"
    
    cursor.execute(query, params)
    videos = cursor.fetchall()
    
    processed_videos = []
    for video in videos:
        video_data = dict(video)
        video_data['tags'] = video_data['tags'].split(',') if video_data['tags'] else []
        video_data['thumbnail_data'] = video_data['thumbnail_data'].split(',') if video_data['thumbnail_data'] else []
        processed_videos.append(video_data)
    
    print(processed_videos)
    return jsonify(processed_videos)

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
    """Handle video opening through the host's default player"""
    if not VIDEO_BASE_PATH or not os.path.isdir(VIDEO_BASE_PATH):
        return jsonify({"error": "VIDEO_BASE_PATH is not configured or not a directory."}), 500

    # Construct the absolute path to the video file
    abs_video_path = os.path.join(VIDEO_BASE_PATH, video_path)

    if os.path.exists(abs_video_path):
        try:
            # Open the video file using the system's default player
            os.startfile(abs_video_path)
            return jsonify({"status": "opened", "path": abs_video_path})
        except Exception as e:
            return jsonify({"error": f"Error opening video: {str(e)}"}), 500
    else:
        return jsonify({"error": "Video file not found"}), 404

@app.route('/api/video/<int:video_id>', methods=['PUT'])
def update_video_details(video_id):
    data = request.get_json()
    new_filename = data.get('filename')
    new_tags = data.get('tags', [])

    db = get_db()
    cursor = db.cursor()

    try:
        # Update filename
        if new_filename:
            cursor.execute("UPDATE videos SET file_name = ? WHERE id = ?", (new_filename, video_id))

        # Update tags
        cursor.execute("DELETE FROM tags WHERE video_id = ?", (video_id,))
        for tag in new_tags:
            cursor.execute(
                "INSERT INTO tags (video_id, tag, confidence) VALUES (?, ?, ?)",
                (video_id, tag, 1.0)
            )

        db.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

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

@app.route('/api/video/<int:video_id>', methods=['DELETE'])
def delete_video(video_id):
    db = get_db()
    cursor = db.cursor()
    
    try:
        # Delete associated tags
        cursor.execute("DELETE FROM tags WHERE video_id = ?", (video_id,))
        
        # Delete the video
        cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        
        db.commit()
        return jsonify({"status": "success", "message": "Video deleted"})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route('/api/video/<int:video_id>/rename', methods=['POST'])
def rename_video_file(video_id):
    data = request.get_json()
    new_filename = data.get('new_filename')

    if not new_filename:
        return jsonify({"error": "New filename not provided"}), 400

    db = get_db()
    cursor = db.cursor()

    try:
        # Get the current file path
        cursor.execute("SELECT file_path FROM videos WHERE id = ?", (video_id,))
        video = cursor.fetchone()
        if not video:
            return jsonify({"error": "Video not found"}), 404

        current_path = os.path.join(VIDEO_BASE_PATH, video['file_path'])
        current_dir = os.path.dirname(current_path)
        new_path = os.path.join(current_dir, new_filename)

        # Rename the file
        os.rename(current_path, new_path)

        # Update the database
        new_relative_path = os.path.relpath(new_path, VIDEO_BASE_PATH)
        cursor.execute("UPDATE videos SET file_name = ?, file_path = ? WHERE id = ?", (new_filename, new_relative_path, video_id))
        db.commit()

        return jsonify({"status": "success", "new_path": new_relative_path})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

if __name__ == '__main__':
    if not VIDEO_BASE_PATH or not os.path.isdir(VIDEO_BASE_PATH):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: VIDEO_BASE_PATH is not set or not a directory. !!!")
        print("!!! You will not be able to open videos.                    !!!")
        print("!!! Set the VIDEO_BASE_PATH environment variable to the     !!!")
        print("!!! absolute path of your video directory.                  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Bind to 0.0.0.0 so the Flask dev server is reachable from outside the container
    app.run(host='0.0.0.0', debug=True, port=5000)