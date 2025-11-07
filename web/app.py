from flask import Flask, render_template, request, jsonify, send_file, Response
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

# --- Database Setup ---
DATABASE = os.environ.get('VIDEO_DB_PATH', './data/video_metadata.db')
VIDEO_BASE_PATH = os.environ.get('VIDEO_BASE_PATH', '/videos')

def get_db():
    """Get a database connection."""
    db_path = os.path.join(os.path.dirname(__file__), '..', DATABASE)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    return db

# --- End Database Setup ---

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Serve the main index.html file
@app.route('/')
def index():
    return render_template('index.html')


# --- API Endpoints ---

@app.route('/api/tags', methods=['GET'])
def get_tags():
    """Get all unique tags with their counts."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT tag, COUNT(*) as count FROM tags GROUP BY tag ORDER BY count DESC, tag ASC")
    tags = cursor.fetchall()
    db.close()
    # Convert to list of dicts for JSON
    tag_list = [{"tag_name": row['tag'], "count": row['count']} for row in tags]
    return jsonify(tag_list)

@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Get videos with filtering and pagination."""
    db = get_db()
    cursor = db.cursor()

    # Basic query
    query = "SELECT * FROM videos ORDER BY parsed_datetime DESC, file_modified_date DESC"
    cursor.execute(query)
    videos = [dict(row) for row in cursor.fetchall()]

    # Add tags and thumbnails to each video
    for video in videos:
        cursor.execute("SELECT tag FROM tags WHERE video_id = ?", (video['id'],))
        video['tags'] = [row['tag'] for row in cursor.fetchall()]
        
        cursor.execute("SELECT image_data FROM thumbnails WHERE video_id = ? ORDER BY thumbnail_index", (video['id'],))
        video['thumbnail_data'] = [row['image_data'] for row in cursor.fetchall()]

    db.close()
    return jsonify(videos)


@app.route('/api/video/<int:video_id>/tags', methods=['POST'])
def update_video_tags(video_id):
    """Update tags for a specific video."""
    data = request.get_json()
    tags = data.get('tags', [])

    if not isinstance(tags, list):
        return jsonify({"error": "Tags must be a list"}), 400

    try:
        db = get_db()
        cursor = db.cursor()
        # Clear existing tags
        cursor.execute("DELETE FROM tags WHERE video_id = ?", (video_id,))
        # Insert new tags
        if tags:
            cursor.executemany("INSERT INTO tags (video_id, tag) VALUES (?, ?)", [(video_id, tag) for tag in tags])
        db.commit()
        db.close()
        return jsonify({"message": "Tags updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- End API Endpoints ---
@app.route('/api/video/<int:video_id>', methods=['GET'])
def get_video_details(video_id):
    """Get full details for a single video."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
    video = cursor.fetchone()
    if not video:
        return jsonify({"error": "Video not found"}), 404

    video_dict = dict(video)
    cursor.execute("SELECT tag FROM tags WHERE video_id = ?", (video_id,))
    video_dict['tags'] = [row['tag'] for row in cursor.fetchall()]
    
    cursor.execute("SELECT image_data FROM thumbnails WHERE video_id = ? ORDER BY thumbnail_index", (video_id,))
    video_dict['thumbnail_data'] = [row['image_data'] for row in cursor.fetchall()]
    db.close()
    return jsonify(video_dict)

@app.route('/video/stream/<int:video_id>')
def stream_video(video_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT file_path FROM videos WHERE id = ?", (video_id,))
    video = cursor.fetchone()
    db.close()

    if not video:
        return "Video not found", 404

    video_path = os.path.join(VIDEO_BASE_PATH, video['file_path'])

    if not os.path.exists(video_path):
        return "Video file not found on disk", 404

    def generate():
        with open(video_path, "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                yield chunk

    return Response(generate(), mimetype="video/mp4")

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