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
import re

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

    # Get query parameters
    tags = request.args.getlist('tags')
    search_text = request.args.get('search')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Build the query dynamically
    query = "SELECT DISTINCT v.* FROM videos v"
    conditions = []
    params = []

    if tags:
        query += " JOIN tags t ON v.id = t.video_id"
        # Ensure that videos have ALL specified tags
        for tag in tags:
            conditions.append("v.id IN (SELECT video_id FROM tags WHERE tag = ?)")
            params.append(tag)

    if search_text:
        conditions.append("(v.file_name LIKE ? OR v.transcript LIKE ?)")
        params.extend([f"%{search_text}%", f"%{search_text}%"])

    if start_date:
        conditions.append("v.parsed_datetime >= ?")
        params.append(start_date)

    if end_date:
        conditions.append("v.parsed_datetime <= ?")
        params.append(end_date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY v.parsed_datetime DESC, v.file_modified_date DESC"
    cursor.execute(query, params)
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

    # The path in the DB is absolute from the scanner's perspective. We need to find
    # its path relative to the directory where the DB is located, which is assumed
    # to be the root of the scanned collection.
    db_directory = os.path.dirname(os.path.abspath(DATABASE))
    try:
        # Make the DB path relative to the DB's directory
        relative_path = os.path.relpath(video['file_path'], db_directory)
    except ValueError:
        # This can happen on Windows if the file path is on a different drive
        # than the database. In this case, we fall back to just using the filename.
        relative_path = Path(video['file_path']).name

    # Join the relative path with the web server's video base path.
    video_relative_path = relative_path
    video_path = os.path.join(VIDEO_BASE_PATH, video_relative_path)

    if not os.path.exists(video_path):
        return f"Video file not found on disk at: {video_path}", 404

    range_header = request.headers.get('Range', None)
    size = os.path.getsize(video_path)

    if not range_header:
        # If no range header, send the whole file
        def generate_full():
            with open(video_path, 'rb') as f:
                yield from f
        return Response(generate_full(), mimetype="video/mp4", headers={"Content-Length": str(size)})
        # If no range header, send the initial part of the file.
        # Browsers will then make subsequent range requests.
        headers = {"Content-Length": str(size), "Accept-Ranges": "bytes"}
        return Response(None, 200, mimetype="video/mp4", headers=headers)

    # Handle range request
    byte1, byte2 = 0, None
    try:
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()

        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])
    except (AttributeError, IndexError):
        return "Invalid Range header", 400

    if byte2 is None or byte2 >= size:
        byte2 = size - 1

    length = byte2 - byte1 + 1
    
    def generate_partial():
        """Generator to stream a chunk of the file."""
        with open(video_path, 'rb') as f:
            f.seek(byte1)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, 65536)) # Read in 64KB chunks
                if not chunk:
                    break
                yield chunk
                remaining -= len(chunk)

    rv = Response(generate_partial(), 206, mimetype="video/mp4")
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv

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