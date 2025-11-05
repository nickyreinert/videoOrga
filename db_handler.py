"""
SQLite Database Handler for Video Metadata
Stores video metadata, tags, and thumbnail previews
"""

import sqlite3
import os
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib


class DatabaseHandler:
    """Handles all database operations for video metadata"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database handler
        
        Args:
            db_path: Path to SQLite database. If None, creates 'video_archive.db'
        """
        if db_path is None:
            db_path = "video_archive.db"
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection and create tables"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.cursor = self.conn.cursor()
        
        self._create_tables()
        print(f"Database initialized: {self.db_path}")
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        
        # Main videos table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size_bytes INTEGER,
                file_hash TEXT,
                
                -- File timestamps
                file_created_date TEXT,
                file_modified_date TEXT,
                
                -- Parsed datetime from filename
                parsed_datetime TEXT,
                
                -- Video properties
                duration_seconds REAL,
                fps REAL,
                width INTEGER,
                height INTEGER,
                resolution TEXT,
                codec TEXT,
                
                -- Processing info
                processed_date TEXT NOT NULL,
                frames_analyzed INTEGER,
                
                -- Metadata
                description TEXT,
                notes TEXT,
                
                -- Timestamps
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tags table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                UNIQUE(video_id, tag)
            )
        """)
        
        # Frame descriptions table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS frame_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                description TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            )
        """)
        
        # Thumbnails table (stores 3 random frames as base64)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS thumbnails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                thumbnail_index INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                image_data TEXT NOT NULL,  -- base64 encoded image
                width INTEGER,
                height INTEGER,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_video_id ON tags(video_id)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_videos_parsed_datetime ON videos(parsed_datetime)
        """)
        
        self.conn.commit()
    
    def video_exists(self, file_path: str) -> bool:
        """Check if video is already in database"""
        self.cursor.execute(
            "SELECT id FROM videos WHERE file_path = ?",
            (file_path,)
        )
        return self.cursor.fetchone() is not None
    
    def get_video_id(self, file_path: str) -> Optional[int]:
        """Get video ID by file path"""
        self.cursor.execute(
            "SELECT id FROM videos WHERE file_path = ?",
            (file_path,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def insert_video(self, video_data: Dict) -> int:
        """
        Insert or update video metadata
        
        Args:
            video_data: Dictionary with video metadata
            
        Returns:
            Video ID
        """
        # Check if video exists
        video_id = self.get_video_id(video_data['file_path'])
        
        if video_id:
            # Update existing record
            self._update_video(video_id, video_data)
        else:
            # Insert new record
            video_id = self._insert_new_video(video_data)
        
        return video_id
    
    def _insert_new_video(self, data: Dict) -> int:
        """Insert new video record"""
        self.cursor.execute("""
            INSERT INTO videos (
                file_path, file_name, file_size_bytes, file_hash,
                file_created_date, file_modified_date, parsed_datetime,
                duration_seconds, fps, width, height, resolution, codec,
                processed_date, frames_analyzed, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('file_path'),
            data.get('file_name'),
            data.get('file_size_bytes'),
            data.get('file_hash'),
            data.get('file_created_date'),
            data.get('file_modified_date'),
            data.get('parsed_datetime'),
            data.get('duration_seconds'),
            data.get('fps'),
            data.get('width'),
            data.get('height'),
            data.get('resolution'),
            data.get('codec'),
            data.get('processed_date'),
            data.get('frames_analyzed'),
            data.get('description')
        ))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def _update_video(self, video_id: int, data: Dict):
        """Update existing video record"""
        self.cursor.execute("""
            UPDATE videos SET
                file_size_bytes = ?,
                file_modified_date = ?,
                duration_seconds = ?,
                fps = ?,
                width = ?,
                height = ?,
                resolution = ?,
                processed_date = ?,
                frames_analyzed = ?,
                description = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            data.get('file_size_bytes'),
            data.get('file_modified_date'),
            data.get('duration_seconds'),
            data.get('fps'),
            data.get('width'),
            data.get('height'),
            data.get('resolution'),
            data.get('processed_date'),
            data.get('frames_analyzed'),
            data.get('description'),
            video_id
        ))
        self.conn.commit()
    
    def insert_tags(self, video_id: int, tags: List[str], confidences: Dict = None):
        """
        Insert tags for a video
        
        Args:
            video_id: Video ID
            tags: List of tag strings
            confidences: Optional dict of tag -> confidence score
        """
        # Delete existing tags
        self.cursor.execute("DELETE FROM tags WHERE video_id = ?", (video_id,))
        
        # Insert new tags
        for tag in tags:
            confidence = confidences.get(tag, 1.0) if confidences else 1.0
            self.cursor.execute("""
                INSERT OR IGNORE INTO tags (video_id, tag, confidence)
                VALUES (?, ?, ?)
            """, (video_id, tag.lower(), confidence))
        
        self.conn.commit()
    
    def insert_frame_descriptions(self, video_id: int, descriptions: List[str]):
        """Insert frame descriptions"""
        # Delete existing descriptions
        self.cursor.execute(
            "DELETE FROM frame_descriptions WHERE video_id = ?",
            (video_id,)
        )
        
        # Insert new descriptions
        for idx, desc in enumerate(descriptions):
            self.cursor.execute("""
                INSERT INTO frame_descriptions (video_id, frame_index, description)
                VALUES (?, ?, ?)
            """, (video_id, idx, desc))
        
        self.conn.commit()
    
    def insert_thumbnails(self, video_id: int, thumbnails: List[Tuple]):
        """
        Insert thumbnail images
        
        Args:
            video_id: Video ID
            thumbnails: List of (frame_number, image_data_base64, width, height) tuples
        """
        # Delete existing thumbnails
        self.cursor.execute("DELETE FROM thumbnails WHERE video_id = ?", (video_id,))
        
        # Insert new thumbnails
        for idx, (frame_num, img_data, width, height) in enumerate(thumbnails):
            self.cursor.execute("""
                INSERT INTO thumbnails (
                    video_id, thumbnail_index, frame_number,
                    image_data, width, height
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (video_id, idx, frame_num, img_data, width, height))
        
        self.conn.commit()
    
    def search_videos(self, 
                     tags: List[str] = None,
                     start_date: str = None,
                     end_date: str = None,
                     search_text: str = None,
                     limit: int = 100) -> List[Dict]:
        """
        Search videos with various filters
        
        Args:
            tags: List of tags to filter by (OR logic)
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            search_text: Search in filename and descriptions
            limit: Maximum results
            
        Returns:
            List of video records
        """
        query = "SELECT DISTINCT v.* FROM videos v"
        conditions = []
        params = []
        
        if tags:
            query += " JOIN tags t ON v.id = t.video_id"
            tag_conditions = " OR ".join(["t.tag = ?" for _ in tags])
            conditions.append(f"({tag_conditions})")
            params.extend(tags)
        
        if start_date:
            conditions.append("v.parsed_datetime >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("v.parsed_datetime <= ?")
            params.append(end_date)
        
        if search_text:
            conditions.append("(v.file_name LIKE ? OR v.description LIKE ?)")
            params.extend([f"%{search_text}%", f"%{search_text}%"])
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY v.parsed_datetime DESC, v.file_modified_date DESC"
        query += f" LIMIT {limit}"
        
        self.cursor.execute(query, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_video_with_tags(self, video_id: int) -> Optional[Dict]:
        """Get complete video info including tags and thumbnails"""
        self.cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        video = self.cursor.fetchone()
        
        if not video:
            return None
        
        video_dict = dict(video)
        
        # Get tags
        self.cursor.execute(
            "SELECT tag, confidence FROM tags WHERE video_id = ?",
            (video_id,)
        )
        video_dict['tags'] = [row[0] for row in self.cursor.fetchall()]
        
        # Get thumbnails
        self.cursor.execute("""
            SELECT thumbnail_index, frame_number, image_data, width, height
            FROM thumbnails WHERE video_id = ?
            ORDER BY thumbnail_index
        """, (video_id,))
        video_dict['thumbnails'] = [dict(row) for row in self.cursor.fetchall()]
        
        return video_dict
    
    def get_all_tags(self) -> List[Tuple[str, int]]:
        """Get all unique tags with counts"""
        self.cursor.execute("""
            SELECT tag, COUNT(*) as count
            FROM tags
            GROUP BY tag
            ORDER BY count DESC, tag ASC
        """)
        return self.cursor.fetchall()
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        self.cursor.execute("SELECT COUNT(*) FROM videos")
        stats['total_videos'] = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(DISTINCT tag) FROM tags")
        stats['unique_tags'] = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT SUM(file_size_bytes) FROM videos")
        total_size = self.cursor.fetchone()[0]
        stats['total_size_gb'] = round(total_size / (1024**3), 2) if total_size else 0
        
        self.cursor.execute("SELECT SUM(duration_seconds) FROM videos")
        total_duration = self.cursor.fetchone()[0]
        stats['total_duration_hours'] = round(total_duration / 3600, 2) if total_duration else 0
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")


def parse_datetime_from_filename(filename: str) -> Optional[str]:
    """
    Extract datetime from filename using various common patterns
    
    Common patterns:
    - VID_20231215_142530.mp4
    - 2023-12-15_14-25-30.mp4
    - 20231215_142530.mp4
    - IMG_20231215.mp4
    - video-2023-12-15-14-25.mp4
    
    Args:
        filename: Video filename
        
    Returns:
        ISO format datetime string or None
    """
    patterns = [
        # YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS
        r'(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})',
        # YYYY-MM-DD_HH-MM-SS or similar
        r'(\d{4})-(\d{2})-(\d{2})[_T-](\d{2})[:-](\d{2})[:-](\d{2})',
        # YYYYMMDD only
        r'(\d{4})(\d{2})(\d{2})',
        # YYYY-MM-DD only
        r'(\d{4})-(\d{2})-(\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 6:
                    # Full datetime
                    dt = datetime(
                        int(groups[0]), int(groups[1]), int(groups[2]),
                        int(groups[3]), int(groups[4]), int(groups[5])
                    )
                elif len(groups) == 3:
                    # Date only
                    dt = datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                else:
                    continue
                
                return dt.isoformat()
            except ValueError:
                continue
    
    return None


def compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of file
    
    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read
        
    Returns:
        Hex digest of file hash
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def extract_file_metadata(file_path: str) -> Dict:
    """
    Extract file system metadata
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file metadata
    """
    path = Path(file_path)
    stat = path.stat()
    
    return {
        'file_path': str(path.resolve()),
        'file_name': path.name,
        'file_size_bytes': stat.st_size,
        'file_created_date': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'file_modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'parsed_datetime': parse_datetime_from_filename(path.name),
        # 'file_hash': compute_file_hash(file_path),  # Optional: can be slow for large files
    }


if __name__ == "__main__":
    # Test the database handler
    db = DatabaseHandler("test_videos.db")
    
    # Test datetime parsing
    test_filenames = [
        "VID_20231215_142530.mp4",
        "2023-12-15_14-25-30.mp4",
        "20231215_142530.mp4",
        "video-2023-12-15.mp4",
        "my_vacation.mp4"
    ]
    
    print("Testing datetime parsing:")
    for filename in test_filenames:
        result = parse_datetime_from_filename(filename)
        print(f"  {filename} -> {result}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    db.close()