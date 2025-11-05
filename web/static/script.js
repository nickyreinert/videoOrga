// Global state
let currentVideoId = null;
let selectedTags = [];

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    // Load tags for the tag cloud
    loadTags();
    
    // Set up search input handler
    const searchInput = document.getElementById('search');
    searchInput.addEventListener('input', debounce(filterVideos, 300));
    
    // Set up date filter handlers
    document.getElementById('start-date').addEventListener('change', filterVideos);
    document.getElementById('end-date').addEventListener('change', filterVideos);
});

// Load and display tags in the tag cloud
async function loadTags() {
    try {
        const response = await fetch('/api/tags');
        const tags = await response.json();
        
        const tagCloud = document.getElementById('tag-cloud');
        tagCloud.innerHTML = tags.map(tag => `
            <span class="tag" onclick="toggleTag('${tag.tag_name}')">
                ${tag.tag_name} (${tag.count})
            </span>
        `).join('');
    } catch (error) {
        console.error('Error loading tags:', error);
    }
}

// Toggle tag selection in the filter
function toggleTag(tagName) {
    const index = selectedTags.indexOf(tagName);
    if (index === -1) {
        selectedTags.push(tagName);
    } else {
        selectedTags.splice(index, 1);
    }
    
    // Update UI
    document.querySelectorAll('.tag').forEach(tag => {
        if (tag.textContent.startsWith(tagName + ' ')) {
            tag.classList.toggle('selected', selectedTags.includes(tagName));
        }
    });
    
    filterVideos();
}

// Filter videos based on current filters
async function filterVideos() {
    const search = document.getElementById('search').value;
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    
    // Build query parameters
    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    selectedTags.forEach(tag => params.append('tags', tag));
    
    try {
        const response = await fetch(`/api/videos?${params.toString()}`);
        const videos = await response.json();
        updateVideoGrid(videos);
    } catch (error) {
        console.error('Error filtering videos:', error);
    }
}

// Update the video grid with filtered results
function updateVideoGrid(videos) {
    const grid = document.getElementById('video-grid');
    grid.innerHTML = videos.map(video => `
        <div class="col video-item" data-tags='${JSON.stringify(video.tags)}'>
            <div class="card h-100">
                ${video.thumbnail_data 
                    ? `<img src="data:image/jpeg;base64,${video.thumbnail_data}" 
                           class="card-img-top" alt="Thumbnail">`
                    : `<div class="card-img-top bg-secondary text-white d-flex 
                           align-items-center justify-content-center" 
                           style="height: 200px;">
                           No Thumbnail
                       </div>`
                }
                <div class="card-body">
                    <h5 class="card-title text-truncate" title="${video.filename}">
                        ${video.filename}
                    </h5>
                    <p class="card-text">
                        <small class="text-muted">
                            ${new Date(video.creation_date).toLocaleDateString()}
                        </small>
                    </p>
                    <div class="tags mb-2">
                        ${(video.tags || []).map(tag => 
                            `<span class="badge bg-primary me-1">${tag}</span>`
                        ).join('')}
                    </div>
                </div>
                <div class="card-footer">
                    <button class="btn btn-primary btn-sm" 
                            onclick="openVideo('${video.file_path}')">
                        Open Video
                    </button>
                    <button class="btn btn-secondary btn-sm" 
                            onclick="editTags(${video.id}, ${JSON.stringify(video.tags || [])})">
                        Edit Tags
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

// Open video in system player or stream
function openVideo(path) {
    fetch(`/video/${path}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error opening video:', error);
            alert('Error opening video');
        });
}

// Edit tags for a video
function editTags(videoId, currentTags) {
    currentVideoId = videoId;
    const modal = new bootstrap.Modal(document.getElementById('tagEditModal'));
    const currentTagsDiv = document.getElementById('currentTags');
    const tagInput = document.getElementById('tagInput');
    
    // Display current tags
    currentTagsDiv.innerHTML = (currentTags || []).map(tag =>
        `<span class="badge bg-primary">${tag} 
            <span onclick="removeTag('${tag}')">&times;</span>
         </span>`
    ).join('');
    
    tagInput.value = '';
    modal.show();
}

// Remove a tag in the edit modal
function removeTag(tag) {
    const tagElement = document.querySelector(`#currentTags .badge:contains('${tag}')`);
    if (tagElement) {
        tagElement.remove();
    }
}

// Save updated tags
async function saveTags() {
    if (!currentVideoId) return;
    
    // Get current tags from badges
    const tags = Array.from(document.querySelectorAll('#currentTags .badge'))
        .map(badge => badge.textContent.trim().replace(' ×', ''));
    
    // Get new tags from input
    const newTags = document.getElementById('tagInput').value
        .split(',')
        .map(tag => tag.trim())
        .filter(tag => tag && !tags.includes(tag));
    
    // Combine all tags
    const allTags = [...tags, ...newTags];
    
    try {
        const response = await fetch(`/api/video/${currentVideoId}/tags`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ tags: allTags })
        });
        
        if (response.ok) {
            // Refresh the video grid
            filterVideos();
            // Close the modal
            bootstrap.Modal.getInstance(document.getElementById('tagEditModal')).hide();
        } else {
            alert('Error saving tags');
        }
    } catch (error) {
        console.error('Error saving tags:', error);
        alert('Error saving tags');
    }
}

// Utility function to debounce filter calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}