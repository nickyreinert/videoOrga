// Global state
let currentVideoId = null;
let selectedTags = [];
let videos = [];
let sortState = { key: 'creation_date', order: 'desc' };
let currentPage = 1;
let itemsPerPage = 12;

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded and parsed');
    try {
        // Load tags for the tag cloud
        loadTags();
        
        // Set up search input handler
        const searchInput = document.getElementById('search');
        searchInput.addEventListener('input', debounce(filterVideos, 300));
        
        // Set up date filter handlers
        document.getElementById('start-date').addEventListener('change', filterVideos);
        document.getElementById('end-date').addEventListener('change', filterVideos);

        // Set up view switch handler
        document.getElementById('viewSwitch').addEventListener('change', () => {
            renderPage();
        });

        // Initial load
        filterVideos();
    } catch (error) {
        console.error('Error during DOMContentLoaded:', error);
    }
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
    
    console.log('Fetching videos with params:', params.toString());
    try {
        const response = await fetch(`/api/videos?${params.toString()}`);
        console.log('Response from server:', response);
        videos = await response.json();
        console.log('Videos data:', videos);
        currentPage = 1; // Reset to first page
        renderPage();
    } catch (error) {
        console.error('Error filtering videos:', error);
    }
}

function updateVideoGrid() {
    const grid = document.getElementById('video-grid');
    const paginatedVideos = videos.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

    grid.innerHTML = paginatedVideos.map(video => {
        const thumbnails = video.thumbnail_data || [];
        const firstThumbnail = thumbnails.length > 0 ? thumbnails[0] : '';

        return `
        <div class="col video-item" data-tags='${JSON.stringify(video.tags)}' onclick='openVideoDetailModal(${JSON.stringify(video)})'>
            <div class="card h-100">
                ${firstThumbnail
                    ? `<img src="data:image/jpeg;base64,${firstThumbnail}" 
                           class="card-img-top" alt="Thumbnail" 
                           data-thumbnails='${JSON.stringify(thumbnails)}' 
                           onmouseenter="startThumbnailCycle(this)" 
                           onmouseleave="stopThumbnailCycle(this)">`
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
                    <button class="btn btn-danger btn-sm" 
                            onclick="deleteVideo(${video.id})">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        </div>
    `}).join('');
}

function renderListView() {
    const grid = document.getElementById('video-grid');
    const paginatedVideos = videos.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

    grid.innerHTML = `
        <table class="table table-striped">
            <thead>
                <tr>
                    <th scope="col" onclick="sortVideos('filename')">Filename <i class="fas fa-sort"></i></th>
                    <th scope="col" onclick="sortVideos('creation_date')">Creation Date <i class="fas fa-sort"></i></th>
                    <th scope="col">Tags</th>
                    <th scope="col">Actions</th>
                </tr>
            </thead>
            <tbody>
                ${paginatedVideos.map(video => `
                    <tr>
                        <td>${video.filename}</td>
                        <td>${new Date(video.creation_date).toLocaleDateString()}</td>
                        <td>
                            ${(video.tags || []).map(tag => `<span class="badge bg-primary me-1">${tag}</span>`).join('')}
                        </td>
                        <td>
                            <button class="btn btn-primary btn-sm" onclick="openVideo('${video.file_path}')">Open</button>
                            <button class="btn btn-secondary btn-sm" onclick='openVideoDetailModal(${JSON.stringify(video)})'>Details</button>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function sortVideos(key) {
    if (sortState.key === key) {
        sortState.order = sortState.order === 'asc' ? 'desc' : 'asc';
    } else {
        sortState.key = key;
        sortState.order = 'asc';
    }

    videos.sort((a, b) => {
        if (a[key] < b[key]) {
            return sortState.order === 'asc' ? -1 : 1;
        }
        if (a[key] > b[key]) {
            return sortState.order === 'asc' ? 1 : -1;
        }
        return 0;
    });

    renderPage();
}

function renderPage() {
    if (document.getElementById('viewSwitch').checked) {
        renderListView();
    } else {
        updateVideoGrid();
    }
    renderPagination();
}

// Render pagination controls
function renderPagination() {
    const pagination = document.getElementById('pagination');
    const pageCount = Math.ceil(videos.length / itemsPerPage);
    pagination.innerHTML = '';

    if (pageCount <= 1) {
        return;
    }

    for (let i = 1; i <= pageCount; i++) {
        const li = document.createElement('li');
        li.classList.add('page-item');
        if (i === currentPage) {
            li.classList.add('active');
        }
        const a = document.createElement('a');
        a.classList.add('page-link');
        a.href = '#';
        a.innerText = i;
        a.onclick = () => changePage(i);
        li.appendChild(a);
        pagination.appendChild(li);
    }
}

// Change the current page
function changePage(page) {
    currentPage = page;
    renderPage();
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

// Delete a video
async function deleteVideo(videoId) {
    if (!confirm('Are you sure you want to delete this video?')) {
        return;
    }

    try {
        const response = await fetch(`/api/video/${videoId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            // Refresh the video grid
            filterVideos();
        } else {
            alert('Error deleting video');
        }
    } catch (error) {
        console.error('Error deleting video:', error);
        alert('Error deleting video');
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

// Start cycling through thumbnails on hover
function startThumbnailCycle(img) {
    const thumbnails = JSON.parse(img.dataset.thumbnails);
    if (thumbnails.length <= 1) {
        return;
    }

    let currentIndex = 0;
    img.dataset.thumbnailIndex = currentIndex;

    const intervalId = setInterval(() => {
        currentIndex = (currentIndex + 1) % thumbnails.length;
        img.src = `data:image/jpeg;base64,${thumbnails[currentIndex]}`;
        img.dataset.thumbnailIndex = currentIndex;
    }, 800); // Change thumbnail every 800ms

    img.dataset.intervalId = intervalId;
}

// Stop cycling through thumbnails and reset to the first one
function stopThumbnailCycle(img) {
    clearInterval(img.dataset.intervalId);
    const thumbnails = JSON.parse(img.dataset.thumbnails);
    img.src = `data:image/jpeg;base64,${thumbnails[0]}`;
    img.dataset.thumbnailIndex = 0;
}

// Open the video detail modal
function openVideoDetailModal(video) {
    currentVideoId = video.id;
    document.getElementById('videoDetailId').value = video.id;
    document.getElementById('videoDetailFilePath').value = video.file_path;
    document.getElementById('videoDetailFilename').value = video.filename;
    document.getElementById('videoDetailCreationDate').value = new Date(video.creation_date).toLocaleString();
    document.getElementById('videoDetailTags').value = (video.tags || []).join(', ');
    
    const thumbnail = document.getElementById('videoDetailThumbnail');
    const firstThumbnail = (video.thumbnail_data && video.thumbnail_data.length > 0) ? video.thumbnail_data[0] : '';
    if (firstThumbnail) {
        thumbnail.src = `data:image/jpeg;base64,${firstThumbnail}`;
    } else {
        thumbnail.src = '';
    }

    const modal = new bootstrap.Modal(document.getElementById('videoDetailModal'));
    modal.show();
}

// Save video details from the modal
async function saveVideoDetails() {
    const videoId = document.getElementById('videoDetailId').value;
    const filename = document.getElementById('videoDetailFilename').value;
    const tags = document.getElementById('videoDetailTags').value.split(',').map(tag => tag.trim());

    try {
        const response = await fetch(`/api/video/${videoId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename: filename, tags: tags })
        });

        if (response.ok) {
            filterVideos();
            bootstrap.Modal.getInstance(document.getElementById('videoDetailModal')).hide();
        } else {
            alert('Error saving video details');
        }
    } catch (error) {
        console.error('Error saving video details:', error);
        alert('Error saving video details');
    }
}

// Play video from the modal
function playVideoFromModal() {
    const filePath = document.getElementById('videoDetailFilePath').value;
    openVideo(filePath);
}


// Delete video from the modal
function deleteVideoFromModal() {
    const videoId = document.getElementById('videoDetailId').value;
    deleteVideo(videoId);
    bootstrap.Modal.getInstance(document.getElementById('videoDetailModal')).hide();
}

// Rename video file
async function renameVideo() {
    const videoId = document.getElementById('videoDetailId').value;
    const newFilename = document.getElementById('videoDetailFilename').value;

    if (!confirm(`Are you sure you want to rename this file to "${newFilename}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/video/${videoId}/rename`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ new_filename: newFilename })
        });

        if (response.ok) {
            filterVideos();
            bootstrap.Modal.getInstance(document.getElementById('videoDetailModal')).hide();
        } else {
            alert('Error renaming video');
        }
    } catch (error) {
        console.error('Error renaming video:', error);
        alert('Error renaming video');
    }
}
