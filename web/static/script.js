// Global state
let currentVideoId = null;
let selectedTags = [];
let videos = [];
let sortState = { key: 'creation_date', order: 'desc' };
let currentPage = 1;
let itemsPerPage = 12;
let player = null;

// Debounce function to limit how often a function can run
function debounce(func, delay) {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), delay);
    };
}

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

        // Set up view switch handlers
        document.querySelectorAll('input[name="viewRadio"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                renderPage(e.target.value);
            });
        });

        // Initial load
        filterVideos();
    } catch (error) {
        console.error('Error during DOMContentLoaded:', error);
    }

    // Handle modal close event to stop video playback
    const videoPlayerModal = document.getElementById('videoPlayerModal');
    videoPlayerModal.addEventListener('hidden.bs.modal', () => {
        if (player) {
            player.pause();
        }
    });
});

// Load and display tags in the tag cloud
async function loadTags() {
    try {
        const response = await fetch('/api/tags');
        const tags = await response.json();
        
        const tagCloud = document.getElementById('tag-cloud');
        tagCloud.innerHTML = tags.map(tag => `
            <span class="tag ${selectedTags.includes(tag.tag_name) ? 'selected' : ''}" data-tag="${tag.tag_name}" onclick="toggleTag('${tag.tag_name}')">
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
    
    // Also re-render the main tag cloud to show selection state
    const tagCloud = document.getElementById('tag-cloud');
    tagCloud.querySelectorAll('.tag').forEach(tagSpan => {
        const currentTagName = tagSpan.dataset.tag;
        tagSpan.classList.toggle('selected', selectedTags.includes(currentTagName));
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
        videos = await response.json();
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
        const videoCopy = JSON.parse(JSON.stringify(video));
        const thumbnails = video.thumbnail_data || [];
        const firstThumbnail = thumbnails.length > 0 ? thumbnails[0] : '';

        return `
        <div class="col video-item" data-tags='${JSON.stringify(video.tags)}'>
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
                <div class="card-body" onclick='openVideoDetailModal(${video.id})'>
                    <h5 class="card-title text-truncate" title="${video.file_name}">
                        ${video.file_name}
                    </h5>
                    <p class="card-text">
                        <small class="text-muted">
                            ${new Date(video.parsed_datetime || video.file_created_date).toLocaleDateString()}
                        </small>
                    </p>
                    <div class="tags mb-2">
                        ${(video.tags || []).map(tag => 
                            `<span class="badge me-1 video-tag ${selectedTags.includes(tag) ? 'bg-success' : 'bg-primary'}" 
                                  onclick="event.stopPropagation(); toggleTag('${tag}');">${tag}
                            </span>`
                        ).join('')}
                    </div>
                </div>
                <div class="card-footer">
                    <button class="btn btn-primary btn-sm" 
                            onclick="openVideo(${video.id})">
                        Open Video
                    </button>
                    <button class="btn btn-secondary btn-sm" 
                            onclick="openVideoDetailModal(${video.id})">
                        Edit Tags
                    </button>
                    <button class="btn btn-danger btn-sm" 
                            onclick="deleteVideo(${videoCopy.id})">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        </div>
    `}).join('');
}

function renderListView(paginatedVideos) {
    const grid = document.getElementById('video-grid');
    grid.className = 'list-view'; // Remove col classes
    
    grid.innerHTML = `
        <table class="table table-striped">
            <thead>
                <tr>
                    <th scope="col">Thumbnail</th>
                    <th scope="col" onclick="sortVideos('filename')">Filename <i class="fas fa-sort"></i></th>
                    <th scope="col" onclick="sortVideos('creation_date')">Creation Date <i class="fas fa-sort"></i></th>
                    <th scope="col">Tags</th>
                    <th scope="col">Actions</th>
                </tr>
            </thead>
            <tbody>
                ${paginatedVideos.map(video => `
                    <tr>
                        <td>
                            ${video.thumbnail_data && video.thumbnail_data.length > 0
                                ? `<img src="data:image/jpeg;base64,${video.thumbnail_data[0]}" 
                                       alt="Thumbnail" style="width: 100px; height: auto;">`
                                : `<div style="width: 100px; height: 56px; background-color: #ccc;"></div>`
                            }
                        </td>
                        <td>${video.file_name}</td>
                        <td>${new Date(video.parsed_datetime || video.file_created_date).toLocaleDateString()}</td>
                        <td>
                            ${(video.tags || []).map(tag => 
                                `<span class="badge me-1 video-tag ${selectedTags.includes(tag) ? 'bg-success' : 'bg-primary'}" onclick="toggleTag('${tag}')">${tag}</span>`
                            ).join('')}
                        </td>
                        <td>
                            <button class="btn btn-primary btn-sm" onclick="openVideo(${video.id})">Open</button>
                            <button class="btn btn-secondary btn-sm" onclick='openVideoDetailModal(${video.id})'>Details</button>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderGridView(paginatedVideos, viewType = 'grid') {
    const grid = document.getElementById('video-grid');
    const colClass = viewType === 'compact' ? 'col-6 col-md-4 col-lg-2' : 'col-12 col-md-6 col-lg-4';
    grid.className = `row g-4 ${viewType}-view`;

    grid.innerHTML = paginatedVideos.map(video => {
        const thumbnails = video.thumbnail_data || [];
        const firstThumbnail = thumbnails.length > 0 ? thumbnails[0] : '';

        return `
        <div class="video-item ${colClass}">
            <div class="card h-100">
                ${firstThumbnail
                    ? `<img src="data:image/jpeg;base64,${firstThumbnail}" class="card-img-top" alt="Thumbnail" data-thumbnails='${JSON.stringify(thumbnails)}' onmouseenter="startThumbnailCycle(this)" onmouseleave="stopThumbnailCycle(this)">`
                    : `<div class="card-img-top bg-secondary text-white d-flex align-items-center justify-content-center" style="height: 150px;">No Thumbnail</div>`
                }
                <div class="card-body" onclick='openVideoDetailModal(${video.id})'>
                    <h5 class="card-title text-truncate" title="${video.file_name}">${video.file_name}</h5>
                    <p class="card-text"><small class="text-muted">${new Date(video.parsed_datetime || video.file_created_date).toLocaleDateString()}</small></p>
                    <div class="tags-compact mb-2">
                        ${(video.tags || []).map(tag => `<span class="badge me-1 ${selectedTags.includes(tag) ? 'bg-success' : 'bg-primary'}" onclick="event.stopPropagation(); toggleTag('${tag}');">${tag}</span>`).join('')}
                    </div>
                </div>
                <div class="card-footer">
                    <button class="btn btn-primary btn-sm" onclick="openVideo(${video.id})">Play</button>
                    <button class="btn btn-secondary btn-sm" onclick="openVideoDetailModal(${video.id})">Edit</button>
                </div>
            </div>
        </div>`;
    }).join('');
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

function renderPage(viewType = null) {
    const selectedViewRadio = document.querySelector('input[name="viewRadio"]:checked');
    // Default to 'grid' view if the radio buttons aren't on the page or none is checked
    const selectedView = viewType || (selectedViewRadio ? selectedViewRadio.value : 'grid');
    const paginatedVideos = videos.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

    if (selectedView === 'list') {
        renderListView(paginatedVideos);
    } else {
        // 'grid' or 'compact'
        renderGridView(paginatedVideos, selectedView);
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
function openVideo(videoId) {
    const videoUrl = `/video/stream/${videoId}`;
    const modal = new bootstrap.Modal(document.getElementById('videoPlayerModal'));

    if (!player) {
        player = videojs('main-video-player');
    }

    player.src({
        src: videoUrl,
        type: 'video/mp4'
    });
    modal.show();
}

// Play video from the modal
function playVideoFromModal() {
    const videoId = document.getElementById('videoDetailId').value;
    openVideo(videoId);
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

// Open the video detail modal and populate it with data
async function openVideoDetailModal(videoId) {
    currentVideoId = videoId;

    // Fetch full video details
    const response = await fetch(`/api/video/${videoId}`);
    if (!response.ok) {
        console.error('Failed to fetch video details');
        return;
    }
    const video = await response.json();


    // Populate modal fields
    document.getElementById('videoDetailId').value = video.id;
    document.getElementById('videoDetailFilePath').value = video.file_path;
    document.getElementById('videoDetailFilename').value = video.file_name;
    document.getElementById('videoDetailCreationDate').value = new Date(video.parsed_datetime || video.file_created_date).toLocaleString();
    
    // Handle tags
    const tags = video.tags || [];
    document.getElementById('videoDetailTags').value = tags.join(', ');

    // Handle transcript
    const transcriptText = video.transcript || 'No transcript available.';
    document.getElementById('videoDetailTranscript').value = transcriptText;

    // Handle thumbnail
    const thumbnailData = video.thumbnail_data || [];
    const thumbnailElement = document.getElementById('videoDetailThumbnail');
    if (thumbnailData.length > 0) {
        thumbnailElement.src = `data:image/jpeg;base64,${thumbnailData[0]}`;
    } else {
        thumbnailElement.src = ''; // Or a placeholder image
    }

    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('videoDetailModal'));
    modal.show();
}

// Save changes from the video detail modal
async function saveVideoDetails() {
    const videoId = document.getElementById('videoDetailId').value;
    const tags = document.getElementById('videoDetailTags').value.split(',')
        .map(tag => tag.trim()).filter(tag => tag);

    try {
        const response = await fetch(`/api/video/${videoId}/tags`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ tags: tags })
        });

        if (response.ok) {
            // Hide the modal
            const modalInstance = bootstrap.Modal.getInstance(document.getElementById('videoDetailModal'));
            modalInstance.hide();

            // Refresh video grid and tags
            await filterVideos();
            await loadTags();
        } else {
            const errorData = await response.json();
            alert(`Error saving tags: ${errorData.error}`);
        }
    } catch (error) {
        console.error('Error saving video details:', error);
        alert('An error occurred while saving. Please check the console.');
    }
}

// Start cycling through thumbnails on mouse enter
function startThumbnailCycle(element) {
    const thumbnails = JSON.parse(element.dataset.thumbnails || '[]');
    if (thumbnails.length <= 1) {
        return; // No need to cycle
    }

    let currentIndex = 0;
    const intervalId = setInterval(() => {
        currentIndex = (currentIndex + 1) % thumbnails.length;
        element.src = `data:image/jpeg;base64,${thumbnails[currentIndex]}`;
    }, 800); // Change image every 800ms

    // Store interval ID on the element to clear it later
    element.dataset.cycleIntervalId = intervalId;
}

// Stop cycling through thumbnails on mouse leave
function stopThumbnailCycle(element) {
    const intervalId = element.dataset.cycleIntervalId;
    if (intervalId) {
        clearInterval(intervalId);
        delete element.dataset.cycleIntervalId;
        // Reset to the first thumbnail
        const thumbnails = JSON.parse(element.dataset.thumbnails || '[]');
        if (thumbnails.length > 0) {
            element.src = `data:image/jpeg;base64,${thumbnails[0]}`;
        }
    }
}
