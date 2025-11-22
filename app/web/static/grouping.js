// -----------------------
// GROUPING MODULE
// -----------------------
// Handles collapse/expand functionality for year and folder groups in video browser

// Global state for group visibility
let groupingState = {
    years: {},
    folders: {}
};

// -----------------------
// INITIALIZATION
// -----------------------

// Initialize grouping state on page load
function initializeGroupingState() {
    loadGroupingState();
}

// -----------------------
// STATE MANAGEMENT
// -----------------------

// Save grouping state to localStorage
function saveGroupingState() {
    try {
        localStorage.setItem('videoOrga_groupingState', JSON.stringify(groupingState));
    } catch (error) {
        console.error('Error saving grouping state:', error);
    }
}

// Load grouping state from localStorage
function loadGroupingState() {
    try {
        const saved = localStorage.getItem('videoOrga_groupingState');
        if (saved) {
            groupingState = JSON.parse(saved);
        }
    } catch (error) {
        console.error('Error loading grouping state:', error);
        groupingState = { years: {}, folders: {} };
    }
}

// -----------------------
// TOGGLE FUNCTIONS
// -----------------------

// Toggle visibility of all folders and videos in a year group
function toggleYearGroup(year) {
    const yearKey = `year-${year}`;
    const isCollapsed = groupingState.years[yearKey] || false;

    // Update state
    groupingState.years[yearKey] = !isCollapsed;
    saveGroupingState();

    // Toggle all elements with matching data-year attribute
    const yearElements = document.querySelectorAll(`[data-year="${year}"]`);
    yearElements.forEach(element => {
        if (element.classList.contains('year-header')) {
            // Update icon rotation
            const icon = element.querySelector('.collapse-icon');
            if (icon) {
                icon.classList.toggle('collapsed', !isCollapsed);
            }
        } else {
            // Toggle visibility of content
            element.classList.toggle('collapsed', !isCollapsed);
        }
    });
}

// Toggle visibility of videos in a folder group
function toggleFolderGroup(year, folder) {
    const folderKey = `folder-${year}-${folder}`;
    const isCollapsed = groupingState.folders[folderKey] || false;

    // Update state
    groupingState.folders[folderKey] = !isCollapsed;
    saveGroupingState();

    // Toggle all elements with matching data-folder attribute
    const folderElements = document.querySelectorAll(`[data-year="${year}"][data-folder="${folder}"]`);
    folderElements.forEach(element => {
        if (element.classList.contains('folder-header')) {
            // Update icon rotation
            const icon = element.querySelector('.collapse-icon');
            if (icon) {
                icon.classList.toggle('collapsed', !isCollapsed);
            }
        } else {
            // Toggle visibility of content
            element.classList.toggle('collapsed', !isCollapsed);
        }
    });
}

// -----------------------
// UTILITY FUNCTIONS
// -----------------------

// Check if a year group is collapsed
function isYearCollapsed(year) {
    const yearKey = `year-${year}`;
    return groupingState.years[yearKey] || false;
}

// Check if a folder group is collapsed
function isFolderCollapsed(year, folder) {
    const folderKey = `folder-${year}-${folder}`;
    return groupingState.folders[folderKey] || false;
}

// Apply saved state to rendered elements
function applyGroupingState() {
    // Apply year states
    Object.keys(groupingState.years).forEach(yearKey => {
        if (groupingState.years[yearKey]) {
            const year = yearKey.replace('year-', '');
            const yearElements = document.querySelectorAll(`[data-year="${year}"]`);
            yearElements.forEach(element => {
                if (element.classList.contains('year-header')) {
                    const icon = element.querySelector('.collapse-icon');
                    if (icon) {
                        icon.classList.add('collapsed');
                    }
                } else {
                    element.classList.add('collapsed');
                }
            });
        }
    });

    // Apply folder states
    Object.keys(groupingState.folders).forEach(folderKey => {
        if (groupingState.folders[folderKey]) {
            const parts = folderKey.replace('folder-', '').split('-');
            const year = parts[0];
            const folder = parts.slice(1).join('-');
            const folderElements = document.querySelectorAll(`[data-year="${year}"][data-folder="${folder}"]`);
            folderElements.forEach(element => {
                if (element.classList.contains('folder-header')) {
                    const icon = element.querySelector('.collapse-icon');
                    if (icon) {
                        icon.classList.add('collapsed');
                    }
                } else {
                    element.classList.add('collapsed');
                }
            });
        }
    });
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initializeGroupingState();
});
