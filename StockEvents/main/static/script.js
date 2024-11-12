window.onload = function() {
    console.log(Plotly);
};


// Select the div with the class 'content'
const contentDiv = document.querySelector('.content');
const contentOutline = document.querySelector('.contentOutline');

// Check if the div is empty
if (!contentDiv.innerHTML.trim()) {
    // If empty, hide the div
    contentOutline.style.display = 'none';
}

