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


document.getElementById('ticker').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent the form's default submission

        const ticker = event.target.value;

        // Send the AJAX request to submit the form data
        fetch("{% url 'fetch_data' %}", {
            method: 'POST',
            headers: {
                'X-CSRFToken': '{{ csrf_token }}',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ticker: ticker })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                console.log(data); // Do something with the data
            }
        })
        .catch(error => console.error('Error:', error));
    }
});


