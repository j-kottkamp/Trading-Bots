window.onload = function() {
    console.log(Plotly);
};


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


document.addEventListener("DOMContentLoaded", function() {
    function fetchAndDisplayMetrics(ticker = '') {
        const metricsUrlBase = document.getElementById("data-container").dataset.metricsUrl;
        let metricsUrl = ticker ? `${metricsUrlBase}?ticker=${ticker}` : metricsUrlBase;

        fetch(metricsUrl)
            .then(response => response.json())
            .then(metrics => {
                document.getElementById("ticker").textContent = metrics.ticker;
                document.getElementById("average_return").textContent = metrics.average_return.toFixed(2);
                document.getElementById("total_return").textContent = metrics.total_return.toFixed(2);
                document.getElementById("volatility").textContent = metrics.volatility.toFixed(2);
                document.getElementById("max_drawdown").textContent = metrics.max_drawdown.toFixed(2);
                document.getElementById("CAGR").textContent = metrics.CAGR.toFixed(2);
                document.getElementById("sharpe_ratio").textContent = metrics.sharpe_ratio.toFixed(2);
                document.getElementById("calmar_ratio").textContent = metrics.calmar_ratio.toFixed(2);
            })
            .catch(error => console.error('Error fetching metrics:', error));
    }

    const form = document.getElementById("ticker-form");
    const input = document.getElementById("ticker");

    if (form && input) {
        form.addEventListener("submit", function(event) {
            event.preventDefault();
            fetchAndDisplayMetrics(input.value);
        });

        input.addEventListener("keydown", function(event) {
            if (event.key === "Enter") {
                event.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        });
    }
});