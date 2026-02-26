let chartInstance = null;

async function analyzeNews() {

    const headline = document.getElementById("headline").value.trim();
    const article = document.getElementById("article").value.trim();

    if (!headline || !article) {
        alert("Please enter both headline and article.");
        return;
    }

    const analyzeBtn = document.getElementById("analyzeBtn");
    analyzeBtn.disabled = true;
    analyzeBtn.innerText = "Analyzing...";

    try {

        // Combine headline + article
        const fullText = headline + " " + article;

        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: fullText
            })
        });

        const data = await response.json();

        // Show result panel
        document.getElementById("resultArea").style.display = "block";

        // Set verdict
        document.getElementById("verdict").innerText = data.prediction_result;

        // Set risk level
        document.getElementById("riskLevel").innerText = data.risk_level;

        // Add accuracy dynamically
        let accuracyElement = document.getElementById("accuracyDisplay");
        if (!accuracyElement) {
            const newElement = document.createElement("p");
            newElement.id = "accuracyDisplay";
            newElement.style.marginTop = "10px";
            newElement.innerHTML = "<strong>Model Accuracy:</strong> " + data.model_accuracy;
            document.querySelector(".metrics-grid").appendChild(newElement);
        } else {
            accuracyElement.innerHTML = "<strong>Model Accuracy:</strong> " + data.model_accuracy;
        }

        // Summary
        document.getElementById("hSummary").innerText = headline;
        document.getElementById("aSummary").innerText = article.substring(0, 200) + "...";

        // Chart Data
        const fakePercent = data.graph_data["Fake (%)"];
        const realPercent = data.graph_data["Real (%)"];

        const ctx = document.getElementById("predictionChart").getContext("2d");

        // Destroy previous chart if exists
        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Fake (%)", "Real (%)"],
                datasets: [{
                    data: [fakePercent, realPercent],
                    backgroundColor: ["#ff4d4d", "#4CAF50"]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: "bottom"
                    }
                }
            }
        });

    } catch (error) {
        console.error("Error:", error);
        alert("Server not responding. Make sure backend is running.");
    }

    analyzeBtn.disabled = false;
    analyzeBtn.innerText = "Analyze Authenticity";
}