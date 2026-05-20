let RunSentimentAnalysis = () => {
    const textToAnalyze = document.getElementById("textToAnalyze").value;
    const systemResponse = document.getElementById("system_response");

    systemResponse.innerHTML = "Processing...";

    const xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () {
        if (this.readyState === 4) {
            if (this.status === 200) {
                systemResponse.innerHTML = xhttp.responseText;
            } else {
                systemResponse.innerHTML =
                    "Unable to process the request right now. Please try again later.";
            }
        }
    };
    xhttp.open("GET", `emotionDetector?textToAnalyze=${encodeURIComponent(textToAnalyze)}`, true);
    xhttp.send();
};
