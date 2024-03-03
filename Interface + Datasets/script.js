function classifyText() {
    var inputText = document.getElementById('sentence').value;

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:5000/classify", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    xhr.onload = function () {
        if (xhr.status == 200) {
            var result = JSON.parse(xhr.responseText).result;

            // Display the result in the specified format
            var resultSection = document.getElementById('resultSection');
            resultSection.innerHTML = "<p id='result'>The predicted sentiment for the sentence \"<strong>" + inputText + "</strong>\" is:</p>";

            // Create a new paragraph for each sentiment category
            var sentimentParagraph = document.createElement('p');
            sentimentParagraph.className = getResultStyle(result); // Apply the style based on sentiment
            sentimentParagraph.innerHTML = result;
            resultSection.appendChild(sentimentParagraph);
        } else {
            console.error("Error:", xhr.statusText);
        }
    };

    xhr.send(JSON.stringify({ text: inputText }));
}

function getResultStyle(sentiment) {
    if (sentiment === 'Positive') {
        return 'positiveSentiment';
    } else if (sentiment === 'Negative') {
        return 'negativeSentiment';
    } else if (sentiment === 'Neutral') {
        return 'neutralSentiment';
    } else {
        return 'defaultSentiment'; // Default style for other cases
    }
}

function clearText() {
    document.getElementById('sentence').value = ''; // Clears the text in the textarea
    clearResult(); // Clears the sentiment result as well
}

function clearResult() {
    var resultSection = document.getElementById('resultSection');
    resultSection.innerHTML = ''; // Clears the result section
}

function uploadFile() {
    var fileInput = document.getElementById('file');
    var file = fileInput.files[0];

    if (file) {
        var formData = new FormData();
        formData.append('file', file);

        var xhr = new XMLHttpRequest();
        xhr.open("POST", "http://localhost:5000/upload", true);
        xhr.onload = function () {
            if (xhr.status == 200) {
                // Handle the response if needed
                console.log("File uploaded successfully!");
            } else {
                console.error("Error:", xhr.statusText);
            }
        };

        xhr.send(formData);
    } else {
        alert("Please choose a file to upload.");
    }
}


function classifyAndDownload() {
    var fileInput = document.getElementById('file');
    var file = fileInput.files[0];

    if (file) {
        var formData = new FormData();
        formData.append('file', file);

        var xhr = new XMLHttpRequest();
        xhr.open("POST", "http://localhost:5000/classify-and-download", true);
        xhr.responseType = 'blob';  // Set response type to blob for file download

        xhr.onload = function () {
            if (xhr.status == 200) {
                var blob = new Blob([xhr.response], { type: 'text/csv' });
                var link = document.createElement('a');
                link.href = window.URL.createObjectURL(blob);
                link.download = 'classified_tweets.csv';
                link.click();
            } else {
                console.error("Error:", xhr.statusText);
            }
        };

        xhr.send(formData);
    } else {
        alert("Please choose a file to upload.");
    }
}

