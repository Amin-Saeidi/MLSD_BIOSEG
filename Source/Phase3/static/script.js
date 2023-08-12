document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const classifyButton = document.getElementById('classify-button');
    const resultText = document.getElementById('result-text');
    const uploadedImage = document.getElementById('uploaded-image');
    const predictedImage = document.getElementById('predicted-image');
    const endpoint = 'http://localhost:8000/predict';

    classifyButton.addEventListener('click', function() {
        const file = imageInput.files[0];

        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
            };
            reader.readAsDataURL(file);

            const formData = new FormData();
            formData.append('image', file);

            const xhr = new XMLHttpRequest();
            xhr.open('POST', endpoint, true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    
                    const predictedImgData = response.image;
                    const img = new Image();
                    img.src = "data:image/png;base64," + predictedImgData;
                    const message = response.message;
                    
                    // Display the predicted image
                    predictedImage.src = img.src;

                    // Update the result text
                    resultText.textContent = `Class: ${message}`;
                } else {
                    resultText.textContent = 'Error occurred during classification.';
                }
            };
            xhr.send(formData);
        }
    });
});
