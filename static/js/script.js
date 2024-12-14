// Canvas and drawing functionality
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const thicknessSlider = document.getElementById('lineThickness');
const thicknessValue = document.getElementById('lineThicknessValue');
const canvasWidthInput = document.getElementById('canvasWidth');
const canvasHeightInput = document.getElementById('canvasHeight');

// Initial canvas settings
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;
let lineThickness = parseInt(thicknessSlider.value, 10);

// Update line thickness when slider changes
thicknessSlider.addEventListener('input', (e) => {
    lineThickness = parseInt(e.target.value, 10);
    thicknessValue.textContent = lineThickness;
});

// Drawing event listeners
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    ctx.lineWidth = lineThickness;
    ctx.lineCap = 'round';
});

canvas.addEventListener('mousemove', (e) => {
    if (drawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
});

// Clear the canvas
function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Resize the canvas
function resizeCanvas() {
    const newWidth = parseInt(canvasWidthInput.value, 10);
    const newHeight = parseInt(canvasHeightInput.value, 10);

    // Save the current drawing
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Resize the canvas
    canvas.width = newWidth;
    canvas.height = newHeight;

    // Restore the drawing to the resized canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(imageData, 0, 0);
}

// Upload the canvas
async function uploadCanvas() {
    const dataURL = canvas.toDataURL('image/png');
    const response = await fetch('/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    });

    const result = await response.json();
    if (response.ok) {
        alert('Image uploaded successfully!');
        processImage();
    } else {
        alert('Failed to upload image: ' + result.error);
    }
}

// Process the uploaded image
async function processImage() {
    const response = await fetch('/process', { method: 'POST' });
    const result = await response.json();

    if (response.ok) {
        const predictionsContainer = document.getElementById('predictions');
        const evaluationContainer = document.getElementById('evaluation');

        // Populate predictions as styled elements
        predictionsContainer.innerHTML = result.predictions
            .map((prediction) => `<div class="prediction-item">${prediction}</div>`)
            .join("");

        // Populate evaluation result
        evaluationContainer.innerHTML = `<div class="evaluation-result">${result.evaluation}</div>`;
    } else {
        alert('Failed to process image: ' + result.error);
    }
}

