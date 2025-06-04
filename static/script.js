document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const speed = parseFloat(document.getElementById('speed').value);
    const acceleration = parseFloat(document.getElementById('acceleration').value);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                speed_ms: speed,
                acceleration_ms2: acceleration
            })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        document.getElementById('predictionValue').textContent = 
            data.emission_prediction.toFixed(2);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('predictionValue').textContent = 'Error';
    }
}); 