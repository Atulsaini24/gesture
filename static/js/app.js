document.addEventListener('DOMContentLoaded', () => {
    // Connect to WebSocket
    const socket = io();
    
    // Elements
    const gestureOutput = document.getElementById('gestureOutput');
    const outputMode = document.getElementById('outputMode');
    const status = document.getElementById('status');
    
    // Audio context for speech synthesis
    let audioContext;
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
        console.error('Web Audio API not supported.');
    }
    
    // Handle gesture detection events
    socket.on('gesture_detected', async (data) => {
        const { gesture, text } = data;
        
        // Add animation class
        gestureOutput.classList.add('gesture-detected');
        setTimeout(() => {
            gestureOutput.classList.remove('gesture-detected');
        }, 500);
        
        // Handle different output modes
        switch (outputMode.value) {
            case 'text':
                gestureOutput.textContent = text;
                break;
            case 'emoji':
                gestureOutput.textContent = text.split(' ')[0]; // Get only the emoji
                break;
            case 'speech':
                gestureOutput.textContent = text;
                try {
                    const response = await fetch('/text_to_speech', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text.replace(/[^\w\s]/gi, '') }) // Remove emojis
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
                        await audio.play();
                    }
                } catch (error) {
                    console.error('Error playing audio:', error);
                    status.textContent = 'Error playing audio';
                }
                break;
        }
    });
    
    // Handle connection status
    socket.on('connect', () => {
        status.textContent = 'Connected';
        status.style.color = '#16a34a';
    });
    
    socket.on('disconnect', () => {
        status.textContent = 'Disconnected';
        status.style.color = '#dc2626';
    });
    
    // Handle errors
    socket.on('error', (error) => {
        console.error('Socket error:', error);
        status.textContent = 'Error: ' + error.message;
        status.style.color = '#dc2626';
    });
}); 