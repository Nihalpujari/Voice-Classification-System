
function setupRecorder() {
    let mediaRecorder;
    let chunks = [];

    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');

    recordButton.onclick = () => {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();
                recordButton.disabled = true;
                stopButton.disabled = false;

                mediaRecorder.ondataavailable = e => chunks.push(e.data);
                mediaRecorder.onstop = () => {
                    const blob = new Blob(chunks, { type: 'audio/webm' });
                    const formData = new FormData();
                    formData.append('audio', blob, 'recorded_audio.webm');

                    fetch('/record', { method: 'POST', body: formData })
                        .then(response => response.text())
                        .then(html => document.documentElement.innerHTML = html);
                };
            });
    };

    stopButton.onclick = () => {
        mediaRecorder.stop();
        recordButton.disabled = false;
        stopButton.disabled = true;
    };
}
