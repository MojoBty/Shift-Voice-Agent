const HOST_NAME = 'com.cursor.voiceagent';

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusEl = document.getElementById('status');

function setStatus(message, type = 'idle') {
  statusEl.textContent = message;
  statusEl.className = `status-${type}`;
}

function sendNativeMessage(action) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendNativeMessage(HOST_NAME, { action }, (response) => {
      if (chrome.runtime.lastError) {
        reject(chrome.runtime.lastError);
      } else {
        resolve(response);
      }
    });
  });
}

startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  setStatus('Starting...', 'idle');
  try {
    const response = await sendNativeMessage('start');
    if (response && response.success) {
      setStatus('Voice agent running', 'running');
    } else {
      setStatus(response?.error || 'Failed to start', 'error');
    }
  } catch (err) {
    setStatus(err.message || 'Failed to start. Is native host installed?', 'error');
  }
  startBtn.disabled = false;
});

stopBtn.addEventListener('click', async () => {
  stopBtn.disabled = true;
  setStatus('Stopping...', 'idle');
  try {
    const response = await sendNativeMessage('stop');
    if (response && response.success) {
      setStatus('Ready', 'idle');
    } else {
      setStatus(response?.error || 'Failed to stop', 'error');
    }
  } catch (err) {
    setStatus(err.message || 'Failed to stop', 'error');
  }
  stopBtn.disabled = false;
});
