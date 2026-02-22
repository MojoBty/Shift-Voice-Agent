const HOST_NAME = 'com.cursor.voiceagent';
const MAX_TAB_CONTENT = 50000;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const calibrateBtn = document.getElementById('calibrateBtn');
const statusEl = document.getElementById('status');
const voiceSelect = document.getElementById('voiceSelect');
const tabPreview = document.getElementById('tabPreview');

chrome.storage.local.get('voiceName', (data) => {
  if (data.voiceName) voiceSelect.value = data.voiceName;
});

voiceSelect.addEventListener('change', () => {
  chrome.storage.local.set({ voiceName: voiceSelect.value });
});

async function getTabContent(tabId) {
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        const text = document.body?.innerText || document.body?.textContent || '';
        return text.replace(/\s+/g, ' ').trim();
      },
    });
    return results?.[0]?.result || '';
  } catch {
    return '';
  }
}

async function loadTabPreview() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || tab.url?.startsWith('chrome://') || tab.url?.startsWith('chrome-extension://')) {
      tabPreview.textContent = 'No tab or restricted page';
      tabPreview.classList.add('empty');
      return;
    }
    const content = await getTabContent(tab.id);
    const words = content.split(/\s+/).filter(Boolean);
    const preview = words.length > 0 ? words.slice(0, 30).join(' ') + (words.length > 30 ? '…' : '') : 'Empty page';
    tabPreview.textContent = preview || 'Empty page';
    tabPreview.classList.toggle('empty', !preview || preview === 'Empty page');
  } catch {
    tabPreview.textContent = 'Could not load tab';
    tabPreview.classList.add('empty');
  }
}

loadTabPreview();

function setStatus(message, type = 'idle') {
  statusEl.className = `status-bar status-${type}`;
  const textEl = statusEl.querySelector('span:last-child');
  if (textEl) textEl.textContent = message;
  else statusEl.textContent = message;
}

function sendNativeMessage(action, extra = {}) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendNativeMessage(HOST_NAME, { action, ...extra }, (response) => {
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
  const voiceName = voiceSelect.value;
  let tabContent = '';
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id && !tab.url?.startsWith('chrome://') && !tab.url?.startsWith('chrome-extension://')) {
      tabContent = (await getTabContent(tab.id)).slice(0, MAX_TAB_CONTENT);
    }
  } catch {}
  const payload = {};
  if (voiceName) payload.voiceName = voiceName;
  if (tabContent) payload.tabContent = tabContent;
  try {
    const response = await sendNativeMessage('start', payload);
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

calibrateBtn.addEventListener('click', async () => {
  calibrateBtn.disabled = true;
  setStatus('Opening calibration…', 'idle');
  try {
    const response = await sendNativeMessage('calibrate');
    if (response && response.success) {
      setStatus('Calibration opened in Terminal. Complete it there.', 'idle');
    } else {
      setStatus(response?.error || 'Calibration failed', 'error');
    }
  } catch (err) {
    setStatus(err.message || 'Calibration failed. Is native host installed?', 'error');
  }
  calibrateBtn.disabled = false;
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
