document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('eye-toggle');
  const status = document.getElementById('status');

  async function setEnabled(enabled){
    status.textContent = 'Status: ' + (enabled ? 'enabled' : 'disabled');
    // send message to active tab content script
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      if (!tabs || !tabs[0]) return;
      chrome.tabs.sendMessage(tabs[0].id, {type: 'EYE_TOGGLE', enabled}, (resp) => {
        // ignore
      });
    });
  }

  toggle.addEventListener('change', (e)=>{
    setEnabled(e.target.checked);
  });

  // initialize
  setEnabled(toggle.checked);
});
