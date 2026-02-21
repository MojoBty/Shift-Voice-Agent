// Simple MV3 service worker (background)
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.type === 'FROM_POPUP') {
    console.log('Background received:', message);
    sendResponse({text: 'Background got it!'});
  }
  return true;
});
