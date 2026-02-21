// Eye-tracking cursor overlay and WebSocket client
console.log('Content script loaded - eye cursor client');

let ws = null;
let enabled = true;
let lastPos = {x: window.innerWidth/2, y: window.innerHeight/2};

const overlay = document.createElement('div');
overlay.id = 'eye-cursor-overlay';
overlay.style.position = 'fixed';
overlay.style.left = '0';
overlay.style.top = '0';
overlay.style.width = '18px';
overlay.style.height = '18px';
overlay.style.background = 'rgba(0,150,255,0.9)';
overlay.style.border = '2px solid white';
overlay.style.borderRadius = '50%';
overlay.style.zIndex = '2147483647';
overlay.style.pointerEvents = 'none';
overlay.style.transform = 'translate(-50%, -50%)';
overlay.style.transition = 'transform 0.03s linear, left 0.03s linear, top 0.03s linear';
document.documentElement.appendChild(overlay);

function dispatchPointerMove(x, y) {
	lastPos.x = x;
	lastPos.y = y;
	overlay.style.left = x + 'px';
	overlay.style.top = y + 'px';

	// Create and dispatch a pointermove event at the target element under the cursor
	const target = document.elementFromPoint(x, y) || document.body;
	const ev = new PointerEvent('pointermove', {
		bubbles: true,
		cancelable: true,
		clientX: x,
		clientY: y,
		pointerType: 'touch',
		isPrimary: true,
	});
	target.dispatchEvent(ev);
}

function dispatchClick(x, y) {
	const target = document.elementFromPoint(x, y) || document.body;
	const down = new PointerEvent('pointerdown', {bubbles:true, cancelable:true, clientX:x, clientY:y, pointerType:'touch', isPrimary:true});
	const up = new PointerEvent('pointerup', {bubbles:true, cancelable:true, clientX:x, clientY:y, pointerType:'touch', isPrimary:true});
	target.dispatchEvent(down);
	target.dispatchEvent(up);
	// also emit click for compatibility
	const click = new MouseEvent('click', {bubbles:true, cancelable:true, clientX:x, clientY:y});
	target.dispatchEvent(click);
}

function connectWS(url='ws://localhost:8765'){
	if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
	try{
		ws = new WebSocket(url);
	}catch(e){
		console.warn('WebSocket create failed', e);
		return;
	}
	ws.addEventListener('open', () => console.log('Eye cursor WS connected'));
	ws.addEventListener('message', (ev)=>{
		try{
			const data = JSON.parse(ev.data);
			if (!enabled) return;
			if (data.type === 'cursor'){
				// normalized coords 0..1 -> viewport
				const x = Math.round(data.x * window.innerWidth);
				const y = Math.round(data.y * window.innerHeight);
				dispatchPointerMove(x, y);
			} else if (data.type === 'click'){
				dispatchClick(lastPos.x, lastPos.y);
			}
		}catch(err){
			console.warn('Invalid WS message', err);
		}
	});
	ws.addEventListener('close', ()=>{
		console.log('Eye cursor WS closed, retry in 1s');
		ws = null;
		setTimeout(()=>connectWS(url), 1000);
	});
	ws.addEventListener('error', (e)=>{
		console.warn('Eye cursor WS error', e);
		try{ ws.close(); }catch(_e){}
	});
}

connectWS();

// allow popup to toggle the overlay
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
	if (msg && msg.type === 'EYE_TOGGLE'){
		enabled = !!msg.enabled;
		overlay.style.display = enabled ? 'block' : 'none';
		sendResponse({ok:true, enabled});
	}
});
