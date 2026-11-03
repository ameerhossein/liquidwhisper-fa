const $ = id => document.getElementById(id);
const picker = $('picker');
let artifacts = [], recorder, stream, chunks = [], started, clock, busy = false;
const api = async (url, options) => {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Something went wrong.');
  return data;
};
const postJSON = (url, data) => api(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
function openPicker() { if (!picker.open) picker.showModal(); }
$('changeRepo').onclick = openPicker;
$('switchModel').onclick = openPicker;
$('closePicker').onclick = () => picker.close();
picker.addEventListener('cancel', e => { if (!$('record').disabled) return; e.preventDefault(); });
$('repoForm').onsubmit = async e => {
  e.preventDefault(); $('find').disabled = true; $('artifacts').replaceChildren(); $('pickerStatus').textContent = 'Looking for models…';
  try {
    const data = await postJSON('/api/discover', {repo:$('repo').value});
    artifacts = data.artifacts; $('repoName').textContent = data.repo; $('switchModel').disabled = !artifacts.length;
    $('pickerStatus').textContent = artifacts.length ? `${artifacts.length} model${artifacts.length === 1 ? '' : 's'} found · Select one to load` : 'No complete model or adapter files found in this repository.';
    for (const artifact of artifacts) {
      const button = document.createElement('button'); button.className = 'artifact';
      const path = document.createElement('span'); path.className = 'artifact-path'; path.textContent = artifact.path || 'Repository root';
      const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = artifact.label === artifact.kind ? artifact.kind : `${artifact.label} · ${artifact.kind}`;
      button.append(path, badge); button.onclick = () => selectModel(artifact); $('artifacts').append(button);
    }
  } catch (error) { $('pickerStatus').textContent = error.message; }
  finally { $('find').disabled = false; }
};
async function selectModel(artifact) {
  $('pickerStatus').textContent = 'Loading model… This may take a while on the first download.';
  $('artifacts').querySelectorAll('button').forEach(b => b.disabled = true); $('find').disabled = true; $('record').disabled = true; $('switchModel').disabled = true; $('closePicker').disabled = true;
  try {
    const data = await postJSON('/api/select', {path:artifact.path});
    $('activeModel').textContent = artifact.path || 'Repository root'; $('repoName').textContent = `${$('repo').value.trim()} · ${artifact.kind} · ${data.device.toUpperCase()}`;
    $('record').disabled = false; $('switchModel').disabled = false; $('status').textContent = 'Model ready.'; picker.close();
  } catch (error) { $('pickerStatus').textContent = error.message; $('activeModel').textContent = 'No model loaded'; $('switchModel').disabled = false; }
  finally { $('artifacts').querySelectorAll('button').forEach(b => b.disabled = false); $('find').disabled = false; $('closePicker').disabled = false; }
}
$('record').onclick = async () => {
  if (recorder && recorder.state === 'recording') { recorder.stop(); return; }
  if (busy) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio:true}); chunks = []; recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = finishRecording; recorder.start(); started = Date.now();
    $('record').innerHTML = '<span class="record-dot"></span> Stop & transcribe'; $('signal').classList.add('live'); $('status').textContent = 'Recording…'; $('switchModel').disabled = true; $('changeRepo').disabled = true;
    clock = setInterval(() => { const seconds = Math.min(30, Math.floor((Date.now()-started)/1000)); $('timer').innerHTML = `00:${String(seconds).padStart(2,'0')} <span>/ 00:30</span>`; if (Date.now()-started >= 30000 && recorder.state === 'recording') recorder.stop(); }, 100);
  } catch (error) { if (stream) stream.getTracks().forEach(t => t.stop()); $('status').textContent = `Microphone unavailable: ${error.message}. Use localhost or HTTPS and allow microphone access.`; }
};
async function finishRecording() {
  clearInterval(clock); stream.getTracks().forEach(t => t.stop()); $('signal').classList.remove('live'); $('record').disabled = true; busy = true; $('record').innerHTML = '<span class="record-dot"></span> Start recording'; $('status').textContent = 'Transcribing…';
  let context;
  try {
    context = new AudioContext(); const buffer = await context.decodeAudioData(await new Blob(chunks, {type:recorder.mimeType}).arrayBuffer());
    const wav = toWav(buffer); const data = await api('/api/transcribe', {method:'POST', headers:{'Content-Type':'audio/wav'}, body:wav});
    $('result').textContent = data.text || 'No speech detected.'; $('result').classList.toggle('empty', !data.text); $('copy').disabled = !data.text; $('status').textContent = 'Transcription complete.';
  } catch (error) { $('status').textContent = error.message; }
  finally { if (context) await context.close(); busy = false; $('record').disabled = false; $('switchModel').disabled = false; $('changeRepo').disabled = false; }
}
function toWav(buffer) {
  const samples = Math.min(buffer.length, Math.floor(buffer.sampleRate * 30)); const output = new ArrayBuffer(44 + samples * 2); const view = new DataView(output);
  function write(offset, text) { for (let i=0; i<text.length; i++) view.setUint8(offset+i,text.charCodeAt(i)); }
  write(0,'RIFF'); view.setUint32(4,36+samples*2,true); write(8,'WAVE'); write(12,'fmt '); view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true); view.setUint32(24,buffer.sampleRate,true); view.setUint32(28,buffer.sampleRate*2,true); view.setUint16(32,2,true); view.setUint16(34,16,true); write(36,'data'); view.setUint32(40,samples*2,true);
  for (let i=0; i<samples; i++) { let value = 0; for (let c=0; c<buffer.numberOfChannels; c++) value += buffer.getChannelData(c)[i] / buffer.numberOfChannels; value = Math.max(-1,Math.min(1,value)); view.setInt16(44+i*2,value < 0 ? value*32768 : value*32767,true); }
  return output;
}
$('copy').onclick = async () => { try { await navigator.clipboard.writeText($('result').textContent); $('status').textContent = 'Copied to clipboard.'; } catch { $('status').textContent = 'Could not copy text. Select it to copy manually.'; } };
openPicker();
