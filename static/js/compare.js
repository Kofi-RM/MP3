'use strict';
const $ = id => document.getElementById(id);
const dropZone = $('dropZone'), fileInput = $('fileInput'), compareBtn = $('compareBtn');
let currentFile = null, previewUrl = null, pending = null, generation = 0;
const hide = (id, value = true) => $(id).classList.toggle('hidden', value);
function error(message) { $('errorMessage').textContent = message; hide('errorMessage', !message); }
function clearResults() { hide('resultsSection'); $('yoloContent').replaceChildren(); $('vitContent').replaceChildren(); }
function cancelRequest() { generation++; if (pending) pending.abort(); pending = null; hide('loadingState'); compareBtn.textContent = 'Compare models ↗'; }
function resetForm() {
  cancelRequest(); currentFile = null; fileInput.value = ''; clearResults(); error('');
  if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null;
  $('uploadPreview').removeAttribute('src'); hide('uploadPreview'); hide('uploadPrompt', false);
  $('fileName').textContent = 'A new point of view starts here.'; compareBtn.disabled = true; $('clearBtn').disabled = true;
}
function handleFile(file) {
  if (!file) return;
  resetForm();
  if (!/\.(png|jpe?g|gif)$/i.test(file.name) || (file.type && !['image/png','image/jpeg','image/gif'].includes(file.type))) return error('Choose a JPG, PNG, or GIF image.');
  if (file.size > 16 * 1024 * 1024) return error('This image is too large. Please choose a file under 16 MB.');
  currentFile = file; previewUrl = URL.createObjectURL(file); $('uploadPreview').src = previewUrl;
  hide('uploadPreview', false); hide('uploadPrompt');
  $('fileName').textContent = file.name + ' · ' + (file.size / 1024 / 1024).toFixed(2) + ' MB';
  compareBtn.disabled = false; $('clearBtn').disabled = false;
}
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
$('uploadPreview').addEventListener('error', () => { if (currentFile) { resetForm(); error('This file could not be opened as an image. Please choose another.'); } });
dropZone.addEventListener('dragover', event => { event.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', event => { event.preventDefault(); dropZone.classList.remove('dragover'); handleFile(event.dataTransfer.files[0]); });
$('clearBtn').addEventListener('click', resetForm);
const escapeHTML = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const percent = value => Math.max(0, Math.min(100, Number(value) * 100 || 0));
const row = (label, value) => '<div class="result-item"><span class="result-label">' + label + '</span><span class="result-value">' + escapeHTML(value) + '</span></div>';
function predictions(items) { return items.slice(0, 5).map(item => '<div class="prediction"><div class="prediction-label"><span>' + escapeHTML(item.class) + '</span><strong>' + percent(item.confidence).toFixed(1) + '%</strong></div><div class="confidence-bar"><div class="confidence-fill" style="width:' + percent(item.confidence) + '%"></div></div></div>').join(''); }
function resultImage(base64, alt) { return typeof base64 === 'string' && /^[A-Za-z0-9+/=\s]+$/.test(base64) ? '<img class="result-image" alt="' + alt + '" src="data:image/png;base64,' + base64 + '">' : ''; }
function displayResults(data, time) {
  if (!data.yolo || !data.vit) throw new Error('Incomplete response');
  const detections = data.yolo.detections || [], top = data.vit.top_predictions || [];
  $('yoloContent').innerHTML = resultImage(data.yolo.image, 'YOLO object detection results') + row('Objects detected', detections.length) + row('Processing time', Number(data.yolo.time).toFixed(3) + ' s') + (detections.length ? predictions(detections) : '<p class="empty-message">No objects detected. Try an image with more clearly visible objects.</p>');
  $('vitContent').innerHTML = resultImage(data.vit.image, 'Image analyzed by Vision Transformer') + row('Top prediction', data.vit.top_prediction || 'No prediction') + row('Processing time', Number(data.vit.time).toFixed(3) + ' s') + predictions(top);
  $('totalTime').textContent = time.toFixed(2) + ' s'; hide('resultsSection', false);
}
compareBtn.addEventListener('click', async () => {
  if (!currentFile || pending) return;
  const requestId = ++generation, controller = new AbortController(); pending = controller;
  const form = new FormData(); form.append('imgFile', currentFile);
  clearResults(); error(''); hide('loadingState', false); compareBtn.disabled = true; compareBtn.textContent = 'Comparing…';
  const start = performance.now();
  try {
    const response = await fetch('/compare', {method:'POST', body:form, signal:controller.signal});
    if (!response.ok) throw new Error('Processing failed');
    const data = await response.json();
    if (requestId !== generation) return;
    displayResults(data, (performance.now() - start) / 1000);
    $('resultsSection').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block:'start'});
  } catch (err) { if (err.name !== 'AbortError' && requestId === generation) error('We couldn’t process this image. Check that the model server is running and try again.'); }
  finally { if (requestId === generation) { pending = null; hide('loadingState'); compareBtn.disabled = !currentFile; compareBtn.textContent = 'Compare models ↗'; } }
});
