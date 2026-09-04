'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const video = $('sourceVideo'), output = $('outputCanvas');
  const capture = document.createElement('canvas');
  let mode = 'webcam', stream = null, videoUrl = null;
  let running = false, epoch = 0, timer = null, pending = null;
  const hide = (id, value = true) => $(id).classList.toggle('hidden', value);
  const status = text => { $('studioStatus').textContent = text; };
  function error(text = '') { $('studioError').textContent = text; hide('studioError', !text); }
  function updateButtons() {
    $('startDetection').disabled = running || (mode === 'video' && !videoUrl);
    $('startDetection').textContent = running ? 'Analyzing…' : mode === 'webcam' ? 'Start webcam ↗' : 'Analyze video ↗';
    $('stopDetection').disabled = !running;
  }
  function stop(message = 'Stopped · last analyzed frame retained') {
    running = false; epoch++; clearTimeout(timer);
    if (pending) pending.abort(); pending = null;
    video.pause();
    if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
    video.srcObject = null;
    if (mode === 'webcam') hide('sourceVideo');
    status(message); updateButtons();
  }
  function clearOutput() {
    hide('outputCanvas'); hide('stageEmpty', false);
    ['objectCount', 'inferenceTime', 'analysisRate'].forEach(id => { $(id).textContent = '—'; });
    const li = document.createElement('li'); li.textContent = 'Detections will appear here.';
    $('detectionList').replaceChildren(li);
  }
  function switchMode(next) {
    if (next === mode) return;
    stop(); mode = next; error(); clearOutput();
    if (videoUrl) URL.revokeObjectURL(videoUrl); videoUrl = null;
    video.removeAttribute('src'); video.load(); $('videoFile').value = ''; hide('sourceVideo');
    video.controls = mode === 'video';
    $('webcamMode').setAttribute('aria-pressed', String(mode === 'webcam'));
    $('videoMode').setAttribute('aria-pressed', String(mode === 'video'));
    hide('filePicker', mode !== 'video');
    $('sourceType').textContent = mode === 'webcam' ? 'LIVE CAMERA' : 'LOCAL VIDEO';
    $('sourceNote').textContent = mode === 'webcam' ? 'Your browser will ask for camera permission. Audio is never requested.' : 'Use the source player to pause, seek, or replay. Analysis follows playback.';
    $('stageEmpty').querySelector('p').textContent = mode === 'webcam' ? 'Start your webcam to see what YOLO recognizes.' : 'Choose a video, then start analysis.';
    $('videoName').textContent = 'MP4, WebM, or another browser-supported video.';
    status('Ready when you are'); updateButtons();
  }
  function paint(data, elapsed) {
    output.width = capture.width; output.height = capture.height;
    const ctx = output.getContext('2d'); ctx.drawImage(capture, 0, 0);
    ctx.lineWidth = Math.max(2, output.width / 400);
    const fontSize = Math.max(14, Math.round(output.width / 48));
    ctx.font = `600 ${fontSize}px sans-serif`;
    const items = [];
    for (const detection of data.detections) {
      const [x1,y1,x2,y2] = detection.bbox;
      const label = `${detection.class} ${Math.round(detection.confidence * 100)}%`;
      ctx.strokeStyle = '#d8ef93'; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      const width = Math.min(output.width, ctx.measureText(label).width + 12);
      const x = Math.max(0, Math.min(x1, output.width - width)), y = Math.max(fontSize + 10, y1);
      ctx.fillStyle = '#d8ef93'; ctx.fillRect(x,y-fontSize-10,width,fontSize+10);
      ctx.fillStyle = '#203a30'; ctx.fillText(label,x+6,y-6);
      const li = document.createElement('li'), name = document.createElement('span'), score = document.createElement('strong');
      name.textContent = detection.class; score.textContent = `${Math.round(detection.confidence*100)}%`;
      li.append(name,score); items.push(li);
    }
    if (!items.length) { const li = document.createElement('li'); li.textContent = 'No objects above this threshold.'; items.push(li); }
    $('detectionList').replaceChildren(...items);
    $('objectCount').textContent = data.detections.length;
    $('inferenceTime').textContent = `${Math.round(data.time*1000)} ms`;
    $('analysisRate').textContent = `${(1000 / Math.max(elapsed, 180)).toFixed(1)} fps`;
    hide('stageEmpty'); hide('outputCanvas', false);
  }
  async function tick(token) {
    if (!running || token !== epoch) return;
    if (video.paused || video.seeking || video.readyState < 2) {
      status(video.paused ? 'Paused · press play in the source player' : 'Waiting for video…');
      timer = setTimeout(() => tick(token), 200); return;
    }
    const started = performance.now();
    const scale = Math.min(1, 960/video.videoWidth, 720/video.videoHeight);
    capture.width = Math.max(1, Math.round(video.videoWidth*scale));
    capture.height = Math.max(1, Math.round(video.videoHeight*scale));
    capture.getContext('2d').drawImage(video,0,0,capture.width,capture.height);
    const frameTime = video.currentTime;
    let timeout;
    try {
      const blob = await new Promise(resolve => capture.toBlob(resolve,'image/jpeg',0.8));
      if (!running || token !== epoch) return;
      if (!blob) throw new Error('Could not read this video frame.');
      const form = new FormData(); form.append('frame',blob,'frame.jpg'); form.append('confidence',Number($('confidence').value)/100);
      const controller = new AbortController(); pending = controller;
      timeout = setTimeout(() => controller.abort(), 20000);
      const response = await fetch('/api/yolo/frame',{method:'POST',body:form,signal:controller.signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Frame analysis failed.');
      if (!running || token !== epoch) return;
      // A seek invalidates a pending frame; normal playback remains timestamped.
      if (!video.seeking && (mode === 'webcam' || (video.currentTime >= frameTime && video.currentTime-frameTime < 5))) {
        paint(data,performance.now()-started);
        status(mode === 'webcam' ? 'Live · analyzing camera frames' : `Analyzing video · frame at ${frameTime.toFixed(1)}s`);
      }
    } catch (err) {
      if (token !== epoch) return;
      stop('Analysis stopped'); error(err.name === 'AbortError' ? 'The server took too long. Please try again.' : err.message);
    } finally { clearTimeout(timeout); if (token === epoch) pending = null; }
    if (running && token === epoch) timer = setTimeout(() => tick(token),Math.max(0,180-(performance.now()-started)));
  }
  async function start() {
    if (running) return;
    error(); running = true; const token = ++epoch; updateButtons();
    status(mode === 'webcam' ? 'Waiting for camera permission…' : 'Starting video…');
    try {
      if (mode === 'webcam') {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires localhost or HTTPS and a supported browser.');
        const camera = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
        if (token !== epoch) { camera.getTracks().forEach(track=>track.stop()); return; }
        stream = camera; video.srcObject = camera;
        camera.getVideoTracks().forEach(track=>track.addEventListener('ended',()=>{ if (running && token===epoch) { stop('Camera disconnected'); error('The camera was disconnected. Reconnect it and try again.'); } }));
      } else if (!videoUrl) throw new Error('Choose a video first.');
      if (video.ended) video.currentTime = 0;
      hide('sourceVideo',false); await video.play();
      if (token !== epoch) return;
      tick(token);
    } catch (err) {
      if (token !== epoch) return;
      stop('Unable to start');
      const messages = {NotAllowedError:'Camera or playback permission was denied. Allow camera access in your browser, or choose a video file.',NotFoundError:'No camera was found. Connect a webcam or choose a video file.',NotReadableError:'The camera is unavailable. Close other apps using it and try again.',NotSupportedError:'This video format cannot be played. Try an MP4 (H.264) or WebM file.'};
      error(messages[err.name] || err.message);
    }
  }
  $('webcamMode').addEventListener('click',()=>switchMode('webcam'));
  $('videoMode').addEventListener('click',()=>switchMode('video'));
  $('startDetection').addEventListener('click',start);
  $('stopDetection').addEventListener('click',()=>stop());
  $('confidence').addEventListener('input',()=>{ $('confidenceValue').textContent = `${$('confidence').value}%`; });
  $('videoFile').addEventListener('change',()=>{
    const file = $('videoFile').files[0]; if (!file) return;
    stop(); error(); clearOutput();
    if (videoUrl) URL.revokeObjectURL(videoUrl); videoUrl = null;
    video.removeAttribute('src'); video.load(); hide('sourceVideo');
    if (file.type && !file.type.startsWith('video/')) { error('Choose a video file.'); updateButtons(); return; }
    videoUrl = URL.createObjectURL(file); video.src = videoUrl; video.controls = true;
    $('videoName').textContent = file.name; hide('sourceVideo',false); status('Video ready · start analysis'); updateButtons();
  });
  video.addEventListener('error',()=>{ if (mode === 'video' && videoUrl) { stop('Video could not be loaded'); URL.revokeObjectURL(videoUrl); videoUrl=null; updateButtons(); error('Your browser cannot decode this video. Try MP4 (H.264) or WebM.'); } });
  video.addEventListener('ended',()=>{ if (running && mode === 'video') stop('Video complete · start again to replay'); });
  document.addEventListener('visibilitychange',()=>{ if (document.hidden && running) stop('Paused for privacy · press start to resume'); });
  window.addEventListener('pagehide',()=>{ stop(); if (videoUrl) URL.revokeObjectURL(videoUrl); videoUrl=null; });
  updateButtons();
})();
