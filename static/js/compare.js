 const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const compareBtn = document.getElementById('compareBtn');
        const loadingState = document.getElementById('loadingState');
        const resultsSection = document.getElementById('resultsSection');
        const yoloContent = document.getElementById('yoloContent');
        const vitContent = document.getElementById('vitContent');
        const totalTime = document.getElementById('totalTime');
        
        let currentFile = null;
        
        // Handle file selection
        function handleFile(file) {
            if (file) {
                // Validate file type
                const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif'];
                if (!validTypes.includes(file.type)) {
                    alert('Please upload a valid image file (PNG, JPG, JPEG, WEBP, GIF)');
                    resetForm();
                    return;
                }
                
                // Validate size (max 16MB)
                if (file.size > 16 * 1024 * 1024) {
                    alert('File too large! Maximum size is 16MB');
                    resetForm();
                    return;
                }
                
                currentFile = file;
                fileName.textContent = `✅ ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                compareBtn.disabled = false;
                
                // Reset results
                resultsSection.classList.add('hidden');
            }
        }
        
        // Click to upload
        dropZone.addEventListener('click', () => fileInput.click());
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });
        
        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFile(fileInput.files[0]);
            }
        });
        
        // Compare button click
        compareBtn.addEventListener('click', async () => {
            if (!currentFile) return;
            
            // Show loading
            loadingState.classList.remove('hidden');
            resultsSection.classList.add('hidden');
            compareBtn.disabled = true;
            
            // Prepare form data
            const formData = new FormData();
            formData.append('imgFile', currentFile);
            
            try {
                const startTime = performance.now();
                
                // Send to server
                const response = await fetch('/compare', {
                    method: 'POST',
                    body: formData
                });
                
                const endTime = performance.now();
                const totalTimeMs = (endTime - startTime) / 1000;
                
                if (!response.ok) {
                    throw new Error('Server error');
                }
                
                const data = await response.json();
                
                // Display results
                displayResults(data, totalTimeMs);
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing image. Please try again.');
            } finally {
                loadingState.classList.add('hidden');
                compareBtn.disabled = false;
            }
        });
        
        // Display results
        function displayResults(data, time) {
            // YOLO Results
            if (data.yolo) {
                let yoloHTML = '';
                
                if (data.yolo.detections && data.yolo.detections.length > 0) {
                    // Show the processed image
                    if (data.yolo.image) {
                        yoloHTML += `<img src="data:image/png;base64,${data.yolo.image}" class="result-image" alt="YOLO Detection">`;
                    }
                    
                    yoloHTML += `<div class="result-details">`;
                    yoloHTML += `<div class="result-item"><span class="result-label">Detections</span><span class="result-value">${data.yolo.detections.length}</span></div>`;
                    yoloHTML += `<div class="result-item"><span class="result-label">Processing Time</span><span class="result-value">${data.yolo.time.toFixed(3)}s</span></div>`;
                    yoloHTML += `<div class="result-item"><span class="result-label">Detected Objects</span></div>`;
                    
                    // List top detections
                    data.yolo.detections.slice(0, 5).forEach((det, i) => {
                        const confPercent = (det.confidence * 100).toFixed(1);
                        const colorClass = det.confidence > 0.7 ? 'high' : (det.confidence > 0.4 ? 'medium' : 'low');
                        yoloHTML += `
                            <div style="margin: 5px 0;">
                                <div style="display: flex; justify-content: space-between; font-size: 14px;">
                                    <span>${i+1}. ${det.class}</span>
                                    <span style="font-weight: bold;">${confPercent}%</span>
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill ${colorClass}" style="width: ${confPercent}%"></div>
                                </div>
                            </div>
                        `;
                    });
                    
                    yoloHTML += `</div>`;
                } else {
                    yoloHTML = `<div class="loading">No objects detected in this image</div>`;
                }
                
                yoloContent.innerHTML = yoloHTML;
            }
            
            // ViT Results
            if (data.vit) {
                let vitHTML = '';
                
                if (data.vit.top_predictions) {
                    // Show original image
                    if (data.vit.image) {
                        vitHTML += `<img src="data:image/png;base64,${data.vit.image}" class="result-image" alt="ViT Classification">`;
                    }
                    
                    vitHTML += `<div class="result-details">`;
                    vitHTML += `<div class="result-item"><span class="result-label">Top Prediction</span><span class="result-value" style="color: #4facfe;">${data.vit.top_prediction}</span></div>`;
                    vitHTML += `<div class="result-item"><span class="result-label">Confidence</span><span class="result-value">${(data.vit.top_confidence * 100).toFixed(1)}%</span></div>`;
                    vitHTML += `<div class="result-item"><span class="result-label">Processing Time</span><span class="result-value">${data.vit.time.toFixed(3)}s</span></div>`;
                    vitHTML += `<div class="result-item"><span class="result-label">Top 5 Predictions</span></div>`;
                    
                    // Show top 5 predictions with bars
                    data.vit.top_predictions.slice(0, 5).forEach((pred, i) => {
                        const confPercent = (pred.confidence * 100).toFixed(1);
                        const colorClass = pred.confidence > 0.7 ? 'high' : (pred.confidence > 0.4 ? 'medium' : 'low');
                        vitHTML += `
                            <div style="margin: 5px 0;">
                                <div style="display: flex; justify-content: space-between; font-size: 14px;">
                                    <span>${i+1}. ${pred.class}</span>
                                    <span style="font-weight: bold;">${confPercent}%</span>
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill ${colorClass}" style="width: ${confPercent}%"></div>
                                </div>
                            </div>
                        `;
                    });
                    
                    vitHTML += `</div>`;
                } else {
                    vitHTML = `<div class="loading">No classification results</div>`;
                }
                
                vitContent.innerHTML = vitHTML;
            }
            
            // Update total time
            totalTime.textContent = `${time.toFixed(2)}s`;
            
            // Show results
            resultsSection.classList.remove('hidden');
        }
        
        // Reset form
        function resetForm() {
            fileInput.value = '';
            fileName.textContent = '';
            compareBtn.disabled = true;
            currentFile = null;
            resultsSection.classList.add('hidden');
            loadingState.classList.add('hidden');
            yoloContent.innerHTML = '<div class="loading">Waiting for results...</div>';
            vitContent.innerHTML = '<div class="loading">Waiting for results...</div>';
        }