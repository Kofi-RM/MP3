# Editing the website

Each page is now a complete, ordinary HTML file. There are no template blocks,
inherited layouts, or generated asset links to follow.

## Where to make changes

- `templates/Home.html`: home page, introduction, and how-it-works section.
- `templates/Compare.html`: image upload and side-by-side model results.
- `templates/Yolo.html`: webcam and video detection studio.
- `static/css/styles.css`: shared colors, fonts, buttons, navigation, and layout.
- `static/css/compare.css`: upload and results styling.
- `static/css/yolo.css`: webcam/video studio styling.
- `static/js/compare.js`: image upload and displaying comparison results.
- `static/js/yolo.js`: camera/video controls and detection boxes.
- `app.py`: Python server and AI model processing.

Section comments in the HTML mark the navigation, upload controls, results, and
other major areas. CSS classes control appearance; JavaScript uses the `id`
attributes to find controls, so keep those IDs unless you update the script too.

The small header and footer are written directly in each page for readability.
If you change a shared navigation link, update it in all three HTML files.

## Running the website

Double-click `start.cmd` and open http://127.0.0.1:5000.
The pages use plain links such as `/static/css/styles.css` and `/yolo`.
Flask still handles the routes and AI requests; Live Server cannot run Python
or process camera frames and image uploads.

For hosting, follow `DEPLOY-RENDER.md`. The Render Blueprint uses Docker so
Python, OpenCV system libraries, and model weights ship together.
