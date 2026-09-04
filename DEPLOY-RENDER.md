# Deploy Model Compare to Render

Nothing is deployed just by adding these files. No paid service has been created.

## 1. Commit and push the changes

Review your Git changes, then commit and push them to the branch you want to deploy
in `Kofi-RM/MP3`. Do not commit `.env`, `.venv`, `.runtime`, or `.models`.

Old screenshots and a database already in the repository are NOT served by the
app or copied into the Docker image. However, anything tracked in a public GitHub
repository (including its history) remains publicly accessible through GitHub.
Review that repository separately before making it public.

## 2. Create the service

1. Open https://dashboard.render.com/.
2. Choose **New > Blueprint** and connect `Kofi-RM/MP3`.
3. Choose the branch with these files. The Blueprint path is `render.yaml`.
4. Review the service and price before confirming creation.

The Blueprint requests the **1c-2g (1 CPU / 2 GB)** compute plan, which is paid.
This is an initial memory estimate, not a guarantee. Watch memory usage after
deployment and adjust if necessary. No credit card or purchase action was taken
by the coding assistant.

Render builds the Dockerfile and generates `SECRET_KEY`. Models are downloaded
during the build, then packaged in the image. Startup uses those local weights
with network model downloads disabled.

### If using New > Web Service instead

- Connect the repository and choose **Docker**, not Python or Static Site.
- Leave Root Directory blank if the Dockerfile is at the repository root.
- Dockerfile Path: `./Dockerfile`.
- Do not supply the earlier pip build command: Docker handles the build.
- Leave Docker Command empty: the Dockerfile starts Gunicorn.
- Add a random `SECRET_KEY` of at least 32 characters using Render's generator.
- Set Health Check Path to `/healthz`.
- Choose a suitable paid compute plan after checking pricing.

## 3. Verify the deployment

Wait for a successful build and healthy service, then visit the HTTPS URL Render
provides. Test `/`, `/compare`, and `/yolo`. `/healthz` should return `{"status":"ok"}`.

- Compare a small JPG and verify both model results.
- Try a short MP4/WebM in YOLO Studio.
- Allow camera access only when you want to use webcam mode. Frames will now be
  sent to Render, not your local computer. No audio is requested.
- Stop webcam mode and confirm the camera-use indicator turns off.

## Security and capacity

- Legacy file-saving routes are removed. Images are decoded in memory, never
  written using client-provided filenames. Flask may spool multipart data to an
  OS temporary file during a request; it does not retain it as an app upload.
- Only explicitly listed CSS, JavaScript, and the homepage sample image are public.
- Request sizes, form parts, image dimensions, and output sizes are bounded.
- One inference request runs at a time. Excess concurrent requests get HTTP 503
  with Retry-After instead of consuming unbounded model memory.
- Cross-site browser uploads are rejected and internal errors are not returned.
- The site is still an unauthenticated public demo. These safeguards are not a
  complete rate limiter or DDoS defense. Add authentication and edge rate limits
  before sharing widely or hosting private/sensitive footage.
- The old hardcoded session secret is gone; production refuses to start without
  a configured secret. Do not reuse the previous secret.

## Troubleshooting

- **Missing SECRET_KEY:** set it in Render Environment, then redeploy.
- **No matching distribution:** inspect the first failing dependency in build logs;
  the current pins were tested on Windows but not yet in a Linux Docker build.
- **Build download failure:** retry the build; model downloads require Hugging Face
  access during the build, not when serving users.
- **Out of memory:** increase memory or reduce workload. Keep one Gunicorn worker
  so model weights are not duplicated.
- **503 busy:** another inference is running; retry shortly.
- **Camera unavailable:** use the HTTPS deployment URL, allow camera permission,
  and try Chrome/Edge directly if an embedded browser blocks camera access.

## Local development

Double-click `start.cmd` (now targets `app.py`) and open http://127.0.0.1:5000.
The Windows launcher does not require Gunicorn or Docker. Local mode generates
an ephemeral secret if none is set; Render requires a configured one.

Deployment references:
- https://render.com/docs/deploy-flask
- https://render.com/docs/blueprint-spec
- https://render.com/docs/compute-plans
