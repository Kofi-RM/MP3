"""Run with: python -m unittest discover -s tests -v (models must be available)."""
import io
import os
from pathlib import Path
import secrets
import unittest
from unittest.mock import patch

os.environ['APP_ENV'] = 'production'
os.environ['SECRET_KEY'] = secrets.token_hex(32)
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import app as module
from PIL import Image


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = module.app.test_client()
        self.image = (Path(module.ROOT) / 'static/images/car1.jpg').read_bytes()

    def upload(self, path='/compare', data=None, headers=None):
        field = 'frame' if path.endswith('/frame') else 'imgFile'
        return self.client.post(path, data=data or {field: (io.BytesIO(self.image), 'car.jpg')}, headers=headers)

    def test_pages_and_assets(self):
        for path in ['/', '/compare', '/yolo', '/healthz', '/static/css/styles.css', '/static/js/yolo.js', '/static/images/car1.jpg']:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
                response.close()

    def test_old_uploads_and_private_assets_are_not_public(self):
        for path in ['/uploadYolo', '/uploadVit', '/vit', '/vitclass', '/yoloclass', '/static/data/layering.db', '/static/images/results.png', '/static/../app.py', '/static/images/sss.png']:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)
                expected = 405 if path.startswith('/static/') else 404
                self.assertEqual(self.client.post(path).status_code, expected)

    def test_actual_inference(self):
        before = (Path(module.ROOT) / 'static/images/car1.jpg').read_bytes()
        result = self.upload()
        self.assertEqual(result.status_code, 200, result.get_json())
        self.assertEqual(len(result.json['vit']['top_predictions']), 5)
        self.assertTrue(result.json['yolo']['image'])
        self.assertEqual(result.headers['Cache-Control'], 'no-store')
        frame = self.upload('/api/yolo/frame')
        self.assertEqual(frame.status_code, 200, frame.json)
        self.assertTrue(any(d['class'] == 'car' for d in frame.json['detections']))
        self.assertEqual(before, (Path(module.ROOT) / 'static/images/car1.jpg').read_bytes())

    def test_invalid_uploads_and_confidence(self):
        for route, field in [('/compare', 'imgFile'), ('/api/yolo/frame', 'frame')]:
            self.assertEqual(self.client.post(route).status_code, 400)
            bad = self.upload(route, {field:(io.BytesIO(b'not an image'),'image.jpg')})
            self.assertEqual(bad.status_code, 400)
        for value in ['nan', 'inf', '0.99', 'abc']:
            response = self.upload('/api/yolo/frame', {'frame':(io.BytesIO(self.image),'f.jpg'), 'confidence':value})
            self.assertEqual(response.status_code,400)

    def test_dimensions_and_request_limits(self):
        buffer = io.BytesIO()
        Image.new('RGB', (3000,3000)).save(buffer, format='PNG')
        for route, field in [('/compare','imgFile'), ('/api/yolo/frame','frame')]:
            response = self.upload(route, {field:(io.BytesIO(buffer.getvalue()),'large.png')})
            self.assertEqual(response.status_code,400)
        too_big = self.client.post('/api/yolo/frame',data=b'x'*(2*1024*1024+1),content_type='application/octet-stream')
        self.assertEqual(too_big.status_code,413)
        too_big = self.upload(data={'imgFile':(io.BytesIO(b'x'*(16*1024*1024+1)),'large.jpg')})
        self.assertEqual(too_big.status_code,413)

    def test_cross_site_protection(self):
        for headers in [{'Origin':'https://evil.example'}, {'Sec-Fetch-Site':'cross-site'}, {'Origin':'http://[invalid'}]:
            self.assertEqual(self.upload(headers=headers).status_code,403)

    def test_busy_capacity(self):
        module.inference_slot.acquire()
        try:
            response = self.upload()
            self.assertEqual(response.status_code,503)
            self.assertEqual(response.headers['Retry-After'],'1')
            self.assertEqual(self.client.get('/healthz').status_code,200)
        finally:
            module.inference_slot.release()

    def test_safe_errors_and_capacity_recovery(self):
        with patch.object(module,'yolo_model',side_effect=RuntimeError('private internal detail')), patch.object(module.app.logger, 'exception'):
            response = self.upload()
            self.assertEqual(response.status_code,500)
            self.assertNotIn('private internal detail',response.get_data(as_text=True))
        self.assertTrue(module.inference_slot.acquire(blocking=False))
        module.inference_slot.release()

    def test_production_secret(self):
        self.assertTrue(module.app.config['SESSION_COOKIE_SECURE'])
        self.assertGreaterEqual(len(module.app.secret_key),32)


if __name__ == '__main__':
    unittest.main()
