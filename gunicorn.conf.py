"""Production server: one copy of the models, bounded inference in app.py."""
import os

bind = '0.0.0.0:' + os.environ.get('PORT', '10000')
workers = 1
worker_class = 'gthread'
threads = 2
timeout = 180
graceful_timeout = 30
accesslog = '-'
errorlog = '-'
capture_output = True
