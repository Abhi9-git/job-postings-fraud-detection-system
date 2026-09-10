"""Vercel serverless entry point for the Flask app.

Vercel's Python runtime looks for an `app` variable inside `api/*.py`
and serves it as the request handler. All routes (/, /api/*) are
rewritten here via vercel.json, so the full Flask app is served
from this single function.
"""

import os
import sys

# Ensure `import app` resolves to the repo-root app.py when Vercel
# runs this file from inside /api.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402  (Vercel expects `app` at module level)
