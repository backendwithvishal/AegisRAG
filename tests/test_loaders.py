import os
import tempfile
import pytest
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.html import parse_html


def test_parse_text_file():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Enterprise Kubernetes Architecture Overview\nSection 1: Networking")
        f_path = f.name

    try:
        content = parse_text(f_path)
        assert "Enterprise Kubernetes" in content
        assert "Section 1: Networking" in content
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def test_parse_html_file():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title><script>alert('bad');</script></head>
    <body>
        <h1>Kubernetes HPA</h1>
        <p>Horizontal Pod Autoscaling automates replica counts.</p>
        <style>.hide { display: none; }</style>
    </body>
    </html>
    """
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html_content)
        f_path = f.name

    try:
        content = parse_html(f_path)
        assert "Kubernetes HPA" in content
        assert "Horizontal Pod Autoscaling" in content
        assert "alert" not in content
        assert "display: none" not in content
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)
