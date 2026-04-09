import os
import sys

# Try to install PyMuPDF if not present
try:
    import fitz
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
    import fitz

files = ["Assignment 2.pdf", "03-camera (1).pdf", "04-lane detection (1).pdf"]

for f in files:
    if os.path.exists(f):
        doc = fitz.open(f)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        
        with open(f.replace(".pdf", ".txt"), "w", encoding="utf-8") as out:
            out.write(text)
        print(f"Extracted {f}")
    else:
        print(f"Not found: {f}")
