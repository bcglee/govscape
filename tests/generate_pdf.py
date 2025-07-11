import sys
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_pdf(file_path, text):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    c = canvas.Canvas(file_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, text)
    c.save()

def delete_pdf(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


create_pdf('/homes/gws/cgong16/govscape/tests/test_files/pdfs/govscape_intro.pdf', "hello my name is govscape")