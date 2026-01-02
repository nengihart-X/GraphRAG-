import os
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import markdown
from sentence_transformers import SentenceTransformer
import structlog

from ..models.schemas import DocumentChunk
from config.settings import settings

logger = structlog.get_logger()

class DocumentProcessor:
    """Handles document parsing and text extraction"""
    
    def __init__(self):
        self.supported_formats = settings.supported_formats
    
    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text and metadata from various document formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_pdf(file_path)
        elif file_path.suffix.lower() == '.docx':
            return self._extract_docx(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            return self._extract_text_file(file_path)
        elif file_path.suffix.lower() == '.html':
            return self._extract_html(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
    
    def _extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF file"""
        text = []
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': 'pdf',
            'file_size': file_path.stat().st_size
        }
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata['page_count'] = len(reader.pages)
                metadata['title'] = reader.metadata.get('/Title', '') if reader.metadata else ''
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text.append({
                            'content': page_text,
                            'page_number': page_num + 1,
                            'char_count': len(page_text)
                        })
        
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
            raise
        
        return {
            'text_sections': text,
            'metadata': metadata,
            'full_text': '\n'.join([section['content'] for section in text])
        }
    
    def _extract_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                text.append({
                    'content': para.text,
                    'paragraph_number': len(text) + 1,
                    'char_count': len(para.text)
                })
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': 'docx',
            'file_size': file_path.stat().st_size,
            'paragraph_count': len(text)
        }
        
        return {
            'text_sections': text,
            'metadata': metadata,
            'full_text': '\n'.join([section['content'] for section in text])
        }
    
    def _extract_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from plain text or markdown file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        text = []
        
        for i, para in enumerate(paragraphs):
            text.append({
                'content': para,
                'paragraph_number': i + 1,
                'char_count': len(para)
            })
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': 'markdown' if file_path.suffix == '.md' else 'text',
            'file_size': file_path.stat().st_size,
            'paragraph_count': len(text)
        }
        
        return {
            'text_sections': text,
            'metadata': metadata,
            'full_text': content
        }
    
    def _extract_html(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from HTML file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        text_sections = []
        
        for i, para in enumerate(paragraphs):
            text_sections.append({
                'content': para,
                'paragraph_number': i + 1,
                'char_count': len(para)
            })
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': 'html',
            'file_size': file_path.stat().st_size,
            'paragraph_count': len(text_sections),
            'title': soup.title.string if soup.title else ''
        }
        
        return {
            'text_sections': text_sections,
            'metadata': metadata,
            'full_text': text
        }
