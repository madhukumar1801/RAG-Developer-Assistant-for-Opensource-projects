from typing import List
import re
from bs4 import BeautifulSoup
import markdown
import ast
import os


class CodeChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_markdown(self, content: str) -> str:
        """Convert Markdown to plain text while preserving semantic structure."""
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()

    def process_python(self, content: str) -> List[str]:
        """Split Python code into logical blocks using `ast`."""
        try:
            tree = ast.parse(content)
            splits = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    splits.append(f"\n\n# {node.name} starts here\n")
            return splits
        except Exception:
            return content.split('\n')

    def chunk_code(self, content: str, file_path: str) -> List[dict]:
        """Chunk code into smaller segments based on file type."""
        chunks = []

        # Handle Markdown files
        if file_path.endswith('.md'):
            content = self.process_markdown(content)

        if file_path.endswith('.py'):
            splits = self.process_python(content)
        elif file_path.endswith(('.java', '.cpp', '.js', '.ts', '.cs', '.rb', '.php', '.go', '.tsx', '.jsx')):
            # Split by common keywords in code files
            splits = re.split(r'(class |def |function |const |let |var |public |private |return )', content)
        elif file_path.endswith(('.log', '.csv', '.json', '.xml', '.yaml')):
            # Split structured or log files by lines
            splits = content.splitlines()
        else:
            # Generic split by lines
            splits = content.split('\n')

        # Create chunks with overlap
        current_chunk = ""
        for split in splits:
            if len(current_chunk) + len(split) > self.chunk_size:
                chunks.append({
                    "content": current_chunk,
                    "metadata": {
                        "file_path": file_path,
                        "start_line": len(chunks) * self.chunk_size
                    }
                })
                current_chunk = split[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
            current_chunk += split

        # Add the last chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk,
                "metadata": {
                    "file_path": file_path,
                    "start_line": len(chunks) * self.chunk_size
                }
            })

        return chunks