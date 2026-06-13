"""Document ingestion helpers for CSV, PDF, and DOCX sources."""

from __future__ import annotations

from io import BytesIO
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.etree import ElementTree

import pandas as pd

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


FIELD_PATTERN = re.compile(r"^\s*([^:：]{2,40})\s*[:：]\s*(.{1,300})\s*$")


def load_source_as_dataframe(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        return df, {"kind": "table", "status": "parsed", "text_preview": ""}
    if suffix == ".pdf":
        text, metadata = extract_pdf_text(path)
        table_df = extract_pdf_tables(path)
        field_df = fields_dataframe_from_text(text, source_name=path.name)
        if not table_df.empty:
            table_df.insert(0, "source", path.name)
            table_df.insert(1, "extract_method", "pdf_table")
            df = table_df
            metadata["kind"] = "table"
            metadata["table_extracted"] = True
            metadata["table_rows"] = int(table_df.shape[0])
            metadata["table_columns"] = int(table_df.shape[1])
        else:
            df = field_df
            metadata["table_extracted"] = False
        return df, metadata
    if suffix == ".docx":
        text, metadata = extract_docx_text(path)
        df = fields_dataframe_from_text(text, source_name=path.name)
        return df, metadata
    if suffix == ".doc":
        raise ValueError(".doc 是老 Word 二进制格式，建议另存为 .docx 后上传。")
    raise ValueError(f"不支持的文件类型: {suffix}")


def extract_pdf_text(path: Path) -> Tuple[str, Dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("当前环境缺少 PyMuPDF，无法解析 PDF。")
    doc = fitz.open(path)
    page_texts: List[str] = []
    for index, page in enumerate(doc):
        text = page.get_text("text") or ""
        page_texts.append(f"\n--- page {index + 1} ---\n{text.strip()}")
    text = "\n".join(page_texts).strip()
    ocr_available = optional_ocr_available()
    parser = "pymupdf_text_layer"
    if not text and ocr_available:
        text = ocr_pdf(path)
        parser = "ocr"
    status = "parsed" if text else "needs_ocr"
    return text, {
        "kind": "document",
        "parser": parser,
        "status": status,
        "pages": len(doc),
        "text_preview": text[:1000],
        "ocr_required": not bool(text),
        "ocr_available": ocr_available,
        "ocr_note": "PDF 没有可读取文本层，可能是扫描件；需要接入 Tesseract/EasyOCR/PaddleOCR 后再识别。"
        if not text
        else "",
    }


def extract_pdf_tables(path: Path) -> pd.DataFrame:
    """Try PyMuPDF table extraction. Returns empty dataframe if unavailable."""
    if fitz is None:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(path)
        for page_no, page in enumerate(doc, start=1):
            if not hasattr(page, "find_tables"):
                continue
            tables = page.find_tables()
            for table_no, table in enumerate(tables.tables, start=1):
                extracted = table.extract()
                if not extracted:
                    continue
                headers = [str(h or f"col_{idx + 1}") for idx, h in enumerate(extracted[0])]
                for row_no, values in enumerate(extracted[1:], start=1):
                    row = {"page": page_no, "table_no": table_no, "row_no": row_no}
                    for header, value in zip(headers, values):
                        row[header] = value
                    rows.append(row)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def optional_ocr_available() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except Exception:
        pass
    try:
        import pytesseract  # noqa: F401

        return True
    except Exception:
        return False


def ocr_pdf(path: Path, max_pages: int = 5) -> str:
    if fitz is None:
        return ""
    try:
        import easyocr
        from PIL import Image
        import numpy as np

        reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        lines: List[str] = []
        doc = fitz.open(path)
        for page_index in range(min(max_pages, len(doc))):
            page_no = page_index + 1
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pix.tobytes("png")))
            texts = reader.readtext(np.array(image), detail=0)
            lines.append(f"\n--- ocr page {page_no} ---\n" + "\n".join(map(str, texts)))
        return "\n".join(lines).strip()
    except Exception:
        pass
    try:
        import pytesseract
        from PIL import Image

        lines = []
        doc = fitz.open(path)
        for page_index in range(min(max_pages, len(doc))):
            page_no = page_index + 1
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")
            lines.append(f"\n--- ocr page {page_no} ---\n{text}")
        return "\n".join(lines).strip()
    except Exception:
        return ""


def extract_docx_text(path: Path) -> Tuple[str, Dict[str, Any]]:
    paragraphs: List[str] = []
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", ns):
        parts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    text = "\n".join(paragraphs)
    return text, {
        "kind": "document",
        "parser": "docx_xml",
        "status": "parsed" if text else "empty",
        "pages": None,
        "text_preview": text[:1000],
        "ocr_required": False,
        "ocr_note": "",
    }


def fields_dataframe_from_text(text: str, source_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = FIELD_PATTERN.match(line)
        if match:
            rows.append(
                {
                    "source": source_name,
                    "line_no": line_no,
                    "field": match.group(1).strip(),
                    "value": match.group(2).strip(),
                    "confidence": 0.85,
                    "extract_method": "key_value_regex",
                    "raw_text": line,
                }
            )
        else:
            rows.append(
                {
                    "source": source_name,
                    "line_no": line_no,
                    "field": "paragraph",
                    "value": line,
                    "confidence": 0.45,
                    "extract_method": "paragraph",
                    "raw_text": line,
                }
            )
    if not rows:
        rows.append(
            {
                "source": source_name,
                "line_no": 0,
                "field": "ocr_required",
                "value": "未识别到文本层，可能需要 OCR。",
                "confidence": 0.0,
                "extract_method": "empty_document",
                "raw_text": "",
            }
        )
    return pd.DataFrame(rows)
