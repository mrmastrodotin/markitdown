import sys
import io
import re
import base64
import html
from typing import BinaryIO, Any

from ._html_converter import HtmlConverter
from ._llm_caption import llm_caption
from .._base_converter import DocumentConverter, DocumentConverterResult
from .._stream_info import StreamInfo
from .._exceptions import MissingDependencyException, MISSING_DEPENDENCY_MESSAGE

# -------------------------------
# Optional dependencies (captured)
# -------------------------------
_fitx_exc = None
_pdfminer_exc = None
_pdfplumber_exc = None

try:
    import fitz  # PyMuPDF
except ImportError:
    _fitx_exc = sys.exc_info()

try:
    import pdfminer.high_level  # type: ignore
except ImportError:
    _pdfminer_exc = sys.exc_info()

try:
    import pdfplumber  # type: ignore
except ImportError:
    _pdfplumber_exc = sys.exc_info()

ACCEPTED_MIME_TYPE_PREFIXES = [
    "application/pdf",
    "application/x-pdf",
]

ACCEPTED_FILE_EXTENSIONS = [".pdf"]


class PdfConverter(DocumentConverter):
    """
    Converts PDFs to Markdown.

    Features:
      - Page markers: <!-- Page N -->
      - Text via PyMuPDF; fallback to pdfminer
      - Embedded images extracted; optional LLM captioning
      - Optional tables via pdfplumber -> HTML -> Markdown using HtmlConverter
      - Chunked page processing for low-RAM environments

    Options (kwargs):
      - keep_data_uris: bool = True  (inline images as data URIs if True)
      - llm_client: OpenAI-like client (optional)
      - llm_model: str (optional; required if llm_client is provided)
      - llm_prompt: str (optional; prompt for image captions)
      - pdfplumber_tables: bool = False (extract tables with pdfplumber)
      - chunk_size: int = 10  (process N pages at a time)
    """

    def __init__(self):
        super().__init__()
        self._html_converter = HtmlConverter()

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in ACCEPTED_FILE_EXTENSIONS:
            return True

        for prefix in ACCEPTED_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

        return False

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        # If both PyMuPDF and pdfminer are missing, fail like PptxConverter does
        if _fitx_exc is not None and _pdfminer_exc is not None:
            raise MissingDependencyException(
                MISSING_DEPENDENCY_MESSAGE.format(
                    converter=type(self).__name__,
                    extension=".pdf",
                    feature="pdf",
                )
            ) from (_fitx_exc[1] if _fitx_exc is not None else _pdfminer_exc[1])

        # Try advanced conversion with PyMuPDF; if that fails, fallback to pdfminer
        try:
            if _fitx_exc is None:
                return self._convert_with_fitz(file_stream, stream_info, **kwargs)
            else:
                raise RuntimeError("PyMuPDF not available")
        except Exception:
            # Last resort: pdfminer text extraction
            if _pdfminer_exc is None:
                try:
                    file_stream.seek(0)
                    text = pdfminer.high_level.extract_text(file_stream)  # type: ignore[attr-defined]
                    return DocumentConverterResult(markdown=(text or "").strip())
                except Exception:
                    return DocumentConverterResult(markdown="[Failed to extract PDF content]")
            else:
                return DocumentConverterResult(markdown="[Failed to extract PDF content]")

    # -----------------------
    # Internal helper methods
    # -----------------------

    def _open_pdf(self, file_stream: BinaryIO, stream_info: StreamInfo):
        """
        Open a PDF in lazy mode if possible.
        - If filename available, open directly (true lazy loading).
        - Otherwise, read into memory.
        """
        try:
            if getattr(stream_info, "filename", None):
                return fitz.open(stream_info.filename)  # lazy loading
            else:
                return fitz.open(stream=file_stream.read(), filetype="pdf")  # memory load
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {e}")

    def _convert_with_fitz(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        """
        Rich conversion using PyMuPDF with chunked page processing.
        """
        doc = self._open_pdf(file_stream, stream_info)

        keep_data_uris: bool = bool(kwargs.get("keep_data_uris", True))
        want_tables: bool = bool(kwargs.get("pdfplumber_tables", False))
        chunk_size: int = int(kwargs.get("chunk_size", 10))

        llm_client = kwargs.get("llm_client")
        llm_model = kwargs.get("llm_model")
        llm_prompt = kwargs.get("llm_prompt")

        md_parts: list[str] = []

        # Process document in chunks
        for start in range(0, len(doc), chunk_size):
            end = min(start + chunk_size, len(doc))
            chunk_pages = doc[start:end]

            for page_index, page in enumerate(chunk_pages, start=start + 1):
                # Page marker
                md_parts.append(f"\n\n<!-- Page {page_index} -->\n")

                # Text
                try:
                    txt = page.get_text("text")
                    if txt:
                        md_parts.append(txt.strip() + "\n")
                except Exception:
                    pass

                # Images
                try:
                    for img_idx, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image.get("image", b"")
                        image_ext = base_image.get("ext", "png") or "png"
                        content_type = f"image/{image_ext}"

                        image_stream = io.BytesIO(image_bytes)
                        image_info = StreamInfo(
                            mimetype=content_type,
                            extension=f".{image_ext}",
                            filename=f"page_{page_index}_img_{img_idx}.{image_ext}",
                        )

                        caption = ""
                        if llm_client is not None and llm_model is not None:
                            try:
                                caption = llm_caption(
                                    image_stream,
                                    image_info,
                                    client=llm_client,
                                    model=llm_model,
                                    prompt=llm_prompt,
                                ) or ""
                            except Exception:
                                caption = ""

                        if not caption:
                            caption = f"Image from page {page_index}, image {img_idx}"

                        caption = re.sub(r"[\r\n\[\]]", " ", caption)
                        caption = re.sub(r"\s+", " ", caption).strip()

                        if keep_data_uris:
                            b64 = base64.b64encode(image_bytes).decode("utf-8")
                            md_parts.append(f"\n![{caption}](data:{content_type};base64,{b64})\n")
                        else:
                            filename = f"page_{page_index}_img_{img_idx}.{image_ext}"
                            md_parts.append(f"\n![{caption}]({filename})\n")
                except Exception:
                    pass

                # Tables
                if want_tables:
                    if _pdfplumber_exc is None:
                        try:
                            file_stream.seek(0)
                            with pdfplumber.open(file_stream) as plumber_doc:  # type: ignore[name-defined]
                                if 0 <= (page_index - 1) < len(plumber_doc.pages):
                                    plumber_page = plumber_doc.pages[page_index - 1]
                                    tables = plumber_page.extract_tables()
                                    for table in tables or []:
                                        html_table = "<html><body><table>"
                                        first_row = True
                                        for row in table:
                                            cells = [(c if c is not None else "") for c in row]
                                            html_table += "<tr>"
                                            for cell in cells:
                                                if first_row:
                                                    html_table += f"<th>{html.escape(str(cell))}</th>"
                                                else:
                                                    html_table += f"<td>{html.escape(str(cell))}</td>"
                                            html_table += "</tr>"
                                            first_row = False
                                        html_table += "</table></body></html>"

                                        md_table = self._html_converter.convert_string(html_table, **kwargs).markdown.strip()
                                        if md_table:
                                            md_parts.append(md_table + "\n")
                        except Exception:
                            md_parts.append("\n[Failed to extract tables]\n")
                    else:
                        md_parts.append("\n[Tables not extracted: pdfplumber not installed]\n")

        return DocumentConverterResult(markdown=("".join(md_parts)).strip())

