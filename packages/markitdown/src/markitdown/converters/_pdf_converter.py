import io
import base64
import fitz  # PyMuPDF
import html
import re

from typing import BinaryIO, Any
from ._html_converter import HtmlConverter
from ._llm_caption import llm_caption
from .._base_converter import DocumentConverter, DocumentConverterResult
from .._stream_info import StreamInfo

class PdfConverter(DocumentConverter):
    """
    Converts PDF files to Markdown. Extracts text, images, tables, and charts.
    """

    def __init__(self):
        super().__init__()
        self._html_converter = HtmlConverter()

    def accepts(self, file_stream: BinaryIO, stream_info, **kwargs: Any) -> bool:
        return (stream_info.extension or "").lower() == ".pdf"

    def convert(self, file_stream: BinaryIO, stream_info, **kwargs: Any) -> DocumentConverterResult:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        md_content = ""

        for page_num, page in enumerate(doc, start=1):
            md_content += f"\n\n<!-- Page {page_num} -->\n"

            # Text blocks
            text = page.get_text("text")
            if text:
                md_content += text.strip() + "\n"

            # Images and charts (treated the same)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_stream = io.BytesIO(image_bytes)

                stream_info = StreamInfo(
                    mimetype=f"image/{image_ext}",
                    extension=f".{image_ext}",
                    filename=f"page_{page_num}_img_{img_index}.{image_ext}",
                )

                caption = ""
                llm_client = kwargs.get("llm_client")
                llm_model = kwargs.get("llm_model")
                if llm_client and llm_model:
                    try:
                        caption = llm_caption(
                            image_stream,
                            stream_info,
                            client=llm_client,
                            model=llm_model,
                            prompt=kwargs.get("llm_prompt"),
                        )
                    except Exception:
                        caption = "Image from PDF"

                b64 = base64.b64encode(image_bytes).decode("utf-8")
                md_content += f"\n![{caption}](data:image/{image_ext};base64,{b64})\n"

            # Tables (optional: requires pdfplumber)
            if kwargs.get("pdfplumber_tables"):
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(doc.write())) as plumber_doc:
                        plumber_page = plumber_doc.pages[page_num - 1]
                        for table in plumber_page.extract_tables():
                            html_table = "<html><body><table>"
                            for row in table:
                                html_table += "<tr>"
                                for cell in row:
                                    html_table += "<td>" + html.escape(cell or "") + "</td>"
                                html_table += "</tr>"
                            html_table += "</table></body></html>"
                            md_content += self._html_converter.convert_string(html_table, **kwargs).markdown.strip() + "\n"
                except Exception:
                    md_content += "\n\n[Failed to extract tables]\n"

        return DocumentConverterResult(markdown=md_content.strip())

