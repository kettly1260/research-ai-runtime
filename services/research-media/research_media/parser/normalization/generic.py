import uuid
from typing import Any, Dict, List, Optional
import jmespath
from contracts import (
    ParsedDocument,
    DocumentBlock,
    DocumentFigure,
    DocumentTable,
    DocumentPage,
)
from ..models import ResponseMappingConfig


def _eval_jmespath(expr: Optional[str], data: Any, default: Any = None) -> Any:
    if not expr or data is None:
        return default
    try:
        val = jmespath.search(expr, data)
        return val if val is not None else default
    except Exception:
        return default


def normalize_response(
    raw_result: Dict[str, Any],
    mapping: ResponseMappingConfig,
    doc_id: Optional[str] = None,
) -> ParsedDocument:
    """Declaratively normalizes any provider raw JSON response into ParsedDocument via JMESPath."""
    document_id = doc_id or str(uuid.uuid4())

    # 1. Extract markdown or text
    markdown_text = _eval_jmespath(mapping.markdown_path, raw_result, "")
    if not markdown_text and mapping.text_path:
        markdown_text = _eval_jmespath(mapping.text_path, raw_result, "")
    if not isinstance(markdown_text, str):
        markdown_text = str(markdown_text or "")

    # 2. Extract pages
    raw_pages = _eval_jmespath(mapping.pages_path, raw_result, [])
    pages: List[DocumentPage] = []
    if isinstance(raw_pages, list):
        for idx, p in enumerate(raw_pages):
            if isinstance(p, dict):
                pages.append(DocumentPage(
                    page_number=p.get("page_number", idx + 1),
                    width=p.get("width"),
                    height=p.get("height"),
                    text=p.get("text"),
                ))

    # 3. Extract blocks
    raw_blocks = _eval_jmespath(mapping.blocks_path, raw_result, [])
    blocks: List[DocumentBlock] = []
    if isinstance(raw_blocks, list):
        for idx, b in enumerate(raw_blocks):
            if isinstance(b, dict):
                b_type = str(_eval_jmespath(mapping.block_type_path, b, "text")).lower()
                b_content = _eval_jmespath(mapping.block_content_path, b, "")
                if isinstance(b_content, list):
                    # In some OCR APIs, content is a list of sub-blocks or lines
                    lines = [line.get("text", "") if isinstance(line, dict) else str(line) for line in b_content]
                    b_content = "\n".join(lines)
                b_page = _eval_jmespath(mapping.block_page_path, b, 1)
                b_bbox = _eval_jmespath(mapping.block_bbox_path, b, None)

                # Normalize block type
                norm_type = "text"
                if any(h in b_type for h in ("head", "title")):
                    norm_type = "heading"
                elif "table" in b_type:
                    norm_type = "table"
                elif any(f in b_type for f in ("figure", "image", "chart")):
                    norm_type = "figure"
                elif "formula" in b_type:
                    norm_type = "formula"

                blocks.append(DocumentBlock(
                    block_id=f"{document_id}-b{idx}",
                    type=norm_type,
                    content=str(b_content),
                    page=int(b_page) if isinstance(b_page, (int, float, str)) and str(b_page).isdigit() else 1,
                    bbox=b_bbox if isinstance(b_bbox, list) else None,
                ))

    # 4. Extract figures
    raw_figures = _eval_jmespath(mapping.figures_path, raw_result, [])
    figures: List[DocumentFigure] = []
    if isinstance(raw_figures, list):
        for idx, f in enumerate(raw_figures):
            if isinstance(f, dict):
                fig_id = _eval_jmespath(mapping.figure_id_path, f, f"{document_id}-fig{idx}")
                fig_img = _eval_jmespath(mapping.figure_image_path, f, None)
                fig_caption = _eval_jmespath(mapping.figure_caption_path, f, None)
                fig_page = _eval_jmespath(mapping.figure_page_path, f, 1)
                fig_bbox = _eval_jmespath(mapping.figure_bbox_path, f, None)
                fig_ocr = _eval_jmespath(mapping.figure_ocr_path, f, None)

                figures.append(DocumentFigure(
                    figure_id=str(fig_id),
                    page=int(fig_page) if isinstance(fig_page, (int, float, str)) and str(fig_page).isdigit() else 1,
                    bbox=fig_bbox if isinstance(fig_bbox, list) else None,
                    image=fig_img,
                    caption=fig_caption,
                    ocr_text=fig_ocr,
                ))

    # 5. Extract tables
    raw_tables = _eval_jmespath(mapping.tables_path, raw_result, [])
    tables: List[DocumentTable] = []
    if isinstance(raw_tables, list):
        for idx, t in enumerate(raw_tables):
            if isinstance(t, dict):
                tbl_id = _eval_jmespath(mapping.table_id_path, t, f"{document_id}-tbl{idx}")
                tbl_md = _eval_jmespath(mapping.table_markdown_path, t, None)
                tbl_html = _eval_jmespath(mapping.table_html_path, t, None)
                tbl_page = _eval_jmespath(mapping.table_page_path, t, 1)
                tbl_caption = _eval_jmespath(mapping.table_caption_path, t, None)

                tables.append(DocumentTable(
                    table_id=str(tbl_id),
                    page=int(tbl_page) if isinstance(tbl_page, (int, float, str)) and str(tbl_page).isdigit() else 1,
                    caption=tbl_caption,
                    html=tbl_html,
                    markdown=tbl_md,
                ))

    # If no markdown text was extracted but blocks exist, compose from blocks
    if not markdown_text and blocks:
        markdown_text = "\n\n".join(b.content for b in blocks if b.content)

    return ParsedDocument(
        document_id=document_id,
        markdown=markdown_text,
        metadata={"raw_length": len(str(raw_result))},
        pages=pages,
        blocks=blocks,
        figures=figures,
        tables=tables,
        raw_provider_result=raw_result,
    )
