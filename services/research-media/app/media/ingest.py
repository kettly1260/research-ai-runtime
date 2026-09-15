import hashlib
import uuid
import time
from typing import Any, Dict, List, Optional
from contracts import ParsedDocument, DocumentFigure, MediaRecord
from .store import LanceMediaStore, MEDIA_STORE
from ..client.gateway import GatewayClient


class MediaIngestor:
    """Ingests figures and media from ParsedDocument into LanceDB with vector embeddings."""

    def __init__(self, store: Optional[LanceMediaStore] = None, gateway: Optional[GatewayClient] = None):
        self.store = store or MEDIA_STORE
        self.gateway = gateway or GatewayClient()

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def ingest_document(
        self,
        doc: ParsedDocument,
        source_uri: str = "",
        embedding_model: str = "jina-clip-v2",
        dino_model: str = "dinov3",
    ) -> Dict[str, Any]:
        ingested = []
        skipped = 0

        for fig in doc.figures:
            if not fig.image:
                continue

            # Compute deduplication hash
            content_hash = self._compute_hash(fig.image)
            existing = self.store.get_by_hash(content_hash)
            if existing:
                skipped += 1
                continue

            # Compute multimodal vector (jina-clip-v2) and DINO vector (dinov3)
            multimodal_vec = None
            dino_vec = None

            try:
                mm_vecs = await self.gateway.get_image_embedding([fig.image], model=embedding_model)
                if mm_vecs:
                    multimodal_vec = mm_vecs[0]
            except Exception as exc:
                print(f"[MediaIngestor] Warning: Multimodal embedding failed for figure {fig.figure_id}: {exc}", flush=True)

            try:
                d_vecs = await self.gateway.get_dino_embedding([fig.image], model=dino_model)
                if d_vecs:
                    dino_vec = d_vecs[0]
            except Exception as exc:
                print(f"[MediaIngestor] Warning: DINO embedding failed for figure {fig.figure_id}: {exc}", flush=True)

            media_id = str(uuid.uuid4())
            record = MediaRecord(
                media_id=media_id,
                source_type=fig.figure_type or "figure",
                source_uri=source_uri or doc.document_id,
                source_native_id=fig.figure_id,
                parent_uri=doc.document_id,
                mime_type="image/png",
                page=fig.page,
                figure_number=fig.figure_id,
                caption=fig.caption,
                ocr_text=fig.ocr_text,
                hash=content_hash,
                created_at=time.time(),
                updated_at=time.time(),
                multimodal_vector=multimodal_vec,
                dino_vector=dino_vec,
                parser_provider=doc.provider_info.provider_name if doc.provider_info else None,
                embedding_model=embedding_model,
            )

            self.store.insert(record)
            ingested.append(record.media_id)

        return {
            "document_id": doc.document_id,
            "ingested_count": len(ingested),
            "skipped_duplicates": skipped,
            "media_ids": ingested,
        }


MEDIA_INGESTOR = MediaIngestor()
