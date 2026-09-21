from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
import lancedb
import pyarrow as pa
from contracts import MediaRecord
from ..config import MEDIA_DATA_DIR


def get_configured_dimensions() -> tuple[int, int]:
    """Reads configured multimodal and DINO vector dimensions from environment or models configuration."""
    multimodal_dim = int(os.getenv("MULTIMODAL_EMBEDDING_DIM", "0") or "0")
    dino_dim = int(os.getenv("DINO_EMBEDDING_DIM", "0") or "0")

    candidate_paths = [
        os.getenv("MODELS_CONFIG_PATH"),
        "/config/models.yaml",
        "/config/models.example.yaml",
        "config/models.yaml",
        "config/models.example.yaml",
    ]
    for path in candidate_paths:
        if path and os.path.exists(path):
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                models = data.get("models", {})
                if multimodal_dim == 0 and "jina-clip-v2" in models:
                    j_cfg = models["jina-clip-v2"]
                    multimodal_dim = int(j_cfg.get("truncate_dimension") or j_cfg.get("output_dimension") or 512)
                if dino_dim == 0 and "dinov3" in models:
                    d_cfg = models["dinov3"]
                    dino_dim = int(d_cfg.get("output_dimension") or 384)
                if multimodal_dim > 0 and dino_dim > 0:
                    break
            except Exception:
                pass

    return (multimodal_dim or 512, dino_dim or 384)


class LanceMediaStore:
    """Embedded LanceDB Media Store with decoupled vector dimensions and strict dimension validation."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        multimodal_dim: Optional[int] = None,
        dino_dim: Optional[int] = None,
    ):
        cfg_multi, cfg_dino = get_configured_dimensions()
        self.data_dir = data_dir or MEDIA_DATA_DIR
        self.multimodal_dim = multimodal_dim or cfg_multi
        self.dino_dim = dino_dim or cfg_dino
        self.db = None
        self.table = None
        self.table_name = f"media_{self.multimodal_dim}_{self.dino_dim}"
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            self.db = lancedb.connect(self.data_dir)
            self._init_table()
        except Exception as exc:
            pass

    def _schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("media_id", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("source_uri", pa.string()),
            pa.field("source_native_id", pa.string()),
            pa.field("parent_uri", pa.string()),
            pa.field("mime_type", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("figure_number", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("ocr_text", pa.string()),
            pa.field("vlm_summary", pa.string()),
            pa.field("hash", pa.string()),
            pa.field("created_at", pa.float64()),
            pa.field("updated_at", pa.float64()),
            pa.field("multimodal_vector", pa.list_(pa.float32(), self.multimodal_dim)),
            pa.field("dino_vector", pa.list_(pa.float32(), self.dino_dim)),
            pa.field("parser_provider", pa.string()),
            pa.field("parser_version", pa.string()),
            pa.field("embedding_model", pa.string()),
            pa.field("embedding_version", pa.string()),
        ])

    def _init_table(self):
        try:
            if hasattr(self.db, "list_tables"):
                existing_tables = list(self.db.list_tables())
            else:
                existing_tables = list(self.db.table_names())
        except Exception:
            existing_tables = []

        if self.table_name not in existing_tables:
            self.table = self.db.create_table(
                self.table_name,
                schema=self._schema(),
                mode="create",
            )
        else:
            self.table = self.db.open_table(self.table_name)

    def insert(self, record: MediaRecord):
        # Strict dimension validation
        if record.multimodal_vector is not None:
            actual_len = len(record.multimodal_vector)
            if actual_len != self.multimodal_dim:
                raise ValueError(
                    f"Multimodal vector dimension mismatch: expected {self.multimodal_dim}, got {actual_len}"
                )
        if record.dino_vector is not None:
            actual_dino_len = len(record.dino_vector)
            if actual_dino_len != self.dino_dim:
                raise ValueError(
                    f"DINO vector dimension mismatch: expected {self.dino_dim}, got {actual_dino_len}"
                )

        data = [record.model_dump()]
        self.table.add(data)

    def get_by_hash(self, hash_val: str) -> Optional[Dict[str, Any]]:
        try:
            results = self.table.search().where(f"hash = '{hash_val}'").limit(1).to_list()
            return results[0] if results else None
        except Exception:
            return None

    def get_by_id(self, media_id: str) -> Optional[Dict[str, Any]]:
        try:
            results = self.table.search().where(f"media_id = '{media_id}'").limit(1).to_list()
            return results[0] if results else None
        except Exception:
            return None

    def search_multimodal(
        self,
        vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if len(vector) != self.multimodal_dim:
            raise ValueError(
                f"Query multimodal vector dimension mismatch: expected {self.multimodal_dim}, got {len(vector)}"
            )
        query = self.table.search(vector, vector_column_name="multimodal_vector").limit(top_k)
        if filter_expr:
            query = query.where(filter_expr)
        return query.to_list()

    def search_dino(
        self,
        vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if len(vector) != self.dino_dim:
            raise ValueError(
                f"Query DINO vector dimension mismatch: expected {self.dino_dim}, got {len(vector)}"
            )
        query = self.table.search(vector, vector_column_name="dino_vector").limit(top_k)
        if filter_expr:
            query = query.where(filter_expr)
        return query.to_list()

    def count(self) -> int:
        try:
            return len(self.table)
        except Exception:
            return 0


MEDIA_STORE = LanceMediaStore()
