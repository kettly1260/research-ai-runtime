import os
import time
from typing import Any, Dict, List, Optional
import lancedb
import pyarrow as pa
from contracts import MediaRecord
from ..config import MEDIA_DATA_DIR


class LanceMediaStore:
    """Embedded LanceDB Media Store."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or MEDIA_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        self.db = lancedb.connect(self.data_dir)
        self.table_name = "media"
        self._init_table()

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
            pa.field("multimodal_vector", pa.list_(pa.float32(), 512)),
            pa.field("dino_vector", pa.list_(pa.float32(), 384)),
            pa.field("parser_provider", pa.string()),
            pa.field("parser_version", pa.string()),
            pa.field("embedding_model", pa.string()),
            pa.field("embedding_version", pa.string()),
        ])

    def _init_table(self):
        try:
            existing_tables = list(self.db.table_names())
        except Exception:
            existing_tables = []
        if self.table_name not in existing_tables:
            # Create an empty table with schema
            self.table = self.db.create_table(
                self.table_name,
                schema=self._schema(),
                mode="create",
            )
        else:
            self.table = self.db.open_table(self.table_name)

    def insert(self, record: MediaRecord):
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
