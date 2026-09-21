from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class AuthConfig(BaseModel):
    """Declarative authentication configuration."""
    type: Literal["none", "bearer", "header", "query"] = "none"
    token: Optional[str] = None
    token_env: Optional[str] = None
    header_name: str = "Authorization"
    query_param: str = "api_key"

    def resolve_token(self) -> Optional[str]:
        if self.token_env and os.getenv(self.token_env):
            return os.getenv(self.token_env)
        return self.token


class RequestField(BaseModel):
    """Field definition in request payload."""
    value: Optional[Any] = None
    value_from: Optional[str] = None  # e.g., "options.language", "file_url", "privacy"
    transform: Optional[Literal["base64", "str", "int", "bool"]] = None


class RequestFileConfig(BaseModel):
    """File input handling in HTTP request."""
    mode: Literal["multipart", "base64", "url", "none"] = "multipart"
    field: str = "file"


class RequestTemplate(BaseModel):
    """Declarative template for building outbound HTTP requests."""
    method: Literal["GET", "POST", "PUT"] = "POST"
    path: str = ""
    encoding: Literal["json", "multipart", "urlencoded"] = "json"
    headers: Dict[str, str] = Field(default_factory=dict)
    fields: Dict[str, Union[RequestField, Any]] = Field(default_factory=dict)
    file: RequestFileConfig = Field(default_factory=RequestFileConfig)


class AsyncPollingConfig(BaseModel):
    """Configuration for asynchronous submit -> poll -> fetch pattern."""
    enabled: bool = False
    job_id_path: str = "data.task_id"  # JMESPath expression to extract job ID
    poll_path: str = "/tasks/{job_id}"
    poll_method: Literal["GET", "POST"] = "GET"
    status_path: str = "data.status"   # JMESPath expression to check status
    success_values: List[str] = Field(default_factory=lambda: ["success", "completed", "done", "2"])
    failure_values: List[str] = Field(default_factory=lambda: ["failed", "error", "-1"])
    result_path: Optional[str] = None  # JMESPath to extract final result, or None for entire response
    result_fetch_path: Optional[str] = None  # Separate endpoint to download result if needed: e.g. "/tasks/{job_id}/result"
    poll_interval_seconds: float = 1.0
    max_poll_seconds: float = 300.0


class ResponseMappingConfig(BaseModel):
    """JMESPath expressions to extract unified ParsedDocument components from provider response."""
    markdown_path: Optional[str] = "markdown || text || data.markdown || data.text"
    text_path: Optional[str] = None
    pages_path: Optional[str] = "pages || data.pages"
    blocks_path: Optional[str] = "blocks || data.blocks || data.layout_blocks || result"
    figures_path: Optional[str] = "figures || data.figures"
    tables_path: Optional[str] = "tables || data.tables"
    formulas_path: Optional[str] = "formulas || data.formulas"

    # Block item mapping
    block_type_path: Optional[str] = "type || label"
    block_content_path: Optional[str] = "content || text || res"
    block_page_path: Optional[str] = "page"
    block_bbox_path: Optional[str] = "bbox"

    # Figure item mapping
    figure_id_path: Optional[str] = "id || figure_id"
    figure_image_path: Optional[str] = "image || img || image_base64 || image_url"
    figure_caption_path: Optional[str] = "caption"
    figure_page_path: Optional[str] = "page"
    figure_bbox_path: Optional[str] = "bbox"
    figure_ocr_path: Optional[str] = "ocr_text || ocr"

    # Table item mapping
    table_id_path: Optional[str] = "id || table_id"
    table_markdown_path: Optional[str] = "markdown || content"
    table_html_path: Optional[str] = "html"
    table_page_path: Optional[str] = "page"
    table_caption_path: Optional[str] = "caption"


class LifecycleConfig(BaseModel):
    mode: Literal["external", "persistent", "on_demand", "model_on_demand"] = "external"
    resource: Optional[str] = None
    startup_timeout_seconds: float = 60.0
    idle_timeout_seconds: Optional[float] = None


class WorkflowStep(BaseModel):
    """Declarative workflow step within GenericHttpDriver."""
    name: str
    type: Literal["http", "request", "poll", "extract"] = "http"
    condition: Optional[str] = None
    method: Literal["GET", "POST", "PUT", "DELETE"] = "POST"
    transport: Literal["httpx", "requests"] = "httpx"
    url: Optional[str] = None
    path: Optional[str] = None
    encoding: Literal["json", "multipart", "binary_file", "raw_file", "urlencoded"] = "json"
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Any] = None
    file_field: Optional[str] = None
    exports: Dict[str, str] = Field(default_factory=dict)
    use_auth: bool = True
    response_format: Literal["json", "jsonl", "text", "binary"] = "json"
    action: Optional[Literal["download_zip", "download_and_extract_zip"]] = None
    retry_statuses: List[int] = Field(default_factory=list)
    retry_attempts: int = 1
    retry_delay_seconds: float = 1.0

    # Polling parameters (used when type == "poll")
    status_path: Optional[str] = None
    success_values: List[str] = Field(default_factory=lambda: ["done", "success", "completed", "2"])
    failure_values: List[str] = Field(default_factory=lambda: ["failed", "error", "-1"])
    poll_interval_seconds: float = 2.0
    max_poll_seconds: float = 300.0

    model_config = ConfigDict(extra="ignore")


class ProviderDefinition(BaseModel):
    """Complete declarative definition for a Parser Provider."""
    name: Optional[str] = None
    driver: str = "http"
    location: Literal["remote", "local"] = "remote"
    endpoint: Optional[str] = None
    endpoint_env: Optional[str] = None
    enabled: bool = True
    priority: int = 100
    timeout: float = 300.0
    timeout_seconds: Optional[float] = None
    capabilities: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    api_mode: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    request: RequestTemplate = Field(default_factory=RequestTemplate)
    async_config: Optional[AsyncPollingConfig] = Field(default=None, alias="async")
    workflow: Optional[List[WorkflowStep]] = None
    response: ResponseMappingConfig = Field(default_factory=ResponseMappingConfig)

    # Generic command driver protocol fields
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    output_format: Literal["json", "text", "lines"] = "json"

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    def resolve_endpoint(self) -> str:
        if self.endpoint_env and os.getenv(self.endpoint_env):
            return os.getenv(self.endpoint_env)
        return self.endpoint or ""
