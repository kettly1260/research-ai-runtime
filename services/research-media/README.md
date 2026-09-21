# Research Media Service

Scientific media indexing, figure parsing, and multimodal retrieval.

## Parser configuration

Parser/OCR providers are defined in a single hot-loaded YAML file. The
repository template is `../../config/providers.example.yaml`; the normal
runtime file is `../../config/providers.yaml` and is Git-ignored because it
may contain API credentials.

`PARSER_CONFIG_PATH` selects the mounted file, normally
`/config/providers.yaml`. URL, API key, model, API mode and options are kept
together in that file. MinerU, PaddleOCR, generic local HTTP OCR and generic
local CLI OCR all use the same provider registry.

The registry atomically reloads valid file changes and keeps the last
known-good provider set when a hot edit is invalid. Inspect
`GET /v1/providers` for the active config path, model/mode selection and any
reload error.

## Signed upload and result URLs

Provider workflows can select `transport: requests` for pre-signed object
storage URLs. This is used for MinerU file upload URLs and PaddleOCR result
resource URLs, while normal API submit/poll calls continue to use the default
async HTTP transport. The split keeps authentication on provider API calls and
avoids attaching provider credentials to object-storage/CDN requests.

See the root `README.md` for MinerU/PaddleOCR examples, local OCR templates,
privacy routing, and Unraid paths.
