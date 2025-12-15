# Trace_mcp MCP Server

This directory provides an MCP-style HTTP server that exposes
trace diagnosis training scripts as tools for LLMs or other clients.

Currently supported training tools:
- `aiops_3c6c`: runs `TraDiag/trace_service_diag/train_aiops3c6c.py`
- `aiops_svnd`: runs `TraDiag/trace_svnd_diag/train_aiops_svnd.py`

## Build and run with Docker

From the repository root:

```bash
docker build -f Trace_mcp/Dockerfile -t trace-mcp-server .
docker run -p 8889:8889 trace-mcp-server
```

## HTTP API

- Endpoint: `POST http://localhost:8889/mcp/train`

### Request body

```json
{
  "method": "aiops_3c6c",
  "extra_args": {
    "data_root": "TraDiag/dataset/aiops_v4_1e52e3",
    "save_dir": "outputs/aiops_3c6c/2025-11-21",
    "task": "superfine",
    "epochs": 5
  }
}
```

For `aiops_svnd`:

```json
{
  "method": "aiops_svnd",
  "extra_args": {
    "data_root": "TraDiag/trace_svnd_diag/dataset/aiops_svnd",
    "save_dir": "outputs/aiops_svnd/2025-11-21",
    "save_pt": "outputs/aiops_svnd/2025-11-21/model.pt",
    "epochs": 10
  }
}
```

All keys in `extra_args` are converted to CLI flags by:
- converting `_` to `-`
- prefixing with `--`

For example, `"save_dir"` becomes `--save-dir`.

### Response body

```json
{
  "status": "success",
  "exit_code": 0,
  "log": "... stdout and stderr from the training script ..."
}
```

