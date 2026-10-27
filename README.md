# comfyui-graph-validate

Check a ComfyUI API-format graph against a server's `/object_info` before you queue it. No GPU, no tokens, no dependencies beyond the Python standard library.

## Why it exists

ComfyUI answers a bad graph with a bare HTTP 400, or worse, accepts a graph that is well formed and produces the wrong picture. I wrote this after a product-photo workflow shipped four "product photos" in which the product never entered the pixel path: it was only a Canny edge map, and every sampler started from an `EmptyLatentImage` at denoise 1.0. Every node existed and every reference resolved, so the first six checks passed. That became check 7, `preservation`.

## Checks

Any failure exits 1.

1. `shape`: each node is a dict with a string `class_type` and a dict `inputs`.
2. `class_type`: every node type exists in the `/object_info` snapshot.
3. `reference`: every `["node_id", output_index]` input resolves, and the output index is in range.
4. `input_name`: every input key is declared by that node type.
5. `asset / enum`: every combo value (checkpoint, LoRA, VAE, sampler, scheduler) is one the server advertises, so a missing model is named instead of failing at run time.
6. `cost_guard`: no node carries `batch_size` above 1.
7. `preservation` (opt-in): the declared subject node's pixels must reach the sampling space or be composited back.

Warnings (for example `control_after_generate`) are printed but never fail the run.

## Install

```
uv tool install .      # or: pip install .
```

## Usage

```
# against a snapshot file (offline, CI friendly)
graph-validate graph.json --object-info object_info.json

# against a running server
graph-validate graph.json --server http://127.0.0.1:8188

# machine-readable report
graph-validate graph.json --object-info object_info.json --json

# opt-in preservation check: node 4 is the image that must survive
graph-validate graph.json --object-info object_info.json \
  --preserve-subject 4 --preserve-mode composite
```

Exit code 0 means PASS, 1 means FAIL. The graph may be the bare API dict (`{"1": {...}}`) or the wrapped form (`{"prompt": {...}}`).

Get a snapshot with `curl http://127.0.0.1:8188/object_info > object_info.json`. For Comfy Cloud, set `COMFY_API_KEY` in the environment; it is read from there only, never from an argument.

From Python:

```python
from comfyui_graph_validate import load_graph, validate
findings = validate(load_graph("graph.json"), object_info)
```

## Tests

```
uv run pytest
```

The fixtures in `tests/fixtures/` are a small hand-trimmed `/object_info` snapshot and a few API graphs, including the broken one that checks 1 to 6 accept and check 7 rejects.

## Limitations

- The `object_info.json` fixture is a trimmed excerpt, not a full node catalogue. Use a snapshot from your own server for real work.
- The validator checks structure and advertised values. It does not run the graph, so it cannot tell you the picture will be good.
- Check 7 is a graph-topology heuristic. It needs you to name the subject node.
- Custom node types with unusual input schemas may be flagged; open an issue with the node's `/object_info` entry.

MIT license.
