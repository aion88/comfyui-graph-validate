#!/usr/bin/env python3
"""
Deterministic validator for a ComfyUI API-format workflow graph.

Runs with ZERO GPU and ZERO tokens: it checks a graph against an /object_info
snapshot (fetched live from a running server, or supplied as a JSON file for
offline / test use).

Checks (any failure => exit 1):
  1. shape        - each node is a dict with a string class_type and a dict inputs
  2. class_type   - every class_type exists in the object_info snapshot
  3. reference    - every ["node_id", output_index] input resolves: node exists
                    and output_index is within the referenced node's output count
  4. input_name   - every input key is a declared input of that node type
  5. asset / enum - every combo input value (model/lora/vae/... filenames, or
                    sampler/scheduler enums) is one object_info advertises
  6. cost_guard   - no node carries batch_size > 1
  7. preservation - OPT-IN. When a manifest declares a real subject that must
                    survive the graph (a product shot, a person, a logo),
                    the subject's pixels must actually reach the sampling space
                    and/or be composited back. A subject that only ever becomes
                    a structure hint (Canny, depth) while every sampler starts
                    from EmptyLatentImage is regenerated, not preserved.
                    Checks 1-6 cannot see this: every node exists and every
                    reference resolves.

Usage:
    graph_validate.py GRAPH.json [--object-info SNAPSHOT.json]
                                 [--comfy-url http://localhost:8188]
                                 [--preserve-subject NODE_ID]
                                 [--preserve-mode free|composite|relight]
                                 [--json]

Graph may be the bare API dict ({"1": {...}}) or the wrapped form
({"prompt": {...}}). Exit 0 => PASS, exit 1 => FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib import request as urlrequest

# Combo inputs whose value is an installed FILE (vs a plain enum). Only used to
# sharpen the error label ("missing model/asset" vs "invalid choice").
MODEL_INPUT_KEYS = {
    "ckpt_name",
    "unet_name",
    "vae_name",
    "clip_name",
    "clip_name1",
    "clip_name2",
    "clip_name3",
    "lora_name",
    "control_net_name",
    "style_model_name",
    "gguf_name",
    "ipadapter_file",
    "pulid_file",
}


def load_graph(path: str) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("prompt"), dict):
        return data["prompt"]
    return data


def _http_get(url: str, timeout: float = 30) -> bytes:
    """GET, carrying an API key when the environment holds one.

    Comfy Cloud answers `401` to an unauthenticated `/api/object_info`, so
    without this the validator cannot preflight a cloud graph at all. The key is
    read from the environment and never from an argument, so it stays out of
    shell history and out of any recorded command line."""
    headers = {"User-Agent": "comfyui-graph-validate"}
    key = os.environ.get("COMFY_API_KEY")
    if key:
        headers["X-API-Key"] = key
    req = urlrequest.Request(url, headers=headers)
    with urlrequest.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read()


def fetch_object_info(
    base_url: str, *, get=_http_get, timeout: float = 30
) -> dict[str, Any]:
    """The node catalogue, from a local server or from Comfy Cloud.

    Cloud serves it at `/api/object_info`; the unprefixed path answers HTTP 200
    with the web app's HTML (measured 2026-08-11). A validator that accepts that
    body validates the graph against a web page and calls every node unknown, so
    the body is required to be a non-empty JSON object before it is accepted."""
    base = base_url.rstrip("/")
    tried: list[str] = []
    for path in ("/object_info", "/api/object_info"):
        url = base + path
        tried.append(url)
        try:
            raw = get(url, timeout=timeout)
            data = json.loads(raw)
        except Exception:  # noqa: BLE001 - any failure just means "try the next path"
            continue
        if isinstance(data, dict) and data:
            return data
    raise ValueError(
        "no JSON object_info at any of " + ", ".join(tried) + " (a 200 text/html "
        "answer is the Comfy Cloud web app, not the node catalogue)"
    )


def load_object_info(args: argparse.Namespace) -> dict[str, Any]:
    if args.object_info:
        with open(args.object_info) as f:
            return json.load(f)
    url = args.comfy_url or os.environ.get("COMFY_URL") or "http://localhost:8188"
    return fetch_object_info(url)


def declared_inputs(schema: dict[str, Any]) -> dict[str, Any]:
    inp = schema.get("input", {}) or {}
    merged: dict[str, Any] = {}
    for group in ("required", "optional"):
        merged.update(inp.get(group, {}) or {})
    return merged


def output_count(schema: dict[str, Any]) -> int:
    out = schema.get("output", [])
    return len(out) if isinstance(out, list) else 0


def is_reference(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


DYNAMIC_COMBO_TYPE = "COMFY_DYNAMICCOMBO_V3"


def dynamic_options(spec: Any) -> dict[str, dict] | None:
    """The `{option key: that option's inputs}` of a V3 dynamic combo, else None.

    ComfyUI's newer schema API has an input whose *type* is a combo and whose
    chosen option brings its own sub-inputs. `/object_info` answers it as
    `["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": ..., "inputs": {...}}, ...]}]`
   : an options list of DICTS, where every other combo carries scalars.

    In an API-format graph the whole thing is FLAT: the option key sits on the
    input's own name and each sub-input becomes a sibling under a dotted key,
    e.g. `"mode": "tracked"` beside `"mode.crop_scale": 1.5`. The node still
    receives one dict, reassembled by `build_nested_inputs` in
    `comfy_api/latest/_io.py`.
    """
    if not isinstance(spec, list) or len(spec) < 2:
        return None
    if spec[0] != DYNAMIC_COMBO_TYPE or not isinstance(spec[1], dict):
        return None
    options = spec[1].get("options")
    if not isinstance(options, list):
        return None
    out: dict[str, dict] = {}
    for option in options:
        if isinstance(option, dict) and isinstance(option.get("key"), str):
            inner = option.get("inputs")
            out[option["key"]] = inner if isinstance(inner, dict) else {}
    return out or None


def declared_child(decl: dict, key: str) -> Any | None:
    """The spec behind a dotted `<dynamic>.<child>` key, or None.

    Without this, every sub-input of a dynamic combo reads as an undeclared
    input name (check 4 warning) and its VALUE is never checked at all: so a
    typo'd `mode.padding` sails through on the one input shape that carries the
    node's actual behaviour.
    """
    if "." not in key:
        return None
    parent, _, child = key.partition(".")
    options = dynamic_options(decl.get(parent))
    if not options:
        return None
    for inputs in options.values():
        for group_name in ("required", "optional"):
            group = inputs.get(group_name)
            if isinstance(group, dict) and child in group:
                return group[child]
    return None


def combo_choices(spec: Any) -> list | None:
    """A combo/enum input's choices, in either shape `/object_info` answers in.

    The old shape puts the list first: `[["euler", "heun", ...], {...}]`.
    The newer one names the type and carries the list in the options dict:
    `["COMBO", {"options": ["euler", ...]}]`.

    Reading only the first shape is a FAIL-OPEN on a money path, and it was
    silently live: `None` here means "not an enum", so check 5 skips the value
    entirely and a model filename that does not exist on the server validates
    clean and 400s at submit time. Found 2026-08-25 on
    `LatentUpscaleModelLoader.model_name`, which answers in the newer shape on
    Comfy Cloud today: every known filename in that category read as missing
    from a catalogue this function reported as empty.
    """
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    # A V3 dynamic combo's options are dicts, not scalars. Returning them raw
    # made check 5 compare a legal key like "tracked" against `{"key": ...}`
    # dicts and call it an invalid choice: a FAIL-CLOSED false positive that
    # refused a graph the server accepts and executes. Return the keys, so the
    # check still catches an option name that does not exist.
    dynamic = dynamic_options(spec)
    if dynamic is not None:
        return list(dynamic)
    if len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options")
        if isinstance(options, list):
            return options
    return None


# --- check 7: preservation -------------------------------------------------
#
# A ControlNet does not preserve a subject, it suggests its shape. A graph that
# claims to keep a real subject (a product shot, a person, a logo) but only
# ever feeds it through a structure preprocessor, samples every branch from an
# EmptyLatentImage, and never composites the original pixels back, produces a
# plausible lookalike. That is invisible to checks 1-6: every node exists, every
# reference resolves. So the manifest declares the intent and this check enforces
# it by reachability, which stays deterministic and needs no model.

# Nodes that reduce an image to a structure hint. Pixel identity dies here, so a
# forward walk from the subject treats them as absorbing barriers.
STRUCTURE_ONLY_TYPES = {
    "Canny",
    "Image Canny Filter",
    "MiDaS Depth Approximation",
    "MiDaS Mask Image",
    "ImageToMask",
    "Image To Mask",
}
# ...plus anything the preprocessor packs name this way.
STRUCTURE_ONLY_SUFFIXES = ("Preprocessor", "_Preprocessor")

# Nodes through which the subject's actual pixels enter the sampling space.
PIXEL_INGEST_TYPES = {
    "VAEEncode",
    "VAEEncodeTiled",
    "VAEEncodeForInpaint",
    "InpaintModelConditioning",
}
# Nodes that put original pixels back into the result.
RESTORE_TYPES = {
    "ImageCompositeMasked",
    "LatentCompositeMasked",
    "JoinImageWithAlpha",
    "Image Paste Crop",
    "Image Paste Crop by Location",
    "ImageUncropByMask",
    "Mask Paste Region",
    "InpaintStitch",
    "InpaintStitchImproved",
}
# Nodes that confine sampling to a masked region, so the subject is protected.
MASKED_LATENT_TYPES = {
    "SetLatentNoiseMask",
    "VAEEncodeForInpaint",
    "InpaintModelConditioning",
}


def is_structure_only(ctype: str) -> bool:
    return ctype in STRUCTURE_ONLY_TYPES or ctype.endswith(STRUCTURE_ONLY_SUFFIXES)


def node_refs(node: dict[str, Any]) -> list[tuple[str, str]]:
    """(input_key, referenced_node_id) for every wired input of this node."""
    out: list[tuple[str, str]] = []
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return out
    for key, value in inputs.items():
        if is_reference(value):
            out.append((key, str(value[0])))
    return out


def forward_closure(graph: dict[str, Any], start: str) -> set[str]:
    """Every node the subject's PIXELS can reach, stopping at structure-only
    nodes: past one of those the subject is a contour, not a photograph."""
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for nid, node in graph.items():
            if nid in seen or not isinstance(node, dict):
                continue
            if any(ref == current for _, ref in node_refs(node)):
                seen.add(nid)
                if not is_structure_only(str(node.get("class_type") or "")):
                    frontier.append(nid)
    return seen


def backward_closure(graph: dict[str, Any], start: str) -> set[str]:
    """Every node that feeds `start`, transitively."""
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        current = frontier.pop()
        node = graph.get(current)
        if not isinstance(node, dict):
            continue
        for _, ref in node_refs(node):
            if ref not in seen:
                seen.add(ref)
                frontier.append(ref)
    return seen


def sampler_ids(graph: dict[str, Any]) -> list[str]:
    """Any node taking a latent to denoise. Covers KSampler, KSamplerAdvanced,
    SamplerCustom and the wrappers, without hardcoding a class list."""
    out = []
    for nid, node in graph.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and (
            "latent_image" in inputs or "latent" in inputs
        ):
            out.append(nid)
    return out


def class_of(graph: dict[str, Any], nid: str) -> str:
    node = graph.get(nid)
    return str(node.get("class_type") or "") if isinstance(node, dict) else ""


def full_renoise_signal(inputs: dict[str, Any]) -> bool | None:
    """Whether a sampler's inputs mean "throw away the input latent's own
    noise and start fresh": `denoise >= 1.0` on a plain KSampler.

    KSamplerAdvanced has no `denoise` input at all (verified against
    object_info: its inputs are `add_noise`, `start_at_step`, `end_at_step`,
    `return_with_leftover_noise`), so reading `inputs["denoise"]` on one
    silently returns None and the check that exists to catch a full renoise
    never fires: confirmed: an identical graph with `KSampler(denoise=1.0)`
    IS flagged, and the same graph with
    `KSamplerAdvanced(add_noise="enable", start_at_step=0)` passes silently,
    though it fully renoises the same latent. `add_noise="enable"` at
    `start_at_step=0` is that node's spelling of the same intent.

    Returns None when the deciding field is wired to another node rather than
    a literal: not evaluable statically, same as a wired `denoise`."""
    if "denoise" in inputs:
        denoise = inputs.get("denoise")
        if is_reference(denoise):
            return None
        try:
            return float(denoise) >= 1.0
        except (TypeError, ValueError):
            return False
    if "add_noise" not in inputs:
        return False
    add_noise, start_at_step = inputs.get("add_noise"), inputs.get("start_at_step")
    if is_reference(add_noise) or is_reference(start_at_step):
        return None
    if str(add_noise).lower() != "enable":
        return False
    try:
        return float(start_at_step if start_at_step is not None else 0) <= 0
    except (TypeError, ValueError):
        return False


def preservation_findings(
    graph: dict[str, Any], subject_node: str, mode: str
) -> list[dict[str, str]]:
    """Check 7. `mode` is one of free | composite | relight."""
    findings: list[dict[str, str]] = []
    if mode == "free":
        return findings
    if mode not in ("composite", "relight"):
        return [
            {
                "node": "-",
                "check": "preservation",
                "detail": f"unknown preserve mode {mode!r} "
                "(expected one of free, composite, relight)",
                "severity": "error",
            }
        ]
    if subject_node not in graph:
        return [
            {
                "node": str(subject_node),
                "check": "preservation",
                "detail": f"preserve.subject_node '{subject_node}' is not in the graph",
                "severity": "error",
            }
        ]

    reachable = forward_closure(graph, subject_node)
    ingested = sorted(n for n in reachable if class_of(graph, n) in PIXEL_INGEST_TYPES)
    restored = sorted(n for n in reachable if class_of(graph, n) in RESTORE_TYPES)

    samplers = sampler_ids(graph)
    empty_latent_samplers = []
    for sid in samplers:
        upstream = backward_closure(graph, sid)
        if any(class_of(graph, u) == "EmptyLatentImage" for u in upstream):
            empty_latent_samplers.append(sid)

    # The product-relight signature: subject enters only as a structure map,
    # every sampler starts from noise, nothing composites the pixels back.
    if (
        not ingested
        and not restored
        and samplers
        and len(empty_latent_samplers) == len(samplers)
    ):
        findings.append(
            {
                "node": str(subject_node),
                "check": "preservation",
                "detail": (
                    f"subject node '{subject_node}' reaches the graph only as a "
                    "structure hint (no VAEEncode / inpaint conditioning), every "
                    f"sampler ({', '.join(samplers)}) starts from EmptyLatentImage, "
                    "and no node composites the original pixels back. The subject is "
                    "regenerated, not preserved. A ControlNet suggests a shape; it "
                    "does not preserve a subject."
                ),
                "severity": "error",
            }
        )
        return findings

    if mode == "composite" and not restored:
        findings.append(
            {
                "node": str(subject_node),
                "check": "preservation",
                "detail": (
                    "preserve.mode is 'composite' but no node downstream of subject "
                    f"'{subject_node}' restores its pixels (expected one of: "
                    f"{', '.join(sorted(RESTORE_TYPES))})"
                ),
                "severity": "error",
            }
        )

    if mode == "relight":
        for sid in samplers:
            upstream = backward_closure(graph, sid)
            carries_subject = any(u in reachable for u in upstream) or any(
                class_of(graph, u) in PIXEL_INGEST_TYPES and u in reachable
                for u in upstream
            )
            if not carries_subject:
                continue
            protected = any(class_of(graph, u) in MASKED_LATENT_TYPES for u in upstream)
            if protected:
                continue
            inputs = graph[sid].get("inputs") or {}
            full = full_renoise_signal(inputs)
            if full is None or not full:
                continue
            if "denoise" in inputs:
                signal = f"denoise={inputs.get('denoise')}"
            else:
                signal = (
                    f"add_noise={inputs.get('add_noise')!r} at "
                    f"start_at_step={inputs.get('start_at_step')}"
                )
            findings.append(
                {
                    "node": str(sid),
                    "check": "preservation",
                    "detail": (
                        f"sampler '{sid}' runs at {signal} over a latent "
                        f"carrying subject '{subject_node}', with no "
                        "SetLatentNoiseMask / VAEEncodeForInpaint / "
                        "InpaintModelConditioning protecting it: the subject is "
                        "fully renoised and will be rewritten"
                    ),
                    "severity": "error",
                }
            )

    return findings


def validate(
    graph: dict[str, Any],
    object_info: dict[str, Any],
    preserve: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return findings, each tagged severity 'error' (run-fatal / cost) or
    'warning' (surfaced but ComfyUI tolerates it, e.g. an undeclared input).

    `preserve`, when given, is the manifest's declaration of a real subject that
    must survive the graph: {"subject_node": "4", "mode": "composite"}."""
    findings: list[dict[str, str]] = []

    def err(node_id: str, check: str, detail: str, severity: str = "error") -> None:
        findings.append(
            {
                "node": str(node_id),
                "check": check,
                "detail": detail,
                "severity": severity,
            }
        )

    for nid, node in graph.items():
        # 1. shape
        if not isinstance(node, dict):
            err(nid, "shape", "node is not an object")
            continue
        ctype = node.get("class_type")
        if not isinstance(ctype, str) or not ctype:
            err(nid, "shape", "missing or non-string class_type")
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            err(nid, "shape", "missing or non-dict inputs")
            continue

        # 2. class_type exists
        schema = object_info.get(ctype)
        if schema is None:
            err(nid, "class_type", f"unknown node type '{ctype}' (not in object_info)")
            continue

        decl = declared_inputs(schema)

        for key, value in inputs.items():
            # 4. input name declared (warning: ComfyUI ignores extra inputs at
            #    runtime, so a stray key does not block the run: but surface it)
            spec = decl.get(key)
            if key not in decl:
                spec = declared_child(decl, key)
                if spec is None:
                    err(
                        nid,
                        "input_name",
                        f"'{ctype}' has no input named '{key}'",
                        severity="warning",
                    )
                    continue

            if is_reference(value):
                # 3. reference resolves
                ref_id, ref_idx = str(value[0]), value[1]
                ref_node = graph.get(ref_id)
                if not isinstance(ref_node, dict):
                    err(
                        nid,
                        "reference",
                        f"input '{key}' points to missing node '{ref_id}'",
                    )
                else:
                    ref_schema = object_info.get(ref_node.get("class_type"))
                    if ref_schema is not None and ref_idx >= output_count(ref_schema):
                        err(
                            nid,
                            "reference",
                            f"input '{key}' -> node '{ref_id}' output index {ref_idx} "
                            f"out of range (node has {output_count(ref_schema)} outputs)",
                        )
                continue

            # 5. asset / enum value installed
            choices = combo_choices(spec)
            if choices is not None and isinstance(value, (str, int, float)):
                if value not in choices:
                    if key in MODEL_INPUT_KEYS:
                        err(
                            nid,
                            "asset",
                            f"input '{key}'={value!r}: missing model/asset "
                            "(not in installed list)",
                        )
                    else:
                        err(nid, "enum", f"input '{key}'={value!r}: invalid choice")

            # 6. cost guard
            if key == "batch_size":
                try:
                    if int(value) > 1:
                        err(
                            nid,
                            "cost_guard",
                            f"batch_size={value} > 1 (cost guard: must be 1)",
                        )
                except (TypeError, ValueError):
                    pass

    # 7. preservation: only when the manifest declared a subject to keep.
    if preserve:
        findings.extend(
            preservation_findings(
                graph,
                str(preserve.get("subject_node", "")),
                str(preserve.get("mode", "free")),
            )
        )

    return findings


def errors_only(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    """The gate fails on errors; warnings are surfaced but never block."""
    return [f for f in findings if f.get("severity") == "error"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a ComfyUI API-format graph.")
    ap.add_argument("graph", help="path to the API-format workflow JSON")
    ap.add_argument("--object-info", help="path to an /object_info snapshot JSON")
    ap.add_argument(
        "--comfy-url",
        "--server",
        dest="comfy_url",
        help="ComfyUI base URL to fetch /object_info from (e.g. http://127.0.0.1:8188)",
    )
    ap.add_argument("--json", action="store_true", help="print a JSON report")
    ap.add_argument(
        "--preserve-subject",
        help="node id of a real subject that must survive the graph (check 7)",
    )
    ap.add_argument(
        "--preserve-mode",
        choices=("free", "composite", "relight"),
        default="composite",
        help="how the subject must survive (default: composite). "
        "Only used with --preserve-subject.",
    )
    args = ap.parse_args()

    try:
        graph = load_graph(args.graph)
    except (OSError, json.JSONDecodeError) as e:
        print(f"FAIL: cannot load graph: {e}", file=sys.stderr)
        return 1
    try:
        object_info = load_object_info(args)
    except Exception as e:  # noqa: BLE001 - surface any fetch/parse failure loudly
        print(f"FAIL: cannot load object_info: {e}", file=sys.stderr)
        return 1

    if not isinstance(graph, dict) or not graph:
        print("FAIL: graph is empty or not a node dict", file=sys.stderr)
        return 1

    preserve = (
        {"subject_node": args.preserve_subject, "mode": args.preserve_mode}
        if args.preserve_subject
        else None
    )
    findings = validate(graph, object_info, preserve=preserve)
    errors = errors_only(findings)
    warns = [f for f in findings if f["severity"] == "warning"]
    passed = not errors  # warnings never fail the gate
    report = {
        "result": "PASS" if passed else "FAIL",
        "nodes": len(graph),
        "error_count": len(errors),
        "warning_count": len(warns),
        "errors": errors,
        "warnings": warns,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if passed:
            print(
                f"PASS: {len(graph)} nodes validated, 0 errors, {len(warns)} warning(s)"
            )
        else:
            print(f"FAIL: {len(errors)} error(s) across {len(graph)} nodes:")
            for e in errors:
                print(f"  [node {e['node']}] {e['check']}: {e['detail']}")
        for w in warns:
            print(f"  warning [node {w['node']}] {w['check']}: {w['detail']}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
