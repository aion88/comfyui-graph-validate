"""Tests for the ComfyUI graph validator. GPU-free and token-free: every test runs
against a static /object_info snapshot fixture that mirrors real ComfyUI node
schemas.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from comfyui_graph_validate.core import load_graph, preservation_findings, validate

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
OBJECT_INFO_PATH = FIXTURES / "object_info.json"
GRAPH_PATH = FIXTURES / "flux2_t2i_api.json"
CONTROLNET_GRAPH_PATH = FIXTURES / "flux2_controlnet_lora_api.json"


@pytest.fixture
def object_info() -> dict:
    return json.loads(OBJECT_INFO_PATH.read_text())


@pytest.fixture
def graph() -> dict:
    """A fresh, deep-copied valid graph each test can mutate in isolation."""
    return copy.deepcopy(json.loads(GRAPH_PATH.read_text()))


@pytest.fixture
def controlnet_graph() -> dict:
    """A complex graph: Flux.2 + LoRA + depth-ControlNet (multi-output wiring)."""
    return copy.deepcopy(json.loads(CONTROLNET_GRAPH_PATH.read_text()))


def checks(findings: list[dict]) -> set[str]:
    return {e["check"] for e in findings}


def errors(findings: list[dict]) -> list[dict]:
    """Fatal findings only: the ones that block a run / fail the gate."""
    return [e for e in findings if e.get("severity", "error") == "error"]


def warnings(findings: list[dict]) -> list[dict]:
    return [e for e in findings if e.get("severity") == "warning"]


# --- the happy path -------------------------------------------------------


def test_valid_flux2_graph_has_no_errors(graph, object_info):
    # The graph carries `control_after_generate` (a real UI widget ComfyUI
    # ignores): that is a warning, never a fatal error.
    assert errors(validate(graph, object_info)) == []


def test_control_after_generate_is_a_warning_not_an_error(graph, object_info):
    # Grounded in the live server: KSampler does not declare
    # control_after_generate, yet ComfyUI runs graphs that include it. So it
    # must surface as a warning and must NOT fail the gate.
    findings = validate(graph, object_info)
    cag = [e for e in findings if "control_after_generate" in e["detail"]]
    assert cag, "expected control_after_generate to be surfaced"
    assert all(e["severity"] == "warning" for e in cag)


# --- check 1: shape -------------------------------------------------------


def test_node_missing_class_type_is_flagged(graph, object_info):
    del graph["1"]["class_type"]
    assert "shape" in checks(validate(graph, object_info))


def test_node_with_non_dict_inputs_is_flagged(graph, object_info):
    graph["1"]["inputs"] = "not a dict"
    assert "shape" in checks(validate(graph, object_info))


# --- check 2: class_type exists ------------------------------------------


def test_unknown_node_type_is_flagged(graph, object_info):
    graph["1"]["class_type"] = "NotARealComfyNode"
    errs = validate(graph, object_info)
    assert any(e["check"] == "class_type" and e["node"] == "1" for e in errs)


# --- check 3: node references resolve ------------------------------------


def test_reference_to_missing_node_is_flagged(graph, object_info):
    graph["8"]["inputs"]["model"] = ["999", 0]
    errs = validate(graph, object_info)
    assert any(e["check"] == "reference" and "999" in e["detail"] for e in errs)


def test_reference_output_index_out_of_range_is_flagged(graph, object_info):
    # VAELoader (node 3) has exactly 1 output; index 5 is out of range.
    graph["9"]["inputs"]["vae"] = ["3", 5]
    errs = validate(graph, object_info)
    assert any(e["check"] == "reference" for e in errs)


def test_valid_reference_is_not_flagged(graph, object_info):
    # sanity: the existing ["2", 0] CLIP wiring must not trip the reference check
    assert not any(e["check"] == "reference" for e in validate(graph, object_info))


# --- check 4: undeclared input names (warning, not fatal) ----------------


def test_undeclared_input_name_is_a_warning(graph, object_info):
    # ComfyUI ignores inputs a node doesn't declare, so a stray key is a
    # warning (surfaced): not a fatal error that would block a valid run.
    graph["8"]["inputs"]["wobble"] = 3
    findings = validate(graph, object_info)
    hit = [
        e for e in findings if e["check"] == "input_name" and "wobble" in e["detail"]
    ]
    assert hit
    assert all(e["severity"] == "warning" for e in hit)
    assert not any(
        e["check"] == "input_name" and e["severity"] == "error" for e in findings
    )


# --- check 5: assets / enum values installed -----------------------------


def test_missing_model_file_is_flagged_as_asset(graph, object_info):
    graph["1"]["inputs"]["unet_name"] = "not-installed.gguf"
    errs = validate(graph, object_info)
    assert any(e["check"] == "asset" and e["node"] == "1" for e in errs)


def test_invalid_enum_choice_is_flagged(graph, object_info):
    graph["8"]["inputs"]["sampler_name"] = "not_a_real_sampler"
    errs = validate(graph, object_info)
    assert any(e["check"] == "enum" for e in errs)


def test_installed_model_and_valid_enum_are_not_flagged(graph, object_info):
    errs = validate(graph, object_info)
    assert not any(e["check"] in ("asset", "enum") for e in errs)


def test_a_combo_declared_the_newer_way_still_gates_its_value(graph, object_info):
    """`["COMBO", {"options": [...]}]`, which Comfy Cloud answers with today.

    Reading only the old `[[...]]` shape did not make this check noisy, it made
    it SILENT: an unparsed combo reads as "not an enum", the value is never
    compared, and a model filename that does not exist on the server validates
    clean and 400s at submit time, after the money is committed. Measured
    2026-08-25 on `LatentUpscaleModelLoader.model_name`.
    """
    node = graph["1"]["class_type"]
    old = object_info[node]["input"]["required"]["unet_name"]
    choices = old[0] if isinstance(old[0], list) else old[1]["options"]
    object_info[node]["input"]["required"]["unet_name"] = [
        "COMBO",
        {"options": list(choices)},
    ]

    assert not any(e["check"] == "asset" for e in validate(graph, object_info))

    graph["1"]["inputs"]["unet_name"] = "not-installed.gguf"
    errs = validate(graph, object_info)
    assert any(e["check"] == "asset" and e["node"] == "1" for e in errs)


# --- check 6: cost guard --------------------------------------------------


def test_batch_size_greater_than_one_is_flagged(graph, object_info):
    graph["6"]["inputs"]["batch_size"] = 4
    errs = validate(graph, object_info)
    assert any(e["check"] == "cost_guard" for e in errs)


def test_batch_size_one_passes_cost_guard(graph, object_info):
    assert not any(e["check"] == "cost_guard" for e in validate(graph, object_info))


# --- multiple errors collected (no early stop) ---------------------------


def test_all_errors_are_collected_not_just_the_first(graph, object_info):
    graph["1"]["class_type"] = "NotARealComfyNode"
    graph["6"]["inputs"]["batch_size"] = 8
    graph["8"]["inputs"]["sampler_name"] = "bogus"
    errs = validate(graph, object_info)
    assert {"class_type", "cost_guard", "enum"} <= checks(errs)


# --- advanced topologies (the PhD-grade architect's real output) ----------


def test_complex_controlnet_lora_graph_has_no_errors(controlnet_graph, object_info):
    # Multi-output LoRA (model[0] + clip[1]) and dual-output ControlNet apply
    # (positive[0] + negative[1]) must all wire cleanly.
    assert errors(validate(controlnet_graph, object_info)) == []


def test_lora_second_output_reference_resolves(controlnet_graph, object_info):
    # CLIPTextEncode reads clip from LoraLoader output index 1: must be in range.
    assert not any(
        e["check"] == "reference" for e in validate(controlnet_graph, object_info)
    )


def test_reference_beyond_multi_output_node_is_flagged(controlnet_graph, object_info):
    # LoraLoader has 2 outputs (0,1); index 2 is out of range.
    controlnet_graph["11"]["inputs"]["model"] = ["4", 2]
    errs = validate(controlnet_graph, object_info)
    assert any(e["check"] == "reference" for e in errs)


def test_hallucinated_node_name_in_complex_graph_is_caught(
    controlnet_graph, object_info
):
    # The architect's #1 failure mode: inventing a plausible-but-nonexistent node.
    controlnet_graph["9"]["class_type"] = "ControlNetApplyAdvancedPlus"
    errs = validate(controlnet_graph, object_info)
    assert any(e["check"] == "class_type" and e["node"] == "9" for e in errs)


# --- load_graph accepts both API forms -----------------------------------


def test_load_graph_unwraps_prompt_form(tmp_path):
    inner = {"1": {"class_type": "SaveImage", "inputs": {}}}
    p = tmp_path / "wrapped.json"
    p.write_text(json.dumps({"prompt": inner}))
    assert load_graph(str(p)) == inner


def test_load_graph_accepts_bare_form(tmp_path):
    bare = {"1": {"class_type": "SaveImage", "inputs": {}}}
    p = tmp_path / "bare.json"
    p.write_text(json.dumps(bare))
    assert load_graph(str(p)) == bare


# --- CLI end-to-end (autonomous exit-code contract) ----------------------


def test_cli_exits_zero_and_reports_pass_on_valid_graph():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "comfyui_graph_validate",
            str(GRAPH_PATH),
            "--object-info",
            str(OBJECT_INFO_PATH),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    report = json.loads(r.stdout)
    # PASS is defined by zero fatal errors; the control_after_generate warning
    # is surfaced but does not fail the gate.
    assert report["result"] == "PASS"
    assert report["error_count"] == 0
    assert report["warning_count"] >= 1


def test_cli_exits_one_and_reports_fail_on_broken_graph(tmp_path):
    broken = json.loads(GRAPH_PATH.read_text())
    broken["1"]["inputs"]["unet_name"] = "ghost.gguf"
    bad = tmp_path / "broken.json"
    bad.write_text(json.dumps(broken))
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "comfyui_graph_validate",
            str(bad),
            "--object-info",
            str(OBJECT_INFO_PATH),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1
    report = json.loads(r.stdout)
    assert report["result"] == "FAIL"
    assert report["error_count"] >= 1


# --- check 7: preservation -------------------------------------------------
#
# The bug these tests exist for: a product-relight workflow shipped four "product photos"
# in which the product was never in the pixel path at all. It reached
# the graph only as a Canny edge map, every branch sampled from an
# EmptyLatentImage at denoise 1.0, and nothing composited the original pixels
# back. Checks 1-6 pass it happily: every node exists, every reference
# resolves. PRODUCT_LOCALIZER_PATH is that exact graph, kept as the fixture that
# must fail forever.

PRODUCT_LOCALIZER_PATH = FIXTURES / "product_localizer_t2i_api.json"
PRESERVING_GRAPH_PATH = FIXTURES / "preserve_masked_composite_api.json"


@pytest.fixture
def regenerating_graph() -> dict:
    """The real, shipped, broken product-relight graph."""
    return copy.deepcopy(json.loads(PRODUCT_LOCALIZER_PATH.read_text()))


@pytest.fixture
def preserving_graph() -> dict:
    """Same intent, done right: rembg -> mask, VAEEncode -> SetLatentNoiseMask,
    ControlNet held to end_percent 1.0, ImageCompositeMasked restoring pixels."""
    return copy.deepcopy(json.loads(PRESERVING_GRAPH_PATH.read_text()))


def test_the_shipped_broken_graph_fails_preservation(regenerating_graph):
    """The regression that gives this check its reason to exist."""
    findings = preservation_findings(regenerating_graph, "4", "composite")
    assert len(findings) == 1
    assert findings[0]["check"] == "preservation"
    assert findings[0]["severity"] == "error"
    assert "structure hint" in findings[0]["detail"]


def test_broken_graph_fails_in_relight_mode_too(regenerating_graph):
    assert preservation_findings(regenerating_graph, "4", "relight")


def test_free_mode_never_complains(regenerating_graph):
    """Opt-in by construction: a graph that does not declare a subject is not
    judged. Every workflow predating this check keeps working."""
    assert preservation_findings(regenerating_graph, "4", "free") == []


def test_a_correctly_preserving_graph_passes(preserving_graph):
    assert preservation_findings(preserving_graph, "4", "composite") == []
    assert preservation_findings(preserving_graph, "4", "relight") == []


def test_composite_mode_requires_the_pixels_to_come_back(preserving_graph):
    """Pixels enter the latent space but nothing restores them at the end."""
    preserving_graph["13"]["inputs"]["images"] = ["12", 0]
    del preserving_graph["46"]
    findings = preservation_findings(preserving_graph, "4", "composite")
    assert len(findings) == 1
    assert "restores its pixels" in findings[0]["detail"]


def test_relight_mode_rejects_full_denoise_over_an_unmasked_subject(
    preserving_graph,
):
    preserving_graph["11"]["inputs"]["denoise"] = 1.0
    preserving_graph["11"]["inputs"]["latent_image"] = ["43", 0]
    del preserving_graph["44"]
    findings = preservation_findings(preserving_graph, "4", "relight")
    assert len(findings) == 1
    assert "fully renoised" in findings[0]["detail"]


def test_relight_mode_accepts_full_denoise_when_the_mask_protects_the_subject(
    preserving_graph,
):
    """denoise 1.0 is fine when SetLatentNoiseMask confines it to the
    background: that is the whole point of a noise mask."""
    preserving_graph["11"]["inputs"]["denoise"] = 1.0
    assert preservation_findings(preserving_graph, "4", "relight") == []


def test_relight_mode_catches_ksampleradvanced_full_renoise_too(preserving_graph):
    """KSamplerAdvanced has no `denoise` input at all, so reading
    `inputs["denoise"]` on one always finds nothing and the check never
    fires. `add_noise='enable'` at `start_at_step=0` over an unmasked subject
    is the same full renoise `denoise=1.0` catches on plain KSampler: same
    graph as `test_relight_mode_rejects_full_denoise_over_an_unmasked_subject`,
    just the advanced node's own spelling of it."""
    preserving_graph["11"]["class_type"] = "KSamplerAdvanced"
    preserving_graph["11"]["inputs"] = {
        "model": preserving_graph["11"]["inputs"]["model"],
        "positive": preserving_graph["11"]["inputs"]["positive"],
        "negative": preserving_graph["11"]["inputs"]["negative"],
        "latent_image": ["43", 0],
        "add_noise": "enable",
        "noise_seed": 0,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "start_at_step": 0,
        "end_at_step": 10000,
        "return_with_leftover_noise": "disable",
    }
    del preserving_graph["44"]
    findings = preservation_findings(preserving_graph, "4", "relight")
    assert len(findings) == 1
    assert "fully renoised" in findings[0]["detail"]


def test_relight_mode_accepts_ksampleradvanced_partial_renoise(preserving_graph):
    """`add_noise='disable'` (a second-stage refiner picking up an already-
    noised latent) is not a full renoise and must not be flagged."""
    preserving_graph["11"]["class_type"] = "KSamplerAdvanced"
    preserving_graph["11"]["inputs"] = {
        "model": preserving_graph["11"]["inputs"]["model"],
        "positive": preserving_graph["11"]["inputs"]["positive"],
        "negative": preserving_graph["11"]["inputs"]["negative"],
        "latent_image": ["43", 0],
        "add_noise": "disable",
        "noise_seed": 0,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "start_at_step": 10,
        "end_at_step": 10000,
        "return_with_leftover_noise": "disable",
    }
    del preserving_graph["44"]
    assert preservation_findings(preserving_graph, "4", "relight") == []


def test_unknown_subject_node_is_an_error(preserving_graph):
    findings = preservation_findings(preserving_graph, "999", "composite")
    assert len(findings) == 1
    assert "not in the graph" in findings[0]["detail"]


def test_unknown_mode_is_an_error(preserving_graph):
    findings = preservation_findings(preserving_graph, "4", "banana")
    assert len(findings) == 1
    assert "unknown preserve mode" in findings[0]["detail"]


def test_validate_ignores_preservation_unless_asked(object_info, regenerating_graph):
    """Backwards compatibility: without a `preserve` declaration the gate never
    raises a preservation finding, whatever else it says about the graph. (The
    shared object_info fixture is a Flux.2 snapshot and knows nothing about this
    graph's loaders, so checks 1-6 have plenty to say: that is not the point
    here.)"""
    assert "preservation" not in checks(validate(regenerating_graph, object_info))


def test_validate_runs_preservation_when_preserve_is_given(
    object_info, regenerating_graph
):
    findings = validate(
        regenerating_graph,
        object_info,
        preserve={"subject_node": "4", "mode": "composite"},
    )
    assert "preservation" in checks(errors(findings))


def test_cli_preserve_flags_fail_the_broken_graph(tmp_path):
    graph_file = tmp_path / "localizer.json"
    graph_file.write_text(PRODUCT_LOCALIZER_PATH.read_text())
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "comfyui_graph_validate",
            str(graph_file),
            "--object-info",
            str(OBJECT_INFO_PATH),
            "--preserve-subject",
            "4",
            "--preserve-mode",
            "composite",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1
    report = json.loads(r.stdout)
    assert report["result"] == "FAIL"
    assert any(e["check"] == "preservation" for e in report["errors"])


# -- fetching object_info over HTTP ----------------------------------------
#
# Comfy Cloud serves the node catalogue at /api/object_info. The unprefixed
# path answers HTTP 200 with the web app's HTML (measured 2026-08-11), so a
# validator that only tries the bare path does not fail: it validates the
# graph against a web page and reports every node as unknown.


def test_object_info_over_http_falls_back_to_the_api_prefix():
    from comfyui_graph_validate import core as gv

    seen = []

    def fake_get(url, timeout=30):
        seen.append(url)
        if url.endswith("/api/object_info"):
            return json.dumps({"KSampler": {}}).encode()
        return b"<!doctype html><html>the SPA</html>"

    info = gv.fetch_object_info("https://cloud.comfy.org", get=fake_get)
    assert info == {"KSampler": {}}
    assert seen == [
        "https://cloud.comfy.org/object_info",
        "https://cloud.comfy.org/api/object_info",
    ]


def test_object_info_over_http_stops_at_the_first_json_body():
    from comfyui_graph_validate import core as gv

    seen = []

    def fake_get(url, timeout=30):
        seen.append(url)
        return json.dumps({"KSampler": {}}).encode()

    assert gv.fetch_object_info("http://127.0.0.1:8188", get=fake_get)
    assert seen == ["http://127.0.0.1:8188/object_info"]


def test_object_info_over_http_reports_when_neither_path_is_json():
    from comfyui_graph_validate import core as gv

    def fake_get(url, timeout=30):
        return b"<html>nope</html>"

    with pytest.raises(ValueError) as exc:
        gv.fetch_object_info("https://cloud.comfy.org", get=fake_get)
    assert "object_info" in str(exc.value)


def test_object_info_over_http_carries_the_api_key_from_the_environment(monkeypatch):
    """Comfy Cloud answers 401 to an unauthenticated /api/object_info, so a
    validator with no key cannot preflight a cloud graph at all. The key comes
    from the environment, never from a command-line argument."""
    from comfyui_graph_validate import core as gv

    seen = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"KSampler": {}}'

    def fake_urlopen(req, timeout=30):
        seen["headers"] = dict(req.headers)
        return FakeResp()

    monkeypatch.setenv("COMFY_API_KEY", "secret-key")
    monkeypatch.setattr(gv.urlrequest, "urlopen", fake_urlopen)
    assert (
        gv._http_get("https://cloud.comfy.org/api/object_info") == b'{"KSampler": {}}'
    )
    assert {k.lower() for k in seen["headers"]} >= {"x-api-key", "user-agent"}

    seen.clear()
    monkeypatch.delenv("COMFY_API_KEY")
    gv._http_get("http://127.0.0.1:8188/object_info")
    assert "x-api-key" not in {k.lower() for k in seen["headers"]}


# --- V3 dynamic combos -----------------------------------------------------
#
# ComfyUI's newer schema API has an input whose chosen option brings its own
# sub-inputs. `/object_info` answers it with an options list of DICTS, and an
# API-format graph writes the whole thing FLAT: the option key on the input's
# own name, each sub-input as a sibling under a dotted key.
#
# Measured 2026-08-31 against MaskVidExperiments on the local server: reading
# those option dicts as if they were scalars made check 5 refuse `"tracked"`,
# a legal value, on a graph ComfyUI accepted and ran to completion. Fail-closed
# and wrong, on the path a client's preflight takes before every submit.


DYNAMIC_NODE = "MVEx_SubjectCrop"


def _with_dynamic_combo(object_info: dict) -> dict:
    """A node whose `mode` is a V3 dynamic combo, shaped like the real one."""
    object_info[DYNAMIC_NODE] = {
        "input": {
            "required": {
                "mode": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {
                                "key": "combined",
                                "inputs": {
                                    "required": {
                                        "crop_scale": ["FLOAT", {"default": 1.5}]
                                    }
                                },
                            },
                            {
                                "key": "tracked",
                                "inputs": {
                                    "required": {
                                        "crop_scale": ["FLOAT", {"default": 1.5}],
                                        "padding": [
                                            "COMBO",
                                            {
                                                "options": [
                                                    "guaranteed",
                                                    "firm",
                                                    "flexible",
                                                ]
                                            },
                                        ],
                                    }
                                },
                            },
                        ]
                    },
                ],
                "divisible_by": ["INT", {"default": 16}],
            }
        },
        "output": ["IMAGE"],
    }
    return object_info


def _dynamic_graph() -> dict:
    return {
        "1": {
            "class_type": DYNAMIC_NODE,
            "inputs": {
                "mode": "tracked",
                "mode.crop_scale": 1.5,
                "mode.padding": "firm",
                "divisible_by": 16,
            },
        }
    }


def test_a_legal_dynamic_combo_option_is_not_an_invalid_choice(object_info):
    """The regression. `"tracked"` is legal and the server runs it."""
    findings = validate(_dynamic_graph(), _with_dynamic_combo(object_info))
    assert not [f for f in findings if f["severity"] == "error"], findings


def test_an_unknown_dynamic_combo_option_is_still_an_error(object_info):
    """The check has to keep biting, or the fix traded a false positive for a
    fail-open on the one input that decides what the node does."""
    graph = _dynamic_graph()
    graph["1"]["inputs"]["mode"] = "zoomed"  # not among this fixture's options
    errs = [
        f
        for f in validate(graph, _with_dynamic_combo(object_info))
        if f["check"] == "enum"
    ]
    assert errs and errs[0]["node"] == "1", errs


def test_a_declared_dotted_sub_input_is_not_flagged_as_unknown(object_info):
    """`mode.crop_scale` is a real input, spelled the way the API format
    spells it. Warning on it buries the warnings that mean something."""
    findings = validate(_dynamic_graph(), _with_dynamic_combo(object_info))
    assert not [f for f in findings if f["check"] == "input_name"], findings


def test_an_undeclared_dotted_sub_input_is_still_flagged(object_info):
    graph = _dynamic_graph()
    graph["1"]["inputs"]["mode.no_such_knob"] = 1
    findings = validate(graph, _with_dynamic_combo(object_info))
    assert [
        f
        for f in findings
        if f["check"] == "input_name" and "mode.no_such_knob" in f["detail"]
    ], findings


def test_a_dotted_sub_input_value_is_gated_against_its_own_enum(object_info):
    """The gain, not just the silenced noise: before this, the value behind a
    dotted key was never compared to anything at all."""
    graph = _dynamic_graph()
    graph["1"]["inputs"]["mode.padding"] = "aggressive"
    errs = [
        f
        for f in validate(graph, _with_dynamic_combo(object_info))
        if f["check"] == "enum"
    ]
    assert errs and "mode.padding" in errs[0]["detail"], errs
