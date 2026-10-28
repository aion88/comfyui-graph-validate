# comfyui-graph-validate

Validateur de graphes ComfyUI au format API, contre le `/object_info` d'un serveur, avant de dépenser du GPU ou du crédit API. Bibliothèque Python sans dépendance, publiée sous `aion88`.

## Statut
- Actif : dernier commit hors « chore: structure Claude Code » le 2026-10-27 (« Initial release »).

## Stack
- Python >= 3.10, aucune dépendance d'exécution (bibliothèque standard). Build hatchling, tests pytest >= 8, `uv`. Licence MIT.
- Version `0.1.0`, script d'entrée `graph-validate` (`comfyui_graph_validate.core:main`).

## Commandes
- `uv run pytest` : tests.
- `uv tool install .` (ou `pip install .`) : installer le CLI.
- `graph-validate graph.json --object-info object_info.json` : valider contre un snapshot.
- `graph-validate graph.json --server http://127.0.0.1:8188` : valider contre un serveur.
- Options documentées : `--json`, `--preserve-subject <id>`, `--preserve-mode composite`.
- Code de sortie : 0 = PASS, 1 = FAIL.

## Structure
- `src/comfyui_graph_validate/core.py` : les 7 contrôles et le CLI (`main`).
- `src/comfyui_graph_validate/__main__.py`, `__init__.py` : point d'entrée et API (`load_graph`, `validate`).
- `tests/test_graph_validate.py` : tests ; `tests/fixtures/` : snapshot `/object_info` réduit et graphes d'exemple.

## Règles du projet
- Les 7 contrôles : `shape`, `class_type`, `reference`, `input_name`, `asset / enum`, `cost_guard` (aucun `batch_size` > 1), `preservation` (opt-in). Les avertissements n'échouent jamais.
- Le contrôle 7 est né d'un cas réel : photos produit où le produit n'entrait jamais dans le chemin des pixels. Ne pas l'affaiblir.
- `COMFY_API_KEY` (Comfy Cloud) se lit dans l'environnement seulement, jamais en argument.
- Le fixture `object_info.json` est un extrait : prendre un snapshot de son propre serveur pour un vrai travail (`curl http://127.0.0.1:8188/object_info`).
- Le validateur ne lance pas le graphe : il ne dit pas si l'image sera bonne.

## Liens
- Dépôt distant : https://github.com/aion88/comfyui-graph-validate.git
- Pas de KB `~/knowledge/comfyui-graph-validate/`.
