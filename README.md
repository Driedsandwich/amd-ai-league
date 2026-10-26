# AMD AI League entries (DriedSandwich)

Source code for my container entries in the lablab.ai **AMD AI League** (individual track).

| Folder | Match | What it does |
|---|---|---|
| `match4/` | Match 4 – Intelligent Web Information Retrieval | Reads a natural-language request, navigates the given site (or a saved copy of it) and returns the one exact value asked for, with its source page. |
| `match3/` | Mini-Challenge 3 – Retrieval-Augmented Generation | Indexes a mixed document corpus once, then answers each question with the value and the minimal set of citing files. |
| `match2/` | Mini-Challenge 2 – Optical Character Recognition | Reads the requested text from an image with a vision-language model kept resident on the GPU. |
| `tools/` | all | Builds the image without Docker: deterministic tar layers appended to the mandated ROCm base image with `crane`, then verified (base-layer identity, size, command, environment, anonymous pull). |

## How the images are built

- Base: the mandated `rocm/pytorch` image (ROCm, PyTorch 2.13, Python 3.14), unchanged and pinned by digest.
- Extra Python packages go to `/opt/x052/site` (`PYTHONPATH`); the base image's torch, torchvision and numpy are used as-is.
- Model weights are shipped inside the images under `/models/` and are **not** in this repository:
  Match 4 uses Qwen3-4B-Instruct-2507; Matches 2 and 3 use Qwen3-VL-8B-Instruct.
- Nothing is downloaded at evaluation time; the model is placed on the GPU explicitly and kept resident behind a unix socket.

## Match 4 note

The detailed Match 4 harness was not published, so `match4/` follows the container contract shared by the earlier matches
(`python3 /app/app.py`, output in `/app/output/<query-id>_output.json` with `answer` and `citations`) and accepts several
argument spellings. See `match4/DESIGN.md`.

## Testing status

- Unit tests run on CPU with stub model backends (`python3 -m unittest discover -s match4/tests`; `pytest` for match2/match3).
- The model-loading and generation code paths were checked on CPU with the base image's own torch/torchvision build.
- Not yet run on an AMD GPU.
- The organizers' sample kits are not redistributed here; tests that need them are skipped or need a local copy.

Code comments are partly in Japanese.

## License

MIT (see `LICENSE`).
