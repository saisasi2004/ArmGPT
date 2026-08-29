# Contributing to ArmGPT

Thanks for taking a look. ArmGPT is a small project with a specific job — turn
plain English into pixel coordinates for a robot arm — and it tries to stay
small. Contributions of any size are welcome.

## Before you start

**If you're changing behaviour, open an issue first.** A short description of
what's wrong or missing saves both of us from a PR that goes the wrong way.
Bug fixes and documentation don't need this.

## Getting set up

```powershell
git clone https://github.com/saisasi2004/ArmGPT.git
cd ArmGPT

python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell (Windows)
# source .venv/bin/activate         # bash/zsh (Linux/macOS)

pip install -r requirements.txt
python app.py
```

Python 3.10–3.12 (mediapipe has no 3.13 wheels yet). You'll also want
[Ollama](https://ollama.com) running with a model pulled — see the README's
Prerequisites.

> **Activate the venv, or run `.venv\Scripts\python.exe app.py` explicitly.**
> The most common "bug" reported against this project is a system Python
> picking up none of the dependencies and the app reporting MongoDB, torch, or
> mediapipe as missing when they're installed a directory away.

You do **not** need a robot to work on ArmGPT. Dry run is the default: commands
are parsed, resolved to coordinates, and logged without a socket ever opening.
For the TCP path, [Hercules](https://www.hw-group.com/software/hercules-setup-utility)
in TCP Client mode stands in for the controller.

## What good looks like here

The codebase has a house style that is worth matching:

- **Comments say *why*, not *what*.** Most of the non-obvious code in this repo
  is non-obvious because of something that was measured or something that
  broke. `services/llm.py` explains why the schema's property order matters;
  `core/camera.py` explains why every cv2 call lives on one thread. If you fix
  something subtle, leave the reason behind.
- **Failures are chat replies, not stack traces.** The router never raises at
  the user; it returns a `status` the UI can style and a sentence an operator
  can act on. New failure modes should follow suit.
- **Refuse rather than guess.** When two objects match, ArmGPT asks. Silently
  picking one is how an arm grabs the wrong thing. Don't add heuristics that
  turn an ambiguity into a motion command.
- **Nothing new in the hot path for free.** The LLM call is CPU-bound and the
  arm blocks on it. Prompt tokens, schema fields, and per-frame work all cost
  latency someone is standing there waiting for.
- Standard library style: 4 spaces, ~79 columns, `from __future__ import
  annotations`, type hints on anything public.

## Testing your change

There is no test suite yet (contributions very welcome). What exists:

```bash
python eval_intent.py            # intent-parser accuracy + latency
python eval_intent.py qwen3:8b   # ...against a specific model tag
```

**Run `eval_intent.py` for anything touching `services/llm.py`** — the prompt
and schema are tuned against those cases, and it is easy to make a change that
reads better and scores worse. Report the before/after numbers in your PR.

Detectors can be run standalone against your webcam, without the web app:

```bash
python -m detectors.color_detector
```

For everything else, launch the app and exercise the path by hand. Say what you
did in the PR — "switched sources twice and rescanned while streaming, feed
survived" is a useful thing for a reviewer to read.

## Safety-related changes

`services/router.py` decides whether a command reaches the arm, and
`_safety_block()` is the hand interlock. Two rules:

1. **Never make the interlock look more trustworthy than it is.** It is a
   convenience check on top of a cell's real safety system. The README, the
   docstring, and the LICENSE all say so, deliberately and in those words.
2. **Never make dry run harder to stay in.** It is the default, it survives
   restarts, and every path that can move an arm honours it.

A PR that removes a refusal path needs to explain what replaces it.

## Pull requests

- Branch off `main`, one logical change per PR.
- Describe what broke or what was missing, not just what you changed.
- Update the README if you add or rename an environment variable — the config
  table is meant to be complete.
- New dependencies need a justification. This project deliberately runs on
  local, open-source pieces with no paid APIs; please keep it that way.

## Reporting bugs

Include:

- what you asked ArmGPT to do, and what it replied;
- the server console output (this is where camera and Ollama problems show up);
- OS, Python version, `ollama list`, and whether Mongo is running;
- for camera issues, rerun with `ARMGPT_CV_VERBOSE=1` to restore OpenCV's raw
  backend logging and include that.

Check the README's **Troubleshooting** section first — the common ones are
already written up there.
