# MatchLens AI

A local Python MVP that processes a football clip into an annotated tactical video and per-frame pitch coordinates. Optional Ollama summaries run on your own machine. No API keys or paid API services are used.

## Start

Use Python 3.12 and run these commands from this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m matchlens demo
```

The demo generates five seconds of **synthetic positions**, demonstrating the renderer only. It is not evidence of football detection accuracy. Open `runs/demo/demo.mp4`.

## Process a real clip

Start with a short, continuous wide-angle shot. Calibrate using at least four known, visible pitch landmarks spread across both axes. Supply their real pitch coordinates in meters, then click those exact landmarks in the first frame in the same order. Origin is one pitch corner; x runs along its length and y across its width. Dimensions default to 105 × 68 m; use the actual dimensions where known.

For example, if all four field corners are visible, use:

```powershell
.venv\Scripts\python -m matchlens calibrate match.mp4 --points '[[0,0],[105,0],[105,68],[0,68]]'
```

Most broadcast frames do **not** show all corners. In that case, specify coordinates of visible penalty-box corners, halfway-line intersections, or other known markings instead. Do not click guessed off-screen corners. At least four noncollinear correspondences are required. Press U to undo, Enter to save, Escape to cancel. Calibration is tied to the first frame and original video resolution.

```powershell
.venv\Scripts\python -m matchlens run match.mp4 --calibration calibration.json --output runs/match01 --max-frames 300
```

Omit `--max-frames` to process the whole clip. Default is CPU; use `--device 0` with a suitable CUDA PyTorch installation. The first run downloads `yolov8n.pt` if absent. You can pass a local model file with `--weights`. The model must expose a `person` or `player` class.

Results: `tactical.mp4` (silent annotated video), `tracks.csv` (track IDs, team clusters, confidence, foot points and pitch coordinates), and `summary.json` (measurement coverage and limitations). Existing runs are protected from overwrite.

## Ollama

Ollama is optional. Install Ollama and download a local model, for example with `ollama pull llama3.2`. Start Ollama, then:

```powershell
.venv\Scripts\python -m matchlens explain runs/match01/summary.json --model llama3.2:latest --output runs/match01/analysis.md
```

Only the numeric summary and its limitations are sent to `127.0.0.1:11434`. Cloud model metadata is rejected. Use downloaded local models; no model is automatically pulled by this command. The LLM describes measurement coverage, not unmeasured match events or tactics. Its output remains a generated interpretation.

## What is implemented

- YOLOv8 detection with ByteTrack IDs and duplicate-observation suppression.
- Torso color distributions preserve stripes and include green, white, gray and black. Local side strips downweight background-like pixels at crop edges, without excluding any hue. Two kit clusters require multiple tracked players and adapt slowly from confident observations. Ambiguous appearances remain unknown; Team A/B are learned automatically without supplied colors. This uses crop heuristics, not learned player segmentation. Similar kits, occlusion, tracking ID switches and unusual lighting can still confuse assignment. Goalkeeper/referee roles are not explicitly recognized.
- Normalized DLT homography and bottom-center bounding-box projection.
- Sparse optical flow plus RANSAC to propagate calibration between frames.
- Side-by-side broadcast annotation and tactical pitch, with visible-player team spread.
- CSV export and optional local Ollama summary.

Camera propagation can drift and is not automatic pitch recognition. On feature failure, mapping stops for the rest of the clip rather than continuing with stale calibration. Split at camera cuts and recalibrate each shot. Cut detection is incomplete; inspect output before treating coordinates as measurements. Team clustering can confuse referees, goalkeepers, similar kits, and lighting changes. Generic YOLO weights are a starting point and may detect spectators. IDs are tracker IDs, not jersey numbers or stable player identities across cuts.

Ball tracking, possession, passing options, speed and accumulated distance are not implemented. A planar mapping also assumes foot points lie on the ground. No real-time throughput or real-footage accuracy claim has been established.

## Checks

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Twenty tests cover projection, calibration failures, kit appearance, automatic clustering, camera motion, and duplicate suppression. A five-second real broadcast clip was also processed and spot-checked; this is not a general accuracy benchmark. See [validation notes](VALIDATION.md).

API references: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/), [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md). Review dependency and model licenses before distributing the project.

## Roadmap

Automatic shot detection and pitch recalibration for long videos, ball tracking, and timestamped coaching reports remain future work. The current Ollama command summarizes measurement quality only.

## Repository contents

Source code and tests are included. Supply your own footage; match videos, trained weights, local calibration, environments and generated outputs are excluded.
