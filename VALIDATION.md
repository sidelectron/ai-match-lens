# Validation status

- Twenty unit tests passed for projection, degenerate calibration, camera motion, kit features, automatic clustering, and duplicate suppression.
- A synthetic 125-frame video was generated and decoded successfully.
- A five-second real football clip was processed with YOLOv8 and ByteTrack. Visual spot checks showed separation of blue and red/white kits.
- Regression checks on frame 82 confirmed suppression of nested duplicate boxes while retaining nearby defenders and the goalkeeper.
- The latest real-video output decoded successfully for all 125 frames.
- Local Ollama summaries were tested with a downloaded llama3.2 model.

These are functional tests and spot checks on one match, not quantified detection, tracking or calibration benchmarks. Goalkeeper and referee roles are not explicitly recognized. Similar kits, lighting, occlusion and ID switches can still cause errors.

Pitch mapping became unavailable for the final 28 of 125 frames in the original clip run. Camera cuts and automatic recalibration still need work. Full-match processing with reliable mapping and coaching reports are not implemented.

Test footage and generated match outputs are not distributed in this repository.
