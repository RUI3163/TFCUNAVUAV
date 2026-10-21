# GPS-denied VLN UAV on Project AirSim — implementation guide

Searches a 40×40 m parking area at 10 m for the red car `Car_65` and reports its position relative to Reference Point R,
using only a nadir camera, IMU and altitude (no GPS).

## 0. Read this first: what is and isn't verified

| Verified here | NOT verified here (needs your machine) |
|---|---|
| Every `projectairsim` call was checked against the real client source (`github.com/iamaisim/ProjectAirSim`, `client/python`): `ProjectAirSimClient`, `World(..., sim_config_path=)`, `Drone`, `drone.sensors[cam]["scene_camera"]`, `robot_info["actual_pose"]`, `sensors[imu]["imu_kinematics"]`, `unpack_image`, `takeoff_async`, `move_by_velocity_z_async`, `move_to_position_async`, `hover_async`, `land_async`, `world.get_object_pose` | Full end-to-end flight (no Unreal/GPU/model weights in my sandbox) |
| 11 unit tests pass (`pytest -q`): camera intrinsics, pinhole axes/yaw, lawn-mower lanes, A* detours, E1/E2 logic, optical-flow dead reckoning on a synthetic shift, N=3 consensus gate, spec validation | Qwen2.5 / Qwen2-VL loading under *your* `transformers` version; YOLO11n on top-down sim cars; IMU gyro sign |

`vln/sim_client.py` follows your working `airsim_driver.py` flow (connect → `World(scene)` → `Drone` → `enable_api_control` → `arm` → `*_async` + `await task`) and adopts three lessons from it: the scene folder is auto-resolved and passed as `World(sim_config_path=<folder that directly contains the scene>)` exactly as in your working `find_object_coords.py`, landing is staged (`land_async()` alone descends ~0.19 m/s), and `unpack_image` frames are BGR. `move_to_position_async` yaws toward the travel direction, so this repo never uses it: all flight is closed-loop `move_by_velocity_async` (the pattern in your `fly_above_object.py`).

## 1. Repository layout

```
vln_uav/
├── run_mission.py              # entry point (Stage 1 prompt → Stage 8 report)
├── requirements.txt
├── sim_config/downcamera_sensor_snippet.jsonc   # nadir cam + IMU block for your robot config
├── vln/
│   ├── config.py               # ALL tunables (camera, thresholds, N=3, models, geometry)
│   ├── geometry.py             # pinhole projection, yaw, range/bearing (pure)
│   ├── sim_client.py           # Project AirSim wrapper: frames, pose, IMU, flight commands
│   ├── pose_estimator.py       # Stage 4: optical-flow + IMU yaw + altitude dead reckoning
│   ├── stage2_mission.py       # Qwen2.5-3B-Instruct → Pydantic MissionSpec (run once, unloaded)
│   ├── stage3_planner.py       # boustrophedon + A* detours (pure)
│   ├── stage4_navigation.py    # waypoint tracking on the dead-reckoned estimate
│   ├── stage5_perception.py    # YOLO11n + MobileNetV3-Small, E1/E2 events, latency tracking
│   ├── stage6_verifier.py      # Qwen2-VL-2B async worker (bounded queue, P(Yes) verdict)
│   ├── stage7_localizer.py     # nadir pinhole fix + N=3 consensus gate + uncertainty
│   ├── stage8_report.py        # landmark-relative JSON report
│   ├── pipeline.py             # orchestrator
│   └── gpu.py                  # VRAM helpers (4-bit config, cache freeing)
├── tools/                      # collect_frames → build_classifier_dataset → train_classifier
└── tests/test_core.py
```

## 2. Step-by-step

**Step 1 — Environment (Python 3.11).**
Install CUDA PyTorch first (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`), then `pip install -r requirements.txt`, then the client: `git clone https://github.com/iamaisim/ProjectAirSim && pip install -e ProjectAirSim/client/python/projectairsim`. Run `pytest -q` — must be 11 passed before touching the simulator.

**Step 2 — Scene/robot config.**
Put your scene `.jsonc` (containing `Car_65`) and robot config in `sim_config/`, set `SimCfg.scene_config` in `vln/config.py`. Merge `sim_config/downcamera_sensor_snippet.jsonc` into the robot's `sensors` (400×225, FOV 90°, `rpy-deg "0 -90 0"`, `capture-enabled: true`). Set `GeometryCfg.r_world_ne` (where R is in world NED) and `start_offset_ne` (UAV start relative to R). Both default to (0,0), i.e. the UAV spawns over R at world origin — **check this against your scene**. The ground under the flight area needs visible texture, or optical flow has nothing to track.

**Step 3 — Smoke-test the connection + flight, no ML.**
`python -m tools.collect_frames --out data/raw` flies the lawn-mower in `groundtruth` mode and saves frames. If this works, the client wiring, camera, pose and planner are good. Check the saved frames are nadir, north-up, 400×225.

**Step 4 — Train the Tier-1 classifier** (no weights are shipped).
`python -m tools.build_classifier_dataset` → **manually review/correct the label folders** (`red_car`, `other_car`, `not_car`; the script only proposes labels via box + HSV redness) → `python -m tools.train_classifier --epochs 20`. Aim for a few hundred crops per class, incl. other-coloured cars and parking-lot clutter as `not_car`. Colour jitter is deliberately off (hue is the signal).

**Step 5 — Check the detector.**
Stock COCO YOLO11n is trained on side/oblique views; top-down sim cars may be missed. Look at recall on your frames. If weak, fine-tune with Ultralytics on your frames (`yolo detect train model=yolo11n.pt data=cars.yaml imgsz=416`), then set `PerceptionCfg.detector_weights` and `det_classes`. The detector is intentionally low-threshold (conf 0.15); the classifier does the filtering.

**Step 6 — Run.**
`python run_mission.py` (or `--nav-mode groundtruth` to bypass dead-reckoned navigation while debugging, `--spec-json spec.json` to skip Stage 2 once you have a valid spec). Output: `results/mission_report_*.json`, logs in `logs/`.

**Step 7 — Tune from the report.** `stats.tier1_latency` (must stay ≤100 ms p95), `tier1_counts` (E1/E2/submitted), `vlm_calls`, and `evaluation_ground_truth.position_error_m` (Car_65's true position, used *only* for scoring).

## 3. How each stage maps to code

| Stage | Where | Key behaviour |
|---|---|---|
| 1 | `config.MISSION_PROMPT` | fixed prompt |
| 2 | `stage2_mission.parse_mission` | Qwen2.5-3B 4-bit, greedy, ≤3 retries with error feedback, cross-checks colour/reference against the prompt, **raises** on failure (no silent default), unloads before anything else loads |
| 3 | `stage3_planner.plan_search` | lanes from camera footprint (20 m swath at 10 m ⇒ 3 lanes at E = −10, 0, 10, 20 % overlap); A* on 1 m grid with inflated no-fly boxes + line-of-sight pruning |
| 4 | `pose_estimator`, `stage4_navigation` | LK optical flow (median) × altitude/fx + yaw → (N,E); P-controller on the *estimate*; yaw rate 0 holds heading so the image stays north-up |
| 5 | `stage5_perception` | every frame: flow update → YOLO → crops → MobileNetV3 → E1 (p_red ≥ 0.60) / E2 (0.25 ≤ p_red < 0.60 or max p < 0.5); latest-frame-only buffer, per-frame latency logged |
| 6 | `stage6_verifier` | own thread, bounded queue (full ⇒ candidate dropped, never blocks flight), one forward pass, P(“Yes”) vs P(“No”) ≥ 0.5; prompt colour comes from the Stage 2 spec |
| 7 | `stage7_localizer` | pixel → ground offset at capture-time pose snapshot; N=3 **consecutive** positive verdicts within 3 m ⇒ fused fix |
| 8 | `stage8_report` | E/N offset, range, bearing (° clockwise from north), 1σ/2σ uncertainty |

## 4. Design decisions you should know about

- **4 GB VRAM.** Qwen2-VL-2B in fp16 is ≈4.4 GB and does not fit; both Qwen models load in 4-bit NF4 (`bitsandbytes`). Stage 2's model is freed before Tier-1/Tier-2 load. **Unreal Engine also uses this GPU.** Run it at low quality / small resolution (e.g. `-ResX=640 -ResY=360`, or off-screen) and watch the `[VRAM]` log lines, which report device-wide use. If it doesn't fit, lower `VerifierCfg`'s image size (`max_pixels` in `stage6_verifier.py`) before considering CPU offload (slow).
- **“Consecutive” (N=3)** is counted over the VLM's result stream. Any negative verdict resets the streak; a positive more than 3 m from the running streak restarts it. Conservative (favours precision); relax in `ConsensusGate` if recall suffers.
- **Uncertainty model**: drift (3 % of distance flown, common-mode, does *not* average down) ⊕ pixel/altitude/yaw terms (averaged over N) ⊕ empirical scatter. The 3 % is a placeholder — **calibrate it** from your own runs using `position_error_m`.
- **What is “GPS-denied” here.** Horizontal position comes only from optical flow + yaw; the sim's x/y are ignored. Altitude and the initial yaw alignment are read from `actual_pose` as stand-ins for a rangefinder/baro and a compass (swap in `DistanceSensor`/`Barometer` topics if your robot has them — `SimInterface._on_pose` is the single place). Ground-truth `Car_65` is read only after landing, for scoring.
- **Dead-reckoning limits**: no roll/pitch de-rotation (fine for slow, level legs), needs textured ground, drift grows with distance. If the estimate misbehaves, use `--nav-mode groundtruth` to separate navigation problems from perception problems.
- **Gyro sign.** `NavCfg.gyro_yaw_sign` — if yaw drifts the wrong way during turns (rare, as heading is held), flip it or set `yaw_source="pose"`.
- **Stage 5 classes** are red-specific. If Stage 2 yields another colour the pipeline warns; retrain the classifier for that colour.

## 5. Facts taken from your other scripts (`vlm_baseline_eval_v4_flight.py`, `vlm_benchmark.py`, `vlm_understanding.py`)

- **Car_65 is not near world (0,0).** Your measured pose: NED north −118.68, east −148.21, down −1.03, yaw 87.55°. "R (0,0)" is therefore a *local* frame: set `GeometryCfg.r_world_ne` to R's world position (near the car), and `start_offset_ne` to the UAV start relative to R. Your baseline flew ~200 m legs, so the spawn is far from the car. The planner then adds a straight, **unchecked** transit leg, and dead-reckoning drift (~3 %) accumulates over it (≈6 m over 200 m). Strong recommendation: move the spawn to R (or inside the area) so the start is a known position.
- **Altitude reference.** Pinhole scale needs height above the *local* ground. Set `GeometryCfg.ground_z_world = -1.03` (car base) instead of the spawn z.
- **The scene has trees and houses** (your script uses a safe transit altitude to avoid them). 10 m may not clear them; add measured boxes to `GeometryCfg.no_fly_boxes`, or verify clearance manually before the first flight.
- **A second red car is parked next to Car_65** (your `DISTRACTORS` comment). The pipeline reports the first red car it confirms, which may be the wrong one; decide the intended behaviour (e.g. sweep everything and report all, or filter by an extra cue) and fill in the distractor's pose for scoring.
- **Colour channel order.** `unpack_image` frames are BGR (your driver flips them for PIL), but `vlm_benchmark.py` carries a `--swap-rgb` escape hatch. Colour is the whole task, so save one frame of a known red object and check it. `CameraCfg.swap_rgb` flips it if needed.
- **First frame after a camera/yaw change can be stale** (your script discards one). Irrelevant for the continuous stream, but if you use `tools.collect_frames` after changing altitude, drop the first frames.
- **VLM history.** llava:7b with schema-constrained JSON misdescribed scenes; gemma3:4b was your later default. Stage 6 avoids JSON/keyword parsing (a single Yes/No logit), but nothing says Qwen2-VL-2B is accurate on top-down sim cars: run `python -m tools.eval_verifier` on your labelled crops first. `rotate_to_yaw_async` takes **radians** (your driver passes degrees; this repo never rotates).

## 6. Facts taken from `find_object_coords.py` / `fly_above_object.py`

- **Map = AirSim Neighborhood (`AirSimNH.exe`)**, scene `scene_basic_drone.jsonc`: houses, trees, many `Car_*` actors. `World(..., sim_config_path=<folder that directly contains the scene>)` works for you (the `...\example_user_scripts\sim_config` folder); `sim_client.resolve_sim_config_path` reuses your resolver. Your earlier driver's "doesn't work" was most likely a one-level-off folder.
- **Verified in your scene:** `drone.sensors["DownCamera"]["scene_camera"]` subscription streams frames (your `fly_above_object.py` displays it), so the streaming path in `SimInterface` is now confirmed (the polling fallback stays as insurance). The `Chase` sensor and `DownCamera` `depth_camera` topics also exist.
- **Flight primitive.** Your own scripts avoid `move_to_position_async` and use closed-loop `move_by_velocity_async` with `v_down = clip(0.7·err, ±3)`; this repo now does the same (`SimInterface.velocity_z`; `move_by_velocity_z_async` is not used because it is unproven on your build). `--nav-mode groundtruth` is exactly your `go_to` loop on sim pose.
- **Finding R, houses and other cars:** `python find_object_coords.py --regex ".*[Hh]ouse.*" ".*Car.*" --mark` gives world poses (use them for `r_world_ne`, `no_fly_boxes`, and to locate the second red car). Poses only give pivots, so you still need to pick box extents. After each mission, `SimInterface.mark()` draws **R** (blue) and the **estimated target** (red) in Unreal via `plot_debug_strings/points`, so you can eyeball the error against the car.
- `get_ground_truth_pose()` returns `translation`/`rotation`; the `actual_pose` topic used here returns `position`/`orientation`. Both are used in your scripts/examples; if the topic's keys differ on your build, `SimInterface._on_pose` is the one place to change.
