# STSC — Coupled Field Simulator

## Quick start

```bash
git clone https://github.com/jibbalit245/asdfsdfsd
cd asdfsdfsd
./start.sh
```

That's it. `start.sh` will:
1. **Detect your GPU arch** and compile `coupled_field` (needs CUDA / `nvcc` on PATH)
2. **Launch a 500 000-tick run** with default settings:
   | Setting | Default |
   |---------|---------|
   | Seed | `random` |
   | Output | `./frames_longrun/` |
   | Snap every | 500 ticks |
   | Device | GPU 0 |

Frames are written as `frames_longrun/pixel_NNNNNN.png`.  
Logs are streamed to `frames_longrun/run.log`.

## Resume from checkpoint

```bash
RESUME=1 ./start.sh
```

## Override defaults

```bash
SEED=sparse OUT_DIR=./my_frames DEVICE=1 ./start.sh
```

## Other scripts

| Script | Purpose |
|--------|---------|
| `deploy/setup.sh` | Build only |
| `deploy/run.sh` | Quick single run |
| `deploy/run_longrun.sh` | 500k-tick run (used by `start.sh`) |
| `deploy/run_multi.sh` | One run per GPU |
| `deploy/status.sh` | GPU / frame-count health check |

## Make video

After a run, assemble frames into a video:

```bash
ffmpeg -y -framerate 240 -i frames_longrun/pixel_%06d.png \
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
    -vf "scale=1920:1080:flags=lanczos" replay.mp4
```
