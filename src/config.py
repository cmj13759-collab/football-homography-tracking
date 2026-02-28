from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
EXTERNAL_DIR = PROJECT_ROOT / "external"

MAPS_DIR = DATA_DIR / "2d_player_maps"
DEBUG_DIR = DATA_DIR / "debug_coordinates"
INV_DIR = DATA_DIR / "inverse_homography"

for d in [DATA_DIR, VIDEO_DIR, EXTERNAL_DIR, MAPS_DIR, DEBUG_DIR, INV_DIR]:
    d.mkdir(parents=True, exist_ok=True)