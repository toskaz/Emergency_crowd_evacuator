from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
image_path = str(BASE_DIR / "assets" / "plan_black_white.jpg")
block_size = 25
drawing_size = 8
acceptance_threshold = 0.40