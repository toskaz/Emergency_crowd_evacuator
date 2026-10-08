from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
image_path = str(BASE_DIR / "assets" / "plan_black_white.jpg")
block_size = 25
drawing_size = 8
acceptance_threshold = 0.40

navigation_block_size = 4
drone_lead_distance = 3

default_exit_capacity = 1.0
exit_capacity_step = 0.5
