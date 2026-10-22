from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
image_path = str(PROJECT_ROOT / "assets" / "plan_black_white.jpg")
block_size = 25
drawing_size = 8
acceptance_threshold = 0.40

navigation_block_size = 4
drone_lead_distance = 3

cluster_size = 3
