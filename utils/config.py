from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
image_path = str(PROJECT_ROOT / "assets" / "plan_black_white.jpg")
block_size = 25
drawing_size = 8
acceptance_threshold = 0.40

navigation_block_size = 4
drone_lead_distance = 3
drone_lookahead_clusters = 2
# Preferred clearance is a soft cost; one-cell doors remain traversable.
drone_wall_clearance = 3
drone_wall_cost = 2.0
drone_max_pacing_wait = 30
drone_visible_evacuee_target = 6

cluster_size = 3

drone_vision_radius = 8
evacuee_vision_radius = 10
q_table_path = str(PROJECT_ROOT / "models" / "evacuee_q_table.npy")
# Used only by the historical drone RL training scripts.
drone_model_path = str(PROJECT_ROOT / "models" / "drone_path_following.pt")

camera_offset_x = 0
camera_offset_y = 0

door_flow_enabled = True
door_flow_per_cell = 0.4
door_flow_smoothing = 0.1
door_congestion_ticks = 8
door_congestion_streak = 3
door_congestion_min_people = 4
door_reroute_enabled = True
flow_forecast_interval = 10
reroute_min_gain = 3
min_ticks_between_reroutes = 20
