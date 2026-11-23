import os
import csv
import datetime
import pygame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
from map.cell import CellType
from grids.cluster import ClusterType


class SimulationMetrics:
    def __init__(self):
        self.step_count = 0
        self.physical_time_s = 0.0
        self.initial_evacuee_count = 0
        self.evacuated_count = 0
        self.walking_steps = 0
        self.waiting_steps = 0
        self.guided_evacuee_ids = set()
        self.exit_counts = {}
        self.floor_clearance_times = {}
        self.floor_initial_counts = {}
        self.last_presence_time = {}
        self.hazards_present = False
        self.history = []
        self.recent_escapes = []
        self._last_positions = {}
        self._consecutive_wait = {}
        self.last_export_message = ""
        self.last_export_timer = 0.0

    def reset(self, simulation=None):
        self.step_count = 0
        self.physical_time_s = 0.0
        self.evacuated_count = 0
        self.walking_steps = 0
        self.waiting_steps = 0
        self.guided_evacuee_ids.clear()
        self.exit_counts.clear()
        self.floor_clearance_times.clear()
        self.floor_initial_counts.clear()
        self.last_presence_time.clear()
        self.hazards_present = False
        self.history.clear()
        self.recent_escapes.clear()
        self._last_positions.clear()
        self._consecutive_wait.clear()
        self.last_export_message = ""
        self.last_export_timer = 0.0

        if simulation and simulation.grid_world:
            for floor_id, floor in simulation.grid_world.floors.items():
                if hasattr(floor, "cluster_grid") and floor.cluster_grid:
                    for cid, cluster in enumerate(floor.cluster_grid.clusters):
                        if getattr(cluster, "type", None) is ClusterType.EXIT:
                            label = simulation.grid_world.get_exit_name(floor_id, cid)
                            self.exit_counts[label] = 0
                f_count = sum(1 for e in simulation.evacuees if e.floor_id == floor_id)
                self.floor_initial_counts[floor_id] = f_count

        self.initial_evacuee_count = len(simulation.evacuees) if simulation else 0

    def on_evacuee_escaped(self, evacuee, grid_world):
        self.evacuated_count += 1
        self.recent_escapes.append(self.physical_time_s)

        floor = grid_world.floors.get(evacuee.floor_id) if grid_world else None
        label = f"Exit (F{evacuee.floor_id + 1})"
        if grid_world and floor and hasattr(floor, "cluster_grid") and floor.cluster_grid:
            cid = floor.cluster_grid.get(evacuee.grid_x, evacuee.grid_y)
            if cid is not None:
                label = grid_world.get_exit_name(evacuee.floor_id, cid)

        self.exit_counts[label] = self.exit_counts.get(label, 0) + 1

    def update(self, simulation):
        if not simulation.started:
            return

        dt = simulation.get_step_duration_s()
        self.physical_time_s += dt
        self.step_count = simulation.step_count

        cutoff = self.physical_time_s - 5.0
        self.recent_escapes = [t for t in self.recent_escapes if t >= cutoff]

        if simulation.grid_world:
            for floor in simulation.grid_world.floors.values():
                if hasattr(floor, "temporary_wall_grid") and floor.temporary_wall_grid:
                    if any(any(row) for row in floor.temporary_wall_grid.cells):
                        self.hazards_present = True
                        break

        floor_counts = {}
        for floor_id in simulation.grid_world.floors.keys():
            floor_counts[floor_id] = 0

        waiting_per_floor = {}
        stuck_per_floor = {}
        for evacuee in simulation.evacuees:
            floor_counts[evacuee.floor_id] = floor_counts.get(evacuee.floor_id, 0) + 1
            if getattr(evacuee, "has_been_guided", False):
                self.guided_evacuee_ids.add(id(evacuee))

            curr_pos = (evacuee.grid_x, evacuee.grid_y, evacuee.floor_id)
            prev_pos = self._last_positions.get(id(evacuee))
            self._last_positions[id(evacuee)] = curr_pos

            in_transit = getattr(evacuee, "in_stair_transit", False)
            if in_transit:
                self.walking_steps += 1
                self._consecutive_wait[id(evacuee)] = 0
            elif prev_pos is not None:
                if prev_pos != curr_pos:
                    self.walking_steps += 1
                    self._consecutive_wait[id(evacuee)] = 0
                else:
                    self.waiting_steps += 1
                    waiting_per_floor[evacuee.floor_id] = waiting_per_floor.get(evacuee.floor_id, 0) + 1
                    wait_count = self._consecutive_wait.get(id(evacuee), 0) + 1
                    self._consecutive_wait[id(evacuee)] = wait_count
                    if wait_count >= 2:
                        stuck_per_floor[evacuee.floor_id] = stuck_per_floor.get(evacuee.floor_id, 0) + 1

        for floor_id, count in floor_counts.items():
            if count > 0:
                self.last_presence_time[floor_id] = self.physical_time_s

        for floor_id in simulation.grid_world.floors.keys():
            if self.floor_initial_counts.get(floor_id, 0) > 0 or self.last_presence_time.get(floor_id, 0) > 0:
                self.floor_clearance_times[floor_id] = self.last_presence_time.get(floor_id, self.physical_time_s)

        flow_rate = self.evacuated_count / self.physical_time_s if self.physical_time_s > 0 else 0.0
        in_stairs = sum(1 for e in simulation.evacuees if getattr(e, "in_stair_transit", False))

        self.history.append({
            "step": self.step_count,
            "time_s": round(self.physical_time_s, 2),
            "remaining": len(simulation.evacuees),
            "evacuated": self.evacuated_count,
            "flow_rate": round(flow_rate, 2),
            "in_stairs": in_stairs,
            "waiting_agents": sum(waiting_per_floor.values())
        })

    def get_live_hud_data(self, simulation):
        remaining = len(simulation.evacuees) if simulation else 0
        total = self.initial_evacuee_count if self.initial_evacuee_count > 0 else (remaining + self.evacuated_count)
        pct = (self.evacuated_count / total) if total > 0 else 0.0

        mins = int(self.physical_time_s // 60)
        secs = int(self.physical_time_s % 60)
        elapsed_str = f"{mins:02d}:{secs:02d}"

        recent_rate = len(self.recent_escapes) / 5.0
        overall_rate = self.evacuated_count / self.physical_time_s if self.physical_time_s > 0 else 0.0
        exit_flow = 0.7 * recent_rate + 0.3 * overall_rate if recent_rate > 0 else overall_rate
        exit_flow_str = f"{exit_flow:.1f} people/s"

        in_stairs = sum(1 for e in simulation.evacuees if getattr(e, "in_stair_transit", False)) if simulation else 0

        bottlenecks_str = "None"
        if simulation:
            stuck_per_floor = {}
            for evacuee in simulation.evacuees:
                if not getattr(evacuee, "in_stair_transit", False):
                    if self._consecutive_wait.get(id(evacuee), 0) >= 2:
                        stuck_per_floor[evacuee.floor_id] = stuck_per_floor.get(evacuee.floor_id, 0) + 1
            if stuck_per_floor and max(stuck_per_floor.values()) >= 3:
                top_floor = max(stuck_per_floor, key=stuck_per_floor.get)
                bottlenecks_str = f"{stuck_per_floor[top_floor]} (Floor {top_floor + 1})"

        return {
            "progress_pct": pct,
            "progress_str": f"{int(round(pct * 100))}%",
            "remaining_str": f"{remaining} / {total}",
            "evacuated_count": self.evacuated_count,
            "total_count": total,
            "sim_time_str": elapsed_str,
            "exit_flow_str": exit_flow_str,
            "on_stairs_str": f"{in_stairs} people" if in_stairs > 0 else "0",
            "bottlenecks_str": bottlenecks_str,
        }

    def get_summary_kpis(self, simulation=None):
        if simulation and simulation.grid_world:
            for floor_id in simulation.grid_world.floors.keys():
                if self.floor_initial_counts.get(floor_id, 0) > 0 or self.last_presence_time.get(floor_id, 0) > 0:
                    self.floor_clearance_times[floor_id] = self.last_presence_time.get(floor_id, self.physical_time_s)

        total = self.initial_evacuee_count if self.initial_evacuee_count > 0 else (self.evacuated_count + (len(simulation.evacuees) if simulation else 0))
        pct = (self.evacuated_count / total * 100) if total > 0 else 0.0

        mins = int(self.physical_time_s // 60)
        secs = int(self.physical_time_s % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        total_steps = self.walking_steps + self.waiting_steps
        if total_steps > 0:
            walking_pct = int(round((self.walking_steps / total_steps) * 100))
            waiting_pct = 100 - walking_pct
        else:
            walking_pct = 100
            waiting_pct = 0

        avg_flow = self.evacuated_count / self.physical_time_s if self.physical_time_s > 0 else 0.0

        drones_count = len(simulation.drones) if simulation else 0
        guided_count = len(self.guided_evacuee_ids)
        if drones_count == 0 and guided_count == 0:
            drone_val = "No guide agents used"
            drone_sub = "guide agents inactive"
        elif self.hazards_present:
            drone_val = f"{guided_count} evacuees guided"
            drone_sub = "assisted & rerouted around hazards"
        else:
            drone_val = f"{guided_count} evacuees assisted"
            drone_sub = "evacuees assisted by guides"

        return {
            "total_time_str": time_str,
            "total_steps": self.step_count,
            "evacuated_str": f"{self.evacuated_count} / {total} ({pct:.0f}%)",
            "avg_flow_str": f"{avg_flow:.1f} people/s",
            "efficiency_str": f"{walking_pct}% Walking / {waiting_pct}% Waiting",
            "efficiency_val": f"{walking_pct}% Walking",
            "efficiency_sub": f"{waiting_pct}% waiting in queues",
            "drone_impact_val": drone_val,
            "drone_impact_sub": drone_sub,
        }

    def generate_charts_surface(self, width, height):
        w_in = max(4.0, width / 100.0)
        h_in = max(2.5, height / 100.0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w_in, h_in), dpi=100)
        fig.patch.set_facecolor("#181818")
        ax1.set_facecolor("#1e1e1e")
        ax2.set_facecolor("#1e1e1e")

        if self.history:
            times = [h["time_s"] for h in self.history]
            rem = [h["remaining"] for h in self.history]
        else:
            times = [0.0]
            rem = [self.initial_evacuee_count]

        ax1.plot(times, rem, color="#3b82f6", linewidth=2.4)
        ax1.fill_between(times, rem, color="#3b82f6", alpha=0.15)
        ax1.set_title("Building Clearance Over Time", color="#f0f0f0", fontsize=11, fontweight="bold", pad=8)
        ax1.set_xlabel("Elapsed Time (s)", color="#c0c0c0", fontsize=9)
        ax1.set_ylabel("Remaining Occupants", color="#c0c0c0", fontsize=9)
        ax1.grid(True, color="#2c2c2c", linestyle="--", linewidth=0.7)
        ax1.tick_params(colors="#b0b0b0", labelsize=9)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_color("#333333")
        ax1.spines["bottom"].set_color("#333333")

        valid_exits = {k: v for k, v in self.exit_counts.items() if v > 0}
        if not valid_exits and self.exit_counts:
            valid_exits = self.exit_counts.copy()
        if not valid_exits:
            valid_exits = {"Exit 1 (F1)": self.evacuated_count}

        labels = list(valid_exits.keys())
        counts = list(valid_exits.values())
        total_e = sum(counts) if sum(counts) > 0 else 1
        pcts = [(c / total_e) * 100.0 for c in counts]

        palette = ["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#6366f1"]
        colors = [palette[i % len(palette)] for i in range(len(labels))]

        bars = ax2.bar(labels, pcts, color=colors, width=0.45, edgecolor="#333333", linewidth=1)
        ax2.set_title("Exit Load Distribution (%)", color="#f0f0f0", fontsize=11, fontweight="bold", pad=8)
        ax2.set_xlabel("Logical Exits", color="#c0c0c0", fontsize=9)
        ax2.set_ylabel("Evacuee Share (%)", color="#c0c0c0", fontsize=9)
        ax2.set_ylim(0, 100)
        ax2.grid(True, axis="y", color="#2c2c2c", linestyle="--", linewidth=0.7)
        ax2.tick_params(colors="#b0b0b0", labelsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color("#333333")
        ax2.spines["bottom"].set_color("#333333")

        for bar, p_val, cnt in zip(bars, pcts, counts):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                p_val + 1.2,
                f"{p_val:.1f}%\n({cnt})",
                ha="center",
                va="bottom",
                color="#ffffff",
                fontsize=8.5,
                fontweight="bold"
            )

        fig.tight_layout()
        fig.canvas.draw()
        rgba = fig.canvas.buffer_rgba()
        surf = pygame.image.frombuffer(rgba, fig.canvas.get_width_height(), "RGBA")
        plt.close(fig)
        return surf

    def generate_auto_insights(self, simulation=None):
        return []

    def export_csv(self, filepath=None):
        if filepath is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ariadne_data_{ts}.csv"
        else:
            p_dir = os.path.dirname(filepath)
            if p_dir:
                os.makedirs(p_dir, exist_ok=True)

        kpis = self.get_summary_kpis()
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ARIADNE EVACUATION SIMULATION REPORT"])
            writer.writerow(["Generated At", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            writer.writerow(["--- SUMMARY KPIS ---"])
            writer.writerow(["Metric", "Value", "Notes"])
            writer.writerow(["Total Evacuation Time", kpis["total_time_str"], f"{kpis['total_steps']} steps"])
            writer.writerow(["Evacuated Occupants", kpis["evacuated_str"], ""])
            writer.writerow(["Average Flow Rate", kpis["avg_flow_str"], "people per second"])
            writer.writerow(["Movement Efficiency", kpis["efficiency_str"], "walking vs waiting ratio"])
            writer.writerow(["Guide Agent Impact", kpis["drone_impact_val"], kpis["drone_impact_sub"]])
            writer.writerow([])
            writer.writerow(["--- EXIT DISTRIBUTION ---"])
            writer.writerow(["Exit", "Evacuee Count"])
            for exit_name, count in self.exit_counts.items():
                writer.writerow([exit_name, count])
            writer.writerow([])
            writer.writerow(["--- STEP-BY-STEP TIME SERIES ---"])
            writer.writerow(["Step", "Physical_Time_s", "Remaining_Inside", "Evacuated", "Flow_Rate_ps", "In_Stairs", "Waiting_Agents"])
            for h in self.history:
                writer.writerow([
                    h["step"],
                    h["time_s"],
                    h["remaining"],
                    h["evacuated"],
                    h["flow_rate"],
                    h["in_stairs"],
                    h["waiting_agents"]
                ])

        abs_path = os.path.abspath(filepath)
        self.last_export_message = f"Saved CSV: {os.path.basename(filepath)}"
        self.last_export_timer = 4.0
        return abs_path

    def export_png(self, surface, filepath=None):
        if filepath is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ariadne_report_{ts}.png"
        else:
            p_dir = os.path.dirname(filepath)
            if p_dir:
                os.makedirs(p_dir, exist_ok=True)

        pygame.image.save(surface, filepath)
        abs_path = os.path.abspath(filepath)
        self.last_export_message = f"Saved PNG: {os.path.basename(filepath)}"
        self.last_export_timer = 4.0
        return abs_path
