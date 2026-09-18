import argparse
import json
from pathlib import Path
import plotly.graph_objects as go

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "super_productivity.json"


def load_data(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def categorize_task(task, all_tasks, projects):
    title = task.get("title", "")
    proj_id = task.get("projectId")
    proj = projects.get(proj_id, "")
    par_id = task.get("parentId")
    parent_title = all_tasks.get(par_id, {}).get("title", "") if par_id else ""

    # Bin 1: Housekeeping & Administration
    if proj == "Housekeeping":
        if parent_title == "Administrative tasks" or title in ["Other", "Meetings"]:
            return ("1. Housekeeping & Administration", "1.1 Admin & Meetings")
        else:
            return ("1. Housekeeping & Administration", "1.2 Planning & Off-Topic")

    # Bin 2: Thesis Writing & Structuring
    if proj.startswith("thesis"):
        if "structure" in title.lower():
            return ("2. Thesis Writing & Structuring", "2.1 Outline & Structure")
        else:
            return ("2. Thesis Writing & Structuring", "2.2 Drafts & Writing")

    # Bin 4: Sutton & Barto: Theory Scanning
    if proj.startswith("literature") and parent_title.startswith("Scan Barto Sutton"):
        if any(ch in title for ch in ["4", "5", "6.1", "7.0", "8.1", "Part I"]):
            return ("4. Sutton & Barto: Theory Scanning", "4.1 Tabular Methods Scanning (Ch 4-8)")
        else:
            return ("4. Sutton & Barto: Theory Scanning", "4.2 Approx & Policy Gradient Scanning (Ch 9-13)")

    # Bin 3: Literature Research & Survey
    if proj.startswith("literature"):
        if title in ["Scan Dreamerv1", "Scan Dreamerv3", "scan hyperparameter optimization", "scan HPO for RL"]:
            return ("3. Literature Research & Survey", "3.2 Dreamer & World Model Papers")
        else:
            return ("3. Literature Research & Survey", "3.1 Topic Exploration & Search")

    # Bin 5: Sutton & Barto: Hands-on Algorithms
    if proj == "Study RL" and parent_title.startswith("Understand BartoSutton"):
        if any(k in title for k in ["5.1", "5.5", "5.8", "CartPole", "6.1", "6.5", "6.6", "6.7", "sarsa", "Q-Learning", "read AI summary"]):
            return ("5. Sutton & Barto: Hands-on Algorithms", "5.1 Monte Carlo & 1-step TD")
        else:
            return ("5. Sutton & Barto: Hands-on Algorithms", "5.2 Multi-step TD & Advanced Planning (Ch 7-13)")

    # Bin 6: Online Courses & Neural Nets
    if proj == "Study RL":
        if any(k in title for k in ["Unit 1", "Unit 2", "Unit 3", "Unit 4"]):
            return ("6. Online Courses & Neural Nets", "6.1 HuggingFace RL Course (Units 1-4)")
        else:
            return ("6. Online Courses & Neural Nets", "6.2 HuggingFace RL Course (Units 5-8 & 3b1b)")

    # Bin 7: Deep Learning Frameworks (PyTorch & TorchRL)
    if proj == "Study tools" and (parent_title in ["study torchrl", "study pytorch", "learnpytorch.io"] or title in ["study numpy", "study python", "study mlagents"]):
        if parent_title in ["study torchrl"] or title in ["study mlagents"]:
            return ("7. Deep Learning Frameworks (PyTorch & TorchRL)", "7.2 TorchRL & Agent Frameworks")
        else:
            return ("7. Deep Learning Frameworks (PyTorch & TorchRL)", "7.1 PyTorch & Python Foundations")

    # Bin 8: Distributed Computing (Ray & RLLib)
    if (proj == "Study tools" and parent_title == "study ray") or (proj.startswith("literature") and parent_title == "Scan" and title in ["ray", "rllib", "ray v2 architecture"]):
        if any(k in title for k in ["core", "jobs", "overview", "Monitoring", "checkpoint", "v2"]):
            return ("8. Distributed Computing (Ray & RLLib)", "8.1 Ray Core, Architecture & Infrastructure")
        else:
            return ("8. Distributed Computing (Ray & RLLib)", "8.2 RLLib & Distributed RL")

    # Bin 9: CyberRunner Gymnasium & Robotics Platform
    if proj.startswith("cr-gymnasium") or proj.startswith("marble-maze-gymnasium"):
        if parent_title in ["Marble Reload", "Cleanup", "Train unchanged"] or title in ["Paint marbles", "Compress git repo", "Migrate to DreamerV3", "study git"]:
            return ("9. CyberRunner & Maze Hardware Environment", "9.1 Hardware Setup, CAD & DVC")
        else:
            return ("9. CyberRunner & Maze Hardware Environment", "9.2 Gym Envs, JAX & Diagnostics")

    # Bin 10: DreamerV3 Deployment & Benchmark Experiments
    if proj.startswith("dreamerv3-deploy") or (proj == "Study tools" and (title in ["study zmq", "study dvc", "study mlflow"] or parent_title in ["study git", "study gymnasium"])):
        if any(k in title for k in ["install", "setup", "clean", "cleanup", "api", "reusable", "dvc", "git"]):
            return ("10. DreamerV3 Deployment & Experiments", "10.1 Container Setup, Environment & Deployment")
        else:
            return ("10. DreamerV3 Deployment & Experiments", "10.2 Benchmark Problems & Evaluation")

    return ("1. Housekeeping & Administration", "1.1 Admin & Meetings")


def generate_treemaps(input_json: Path, output_dir: Path):
    data = load_data(input_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = {p["id"]: p.get("title", "") for p in data.get("project", {}).get("entities", {}).values()}

    all_tasks = {}
    for src in [data.get("task", {}), data.get("archiveYoung", {}).get("task", {}), data.get("archiveOld", {}).get("task", {})]:
        if src and isinstance(src, dict):
            for tid, t in src.get("entities", {}).items():
                all_tasks[tid] = t

    # Flatten leaf tasks (subtasks flattened, parent containers excluded from direct leaves to avoid double counting)
    leaf_tasks = [t for t in all_tasks.values() if not t.get("subTaskIds")]

    # Categorize all tasks
    task_records = []
    for t in leaf_tasks:
        b, sb = categorize_task(t, all_tasks, projects)
        hours = t.get("timeSpent", 0) / 3600000.0
        task_records.append({
            "id": t["id"],
            "title": t.get("title", "Untitled"),
            "bin": b,
            "sub_bin": sb,
            "hours": hours,
            "time_spent_ms": t.get("timeSpent", 0),
        })

    # Color palette for the 10 bins
    bin_colors = {
        "1. Housekeeping & Administration": "#636EFA",
        "2. Thesis Writing & Structuring": "#EF553B",
        "3. Literature Research & Survey": "#00CC96",
        "4. Sutton & Barto: Theory Scanning": "#AB63FA",
        "5. Sutton & Barto: Hands-on Algorithms": "#FFA15A",
        "6. Online Courses & Neural Nets": "#19D3F3",
        "7. Deep Learning Frameworks (PyTorch & TorchRL)": "#FF6692",
        "8. Distributed Computing (Ray & RLLib)": "#B6E880",
        "9. CyberRunner & Maze Hardware Environment": "#FF97FF",
        "10. DreamerV3 Deployment & Experiments": "#FECB52",
    }

    # ==========================================
    # Variant 1: Detailed View (Bins -> Sub-bins -> Tasks)
    # ==========================================
    ids_v1 = ["Total"]
    labels_v1 = ["All Tasks"]
    parents_v1 = [""]
    values_v1 = [0]
    customdata_v1 = [""]
    colors_v1 = ["#FFFFFF"]

    # Calculate subbin and bin totals
    bin_totals = {}
    subbin_totals = {}
    for r in task_records:
        b = r["bin"]
        sb = r["sub_bin"]
        h = r["hours"]
        bin_totals[b] = bin_totals.get(b, 0.0) + h
        subbin_totals[sb] = subbin_totals.get(sb, 0.0) + h

    # Add Bins
    for b in sorted(bin_totals.keys()):
        ids_v1.append(f"bin_{b}")
        labels_v1.append(f"<b>{b}</b><br>{bin_totals[b]:.2f} hrs")
        parents_v1.append("Total")
        values_v1.append(bin_totals[b])
        customdata_v1.append(f"<b>{b}</b><br>Total Hours: {bin_totals[b]:.2f}h")
        colors_v1.append(bin_colors.get(b, "#888888"))

    # Add Sub-bins
    subbin_set = sorted({r["sub_bin"] for r in task_records})
    for sb in subbin_set:
        b_prefix = [b for b in bin_totals.keys() if sb.startswith(b.split(".")[0] + ".")]
        parent_b = b_prefix[0] if b_prefix else "Total"
        ids_v1.append(f"sub_{sb}")
        labels_v1.append(f"<b>{sb}</b><br>{subbin_totals[sb]:.2f} hrs")
        parents_v1.append(f"bin_{parent_b}")
        values_v1.append(subbin_totals[sb])
        customdata_v1.append(f"<b>{sb}</b><br>Sub-bin Hours: {subbin_totals[sb]:.2f}h")
        colors_v1.append(bin_colors.get(parent_b, "#888888"))

    # Set root total value
    values_v1[0] = sum(bin_totals.values())
    customdata_v1[0] = f"Total Tracked: {values_v1[0]:.2f} hrs"

    # Add Tasks
    for r in task_records:
        tid = r["id"]
        sb = r["sub_bin"]
        b = r["bin"]
        h = r["hours"]
        ids_v1.append(f"task_{tid}")
        labels_v1.append(r["title"])
        parents_v1.append(f"sub_{sb}")
        values_v1.append(h)
        customdata_v1.append(f"<b>{r['title']}</b><br>Duration: {h:.2f} hrs ({h*60:.1f} mins)<br>Bin: {b}<br>Sub-bin: {sb}")
        colors_v1.append(bin_colors.get(b, "#888888"))

    # Use branchvalues='total' so area strictly reflects total hours at every level
    fig_v1 = go.Figure(go.Treemap(
        ids=ids_v1,
        labels=labels_v1,
        parents=parents_v1,
        values=values_v1,
        branchvalues="total",
        marker=dict(colors=colors_v1),
        textinfo="label+value",
        hovertemplate="%{customdata}<extra></extra>",
        maxdepth=3,
        pathbar=dict(visible=True),
    ))

    fig_v1.update_layout(
        title=dict(
            text="<b>SuperProductivity Tasks Breakdown by Cohesive Bins & Sub-bins (Detailed)</b>",
            font=dict(size=18),
        ),
        margin=dict(t=50, l=15, r=15, b=15),
        height=900,
    )

    out_file_v1 = output_dir / "super_productivity_treemap_tasks.html"
    fig_v1.write_html(str(out_file_v1), include_plotlyjs="cdn")
    print(f"Saved: {out_file_v1}")

    # ==========================================
    # Variant 2: Bins & Sub-bins Only View
    # ==========================================
    ids_v2 = ["Total"]
    labels_v2 = ["Total Portfolio"]
    parents_v2 = [""]
    values_v2 = [sum(bin_totals.values())]
    customdata_v2 = [f"Total Tracked: {sum(bin_totals.values()):.2f} hrs"]
    colors_v2 = ["#FFFFFF"]

    for b in sorted(bin_totals.keys()):
        ids_v2.append(f"bin_{b}")
        labels_v2.append(f"<b>{b}</b><br>{bin_totals[b]:.2f} hrs")
        parents_v2.append("Total")
        values_v2.append(bin_totals[b])
        customdata_v2.append(f"<b>{b}</b><br>Total Hours: {bin_totals[b]:.2f} hrs ({bin_totals[b]/sum(bin_totals.values())*100:.1f}%)")
        colors_v2.append(bin_colors.get(b, "#888888"))

    for sb in subbin_set:
        b_prefix = [b for b in bin_totals.keys() if sb.startswith(b.split(".")[0] + ".")]
        parent_b = b_prefix[0] if b_prefix else "Total"
        ids_v2.append(f"sub_{sb}")
        labels_v2.append(f"<b>{sb}</b><br>{subbin_totals[sb]:.2f} hrs")
        parents_v2.append(f"bin_{parent_b}")
        values_v2.append(subbin_totals[sb])
        customdata_v2.append(f"<b>{sb}</b><br>Hours: {subbin_totals[sb]:.2f} hrs ({subbin_totals[sb]/bin_totals[parent_b]*100:.1f}% of bin)")
        colors_v2.append(bin_colors.get(parent_b, "#888888"))

    fig_v2 = go.Figure(go.Treemap(
        ids=ids_v2,
        labels=labels_v2,
        parents=parents_v2,
        values=values_v2,
        branchvalues="total",
        marker=dict(colors=colors_v2),
        textinfo="label+percent parent+percent root",
        hovertemplate="%{customdata}<extra></extra>",
        maxdepth=2,
        pathbar=dict(visible=True),
    ))

    fig_v2.update_layout(
        title=dict(
            text="<b>SuperProductivity Effort Distribution by Bins & Sub-bins (High-Level Summary)</b>",
            font=dict(size=18),
        ),
        margin=dict(t=50, l=15, r=15, b=15),
        height=750,
    )

    out_file_v2 = output_dir / "super_productivity_treemap_bins.html"
    fig_v2.write_html(str(out_file_v2), include_plotlyjs="cdn")
    print(f"Saved: {out_file_v2}")


def main():
    parser = argparse.ArgumentParser(description="Generate Plotly treemaps from SuperProductivity data")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to super_productivity.json (defaults to packaged data)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for generated HTML files (defaults to current working directory)",
    )
    args = parser.parse_args()

    generate_treemaps(input_json=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
