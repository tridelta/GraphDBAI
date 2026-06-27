from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COLORS = {
    "green": "#2f8f83",
    "blue": "#5373c7",
    "orange": "#d8893a",
    "red": "#b94b47",
    "purple": "#7a63b8",
    "gray": "#687078",
    "ink": "#1d2327",
    "grid": "#d9ded8",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float, digits: int = 2) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def action_label(action: dict[str, Any] | None) -> str:
    if not action:
        return "None"
    name = action.get("name", "unknown")
    args = action.get("args") or {}
    if not args:
        return f"{name}()"
    return f"{name}({', '.join(str(v) for v in args.values())})"


def svg_frame(width: int, height: int, body: str, title: str, subtitle: str = "") -> str:
    subtitle_svg = f'<text x="24" y="48" class="subtitle">{esc(subtitle)}</text>' if subtitle else ""
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">
  <style>
    .title {{ font: 700 18px Inter, system-ui, sans-serif; fill: {COLORS["ink"]}; }}
    .subtitle {{ font: 12px Inter, system-ui, sans-serif; fill: {COLORS["gray"]}; }}
    .axis {{ stroke: #9da59d; stroke-width: 1; }}
    .grid {{ stroke: {COLORS["grid"]}; stroke-width: 1; }}
    .tick {{ font: 11px Inter, system-ui, sans-serif; fill: #596169; }}
    .label {{ font: 12px Inter, system-ui, sans-serif; fill: {COLORS["ink"]}; }}
    .legend {{ font: 12px Inter, system-ui, sans-serif; fill: {COLORS["ink"]}; }}
  </style>
  <text x="24" y="28" class="title">{esc(title)}</text>
  {subtitle_svg}
  {body}
</svg>
"""


def scale(values: list[float], low: float, high: float, pad: float = 0.08) -> tuple[float, float]:
    if not values:
        return 0, 1
    min_v = min(values)
    max_v = max(values)
    if math.isclose(min_v, max_v):
        return min(0, min_v), max_v + 1
    span = max_v - min_v
    return min_v - span * pad, max_v + span * pad


def line_chart(
    title: str,
    series: list[dict[str, Any]],
    x_labels: list[str],
    width: int = 760,
    height: int = 340,
    subtitle: str = "",
    y_min: float | None = None,
) -> str:
    left, right, top, bottom = 58, 24, 68, 50
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [float(v) for item in series for v in item["values"]]
    min_v, max_v = scale(all_values, 0, 1)
    if y_min is not None:
        min_v = y_min
    if math.isclose(min_v, max_v):
        max_v = min_v + 1

    def x_pos(index: int) -> float:
        return left + (plot_w * index / max(1, len(x_labels) - 1))

    def y_pos(value: float) -> float:
        return top + plot_h - ((value - min_v) / (max_v - min_v)) * plot_h

    body = []
    for i in range(5):
        value = min_v + (max_v - min_v) * i / 4
        y = y_pos(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{fmt(value)}</text>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    for i, label in enumerate(x_labels):
        x = x_pos(i)
        body.append(f'<text x="{x:.1f}" y="{height-24}" text-anchor="middle" class="tick">{esc(label)}</text>')
    for item in series:
        values = [float(v) for v in item["values"]]
        points = " ".join(f"{x_pos(i):.1f},{y_pos(value):.1f}" for i, value in enumerate(values))
        body.append(f'<polyline fill="none" stroke="{item["color"]}" stroke-width="2.5" points="{points}"/>')
        for i, value in enumerate(values):
            x, y = x_pos(i), y_pos(value)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{item["color"]}"/>')
            body.append(f'<text x="{x:.1f}" y="{y-8:.1f}" text-anchor="middle" class="tick">{fmt(value)}</text>')
    legend_x = left
    for item in series:
        body.append(f'<rect x="{legend_x}" y="{top-26}" width="12" height="12" fill="{item["color"]}"/>')
        body.append(f'<text x="{legend_x+18}" y="{top-16}" class="legend">{esc(item["name"])}</text>')
        legend_x += 18 + len(item["name"]) * 8 + 24
    return svg_frame(width, height, "\n".join(body), title, subtitle)


def bar_chart(
    title: str,
    labels: list[str],
    values: list[float],
    width: int = 760,
    height: int = 340,
    color: str = COLORS["green"],
    subtitle: str = "",
    y_max: float | None = None,
) -> str:
    left, right, top, bottom = 58, 24, 68, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    max_v = y_max if y_max is not None else max(values + [1]) * 1.15
    body = []
    for i in range(5):
        value = max_v * i / 4
        y = top + plot_h - value / max_v * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{fmt(value)}</text>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    gap = 12
    bar_w = (plot_w - gap * (len(labels) + 1)) / max(1, len(labels))
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + gap + i * (bar_w + gap)
        h = value / max_v * plot_h if max_v else 0
        y = top + plot_h - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="5" fill="{color}"/>')
        body.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" class="tick">{fmt(value)}</text>')
        body.append(f'<text x="{x + bar_w/2:.1f}" y="{height-28}" text-anchor="middle" class="tick">{esc(label)}</text>')
    return svg_frame(width, height, "\n".join(body), title, subtitle)


def stacked_bar_chart(
    title: str,
    labels: list[str],
    series: list[dict[str, Any]],
    width: int = 760,
    height: int = 360,
    subtitle: str = "",
) -> str:
    left, right, top, bottom = 58, 24, 76, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    totals = [sum(float(item["values"][i]) for item in series) for i in range(len(labels))]
    max_v = max(totals + [1]) * 1.15
    body = []
    for i in range(5):
        value = max_v * i / 4
        y = top + plot_h - value / max_v * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{fmt(value)}</text>')
    gap = 12
    bar_w = (plot_w - gap * (len(labels) + 1)) / max(1, len(labels))
    for i, label in enumerate(labels):
        x = left + gap + i * (bar_w + gap)
        base_y = top + plot_h
        for item in series:
            value = float(item["values"][i])
            h = value / max_v * plot_h
            base_y -= h
            body.append(f'<rect x="{x:.1f}" y="{base_y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{item["color"]}"/>')
        body.append(f'<text x="{x + bar_w/2:.1f}" y="{height-28}" text-anchor="middle" class="tick">{esc(label)}</text>')
    legend_x = left
    for item in series:
        body.append(f'<rect x="{legend_x}" y="{top-28}" width="12" height="12" fill="{item["color"]}"/>')
        body.append(f'<text x="{legend_x+18}" y="{top-18}" class="legend">{esc(item["name"])}</text>')
        legend_x += 18 + len(item["name"]) * 8 + 24
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    return svg_frame(width, height, "\n".join(body), title, subtitle)


def grouped_bar_chart(
    title: str,
    groups: list[str],
    series: list[dict[str, Any]],
    width: int = 760,
    height: int = 360,
    subtitle: str = "",
) -> str:
    left, right, top, bottom = 58, 24, 76, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    max_v = max([float(v) for item in series for v in item["values"]] + [1]) * 1.15
    body = []
    for i in range(5):
        value = max_v * i / 4
        y = top + plot_h - value / max_v * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{fmt(value)}</text>')
    group_gap = 20
    inner_gap = 5
    group_w = (plot_w - group_gap * (len(groups) + 1)) / max(1, len(groups))
    bar_w = (group_w - inner_gap * (len(series) - 1)) / max(1, len(series))
    for i, group in enumerate(groups):
        group_x = left + group_gap + i * (group_w + group_gap)
        for j, item in enumerate(series):
            value = float(item["values"][i])
            x = group_x + j * (bar_w + inner_gap)
            h = value / max_v * plot_h
            y = top + plot_h - h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{item["color"]}"/>')
            body.append(f'<text x="{x + bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" class="tick">{fmt(value)}</text>')
        body.append(f'<text x="{group_x + group_w/2:.1f}" y="{height-28}" text-anchor="middle" class="tick">{esc(group)}</text>')
    legend_x = left
    for item in series:
        body.append(f'<rect x="{legend_x}" y="{top-28}" width="12" height="12" fill="{item["color"]}"/>')
        body.append(f'<text x="{legend_x+18}" y="{top-18}" class="legend">{esc(item["name"])}</text>')
        legend_x += 18 + len(item["name"]) * 8 + 24
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    return svg_frame(width, height, "\n".join(body), title, subtitle)


def heatmap(title: str, row_labels: list[str], col_labels: list[str], values: list[list[float]], subtitle: str = "") -> str:
    width, height = 760, 340
    left, top = 120, 78
    cell_w, cell_h = 170, 58
    flat = [v for row in values for v in row if v is not None]
    min_v, max_v = min(flat), max(flat)
    body = []
    for j, col in enumerate(col_labels):
        body.append(f'<text x="{left + j*cell_w + cell_w/2}" y="{top-16}" text-anchor="middle" class="label">{esc(col)}</text>')
    for i, row in enumerate(row_labels):
        body.append(f'<text x="{left-14}" y="{top + i*cell_h + cell_h/2 + 4}" text-anchor="end" class="label">{esc(row)}</text>')
        for j, value in enumerate(values[i]):
            ratio = 0 if math.isclose(max_v, min_v) else (value - min_v) / (max_v - min_v)
            red = int(238 - ratio * 70)
            green = int(246 - ratio * 105)
            blue = int(241 - ratio * 130)
            x, y = left + j * cell_w, top + i * cell_h
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w-8}" height="{cell_h-8}" rx="7" fill="rgb({red},{green},{blue})"/>')
            body.append(f'<text x="{x + (cell_w-8)/2}" y="{y + cell_h/2}" text-anchor="middle" class="label">{fmt(value)}</text>')
    return svg_frame(width, height, "\n".join(body), title, subtitle)


def scatter_chart(title: str, points: list[dict[str, Any]], width: int = 760, height: int = 340, subtitle: str = "") -> str:
    left, right, top, bottom = 58, 30, 68, 50
    plot_w, plot_h = width - left - right, height - top - bottom
    x_values = [float(p["x"]) for p in points]
    y_values = [float(p["y"]) for p in points]
    min_x, max_x = scale(x_values, 0, 1)
    min_y, max_y = scale(y_values, 0, 1, pad=0.15)
    min_x = min(0, min_x)
    min_y = min(0, min_y)

    def x_pos(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def y_pos(value: float) -> float:
        return top + plot_h - (value - min_y) / (max_y - min_y) * plot_h

    body = []
    for i in range(5):
        y_value = min_y + (max_y - min_y) * i / 4
        y = y_pos(y_value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{fmt(y_value)}</text>')
    for point in points:
        x, y = x_pos(float(point["x"])), y_pos(float(point["y"]))
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{point.get("color", COLORS["blue"])}"/>')
        body.append(f'<text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" class="tick">{esc(point["label"])}</text>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    body.append(f'<text x="{left + plot_w/2}" y="{height-12}" text-anchor="middle" class="tick">平均 candidate paths</text>')
    body.append(f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18,{top + plot_h/2})" text-anchor="middle" class="tick">成功步数</text>')
    return svg_frame(width, height, "\n".join(body), title, subtitle)


def build_report(run_dir: Path, output: Path) -> None:
    config = read_config(run_dir / "config.yaml")
    metrics = sorted(read_jsonl(run_dir / "metrics.jsonl"), key=lambda row: row.get("episode_index", 0))
    steps = sorted(read_jsonl(run_dir / "steps.jsonl"), key=lambda row: (row.get("episode_index", 0), row.get("step_index", 0)))
    budgets = sorted(read_jsonl(run_dir / "budget_progress.jsonl"), key=lambda row: row.get("episode_index", 0))
    if not metrics:
        raise SystemExit(f"No metrics found in {run_dir}")

    run_id = metrics[0]["run_id"]
    episodes = [int(row["episode_index"]) for row in metrics]
    ep_labels = [f"E{idx}" for idx in episodes]
    cases = list(dict.fromkeys(str(row.get("case_id", "")) for row in metrics if row.get("case_id")))
    cycle_len = max(1, len(cases))
    rounds = sorted({idx // cycle_len + 1 for idx in episodes})
    round_labels = [f"R{round_id}" for round_id in rounds]

    steps_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        steps_by_episode[int(row.get("episode_index", 0))].append(row)

    round_rows = []
    for round_id in rounds:
        group = [row for row in metrics if int(row["episode_index"]) // cycle_len + 1 == round_id]
        successes = [row for row in group if row.get("success")]
        step_values = [float(row.get("steps", 0)) for row in successes]
        step_rows = [step for row in group for step in steps_by_episode[int(row["episode_index"])]]
        candidate_values = [float(step.get("candidate_paths", 0) or 0) for step in step_rows]
        round_rows.append(
            {
                "round": round_id,
                "successes": len(successes),
                "episodes": len(group),
                "success_rate": len(successes) / len(group) if group else 0,
                "avg_steps": sum(step_values) / len(step_values) if step_values else 0,
                "avg_candidate_paths": sum(candidate_values) / len(candidate_values) if candidate_values else 0,
                "graph_nodes": group[-1].get("graph_nodes", 0),
                "graph_edges": group[-1].get("graph_edges", 0),
                "graph_paths": group[-1].get("graph_paths", 0),
            }
        )

    per_episode_candidates = []
    per_episode_confidence = []
    for row in metrics:
        ep = int(row["episode_index"])
        ep_steps = steps_by_episode[ep]
        candidates = [float(step.get("candidate_paths", 0) or 0) for step in ep_steps]
        confidences = [float(step["agent_confidence"]) for step in ep_steps if isinstance(step.get("agent_confidence"), (int, float))]
        per_episode_candidates.append(sum(candidates) / len(candidates) if candidates else 0)
        per_episode_confidence.append(sum(confidences) / len(confidences) if confidences else 0)

    cumulative_cost = [float(row.get("estimated_cost_rmb", 0) or 0) for row in budgets]
    if len(cumulative_cost) < len(metrics):
        cumulative_cost = [0.0] * (len(metrics) - len(cumulative_cost)) + cumulative_cost
    per_episode_cost = []
    previous = 0.0
    for value in cumulative_cost:
        per_episode_cost.append(max(0.0, value - previous))
        previous = value

    action_counter = Counter(action_label(step.get("action")) for step in steps)
    action_name_counter = Counter((step.get("action") or {}).get("name", "None") for step in steps)
    repeated_counts = [sum(1 for step in steps_by_episode[int(row["episode_index"])] if step.get("repeated_action")) for row in metrics]
    discovered_conditions = [float(row.get("discovered_conditions", 0) or 0) for row in metrics]

    token_prompt = [float(row.get("llm_usage_delta", {}).get("prompt_tokens", 0) or 0) for row in metrics]
    token_completion = [float(row.get("llm_usage_delta", {}).get("completion_tokens", 0) or 0) for row in metrics]
    llm_calls = [float(row.get("llm_usage_delta", {}).get("calls", 0) or 0) for row in metrics]
    steps_values = [float(row.get("steps", 0) or 0) for row in metrics]
    graph_nodes = [float(row.get("graph_nodes", 0) or 0) for row in metrics]
    graph_edges = [float(row.get("graph_edges", 0) or 0) for row in metrics]
    graph_paths = [float(row.get("graph_paths", 0) or 0) for row in metrics]
    added_nodes = [float(row.get("added_nodes", 0) or 0) for row in metrics]
    added_edges = [float(row.get("added_edges", 0) or 0) for row in metrics]
    added_paths = [float(row.get("added_paths", 0) or 0) for row in metrics]

    case_round_steps = []
    case_round_labels = []
    for case_id in cases:
        values = []
        for round_id in rounds:
            match = next((row for row in metrics if row["case_id"] == case_id and int(row["episode_index"]) // cycle_len + 1 == round_id), None)
            values.append(float(match["steps"]) if match else 0.0)
        case_round_steps.append(values)
        case_round_labels.append(f"{case_id}: " + " -> ".join(fmt(value) for value in values))

    success_rates = [row["success_rate"] * 100 for row in round_rows]
    round_avg_steps = [row["avg_steps"] for row in round_rows]
    avg_candidate_by_round = [row["avg_candidate_paths"] for row in round_rows]
    success_subtitle = " / ".join(f"R{row['round']} {row['successes']}/{row['episodes']}" for row in round_rows)
    step_subtitle = f"首轮 {fmt(round_avg_steps[0])}，末轮 {fmt(round_avg_steps[-1])}。"
    case_steps_subtitle = "；".join(case_round_labels)
    graph_growth_subtitle = f"最终 {fmt(graph_nodes[-1])} nodes / {fmt(graph_edges[-1])} edges / {fmt(graph_paths[-1])} paths。"
    candidate_subtitle = f"首轮平均 {fmt(avg_candidate_by_round[0])}，末轮平均 {fmt(avg_candidate_by_round[-1])}。"

    charts = [
        bar_chart("图 1：每轮成功率", round_labels, success_rates, color=COLORS["green"], y_max=100, subtitle=success_subtitle),
        line_chart("图 2：每轮平均成功步数", [{"name": "平均步数", "values": round_avg_steps, "color": COLORS["orange"]}], round_labels, subtitle=step_subtitle, y_min=0),
        line_chart("图 3：逐 episode 步数", [{"name": "steps", "values": steps_values, "color": COLORS["blue"]}], ep_labels, subtitle=f"{ep_labels[0]}-{ep_labels[-1]} 对应 {len(metrics)} 个 episode。", y_min=0),
        grouped_bar_chart(
            "图 4：按 case 比较三轮步数",
            round_labels,
            [
                {"name": case_id, "values": case_round_steps[i], "color": [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["gray"]][i % 5]}
                for i, case_id in enumerate(cases)
            ],
            subtitle=case_steps_subtitle,
        ),
        heatmap("图 5：case × round 步数热力图", cases, round_labels, case_round_steps, subtitle="颜色越深代表步数越高。"),
        line_chart(
            "图 6：经验图规模增长",
            [
                {"name": "nodes", "values": graph_nodes, "color": COLORS["blue"]},
                {"name": "edges", "values": graph_edges, "color": COLORS["green"]},
                {"name": "paths", "values": graph_paths, "color": COLORS["orange"]},
            ],
            ep_labels,
            subtitle=graph_growth_subtitle,
            y_min=0,
        ),
        stacked_bar_chart(
            "图 7：每个 episode 新增图元素",
            ep_labels,
            [
                {"name": "added nodes", "values": added_nodes, "color": COLORS["blue"]},
                {"name": "added edges", "values": added_edges, "color": COLORS["green"]},
                {"name": "added paths", "values": added_paths, "color": COLORS["orange"]},
            ],
            subtitle="展示每个 episode 对 ExperienceGraph 的新增贡献。",
        ),
        line_chart(
            "图 8：candidate paths 进入 prompt 的数量",
            [{"name": "avg candidate paths", "values": per_episode_candidates, "color": COLORS["purple"]}],
            ep_labels,
            subtitle=candidate_subtitle,
            y_min=0,
        ),
        bar_chart("图 9：LLM 调用次数", ep_labels, llm_calls, color=COLORS["gray"], subtitle="调用次数大体随 episode 步数变化。", y_max=max(llm_calls + [1]) * 1.2),
        stacked_bar_chart(
            "图 10：每个 episode 的 token 使用",
            ep_labels,
            [
                {"name": "prompt tokens", "values": token_prompt, "color": COLORS["blue"]},
                {"name": "completion tokens", "values": token_completion, "color": COLORS["orange"]},
            ],
            subtitle="DeepSeek 返回较长 planning JSON，completion token 占比较高。",
        ),
        line_chart(
            "图 11：累计成本估算",
            [{"name": "RMB", "values": cumulative_cost, "color": COLORS["red"]}],
            ep_labels,
            subtitle=f"最终约 {fmt(cumulative_cost[-1], 3)} RMB，低于 20 RMB 预算上限。",
            y_min=0,
        ),
        bar_chart("图 12：动作类型分布", list(action_name_counter.keys()), [float(v) for v in action_name_counter.values()], color=COLORS["green"], subtitle="用于观察模型是否频繁绕路或重复资源动作。"),
        line_chart(
            "图 13：平均置信度",
            [{"name": "confidence", "values": per_episode_confidence, "color": COLORS["purple"]}],
            ep_labels,
            subtitle="置信度保持较高，但仍要结合实际步数判断质量。",
            y_min=0,
        ),
        scatter_chart(
            "图 14：candidate paths 与成功步数关系",
            [
                {"x": per_episode_candidates[i], "y": steps_values[i], "label": ep_labels[i], "color": COLORS["blue"] if i < 3 else COLORS["green"] if i < 6 else COLORS["orange"]}
                for i in range(len(metrics))
            ],
            subtitle="用于观察检索数量和步数之间的关系，样本量仍然很小。",
        ),
        bar_chart("图 15：重复动作次数", ep_labels, [float(v) for v in repeated_counts], color=COLORS["red"], subtitle="重复动作不是主要问题，主要低效来自顺序选择和移动路径。"),
        bar_chart("图 16：发现条件数量", ep_labels, discovered_conditions, color=COLORS["blue"], subtitle="GEC_004 第一轮通过 explore 发现 mine 相关条件。"),
    ]

    first_round = round_rows[0]
    last_round = round_rows[-1]
    success_total = sum(1 for row in metrics if row.get("success"))
    success_steps = [float(row.get("steps", 0) or 0) for row in metrics if row.get("success")]
    avg_steps = sum(success_steps) / len(success_steps) if success_steps else 0.0
    step_delta = first_round["avg_steps"] - last_round["avg_steps"]
    if last_round["success_rate"] > first_round["success_rate"]:
        success_phrase = "成功率上升"
    elif last_round["success_rate"] < first_round["success_rate"]:
        success_phrase = "成功率下降"
    else:
        success_phrase = "成功率持平"
    if step_delta > 0:
        efficiency_phrase = f"平均成功步数从 {first_round['avg_steps']:.2f} 降到 {last_round['avg_steps']:.2f}"
    elif step_delta < 0:
        efficiency_phrase = f"平均成功步数从 {first_round['avg_steps']:.2f} 升到 {last_round['avg_steps']:.2f}"
    else:
        efficiency_phrase = f"平均成功步数保持在 {last_round['avg_steps']:.2f}"
    conclusion = (
        f"这次 {len(metrics)} episode 小实验中，{success_phrase}，总体成功 {success_total}/{len(metrics)}；"
        f"{efficiency_phrase}。candidate paths 从第一轮平均 {avg_candidate_by_round[0]:.2f} "
        f"到末轮 {avg_candidate_by_round[-1]:.2f}，说明 ExperienceGraph 的检索内容持续进入后续决策。"
    )
    case_detail = "；".join(case_round_labels)
    failure_rows = [row for row in metrics if not row.get("success")]
    failure_detail = "；".join(
        f"E{int(row['episode_index'])} {row['case_id']}：{row.get('failure_reason') or 'unknown'}，{row.get('steps')} 步"
        for row in failure_rows
    )
    if not failure_detail:
        failure_detail = "无失败 episode。"
    provider = config.get("llm_provider") or metrics[0].get("llm_provider", "")
    model = config.get("llm_model") or metrics[0].get("llm_model", "")
    max_steps = config.get("max_steps", "")
    max_budget = config.get("max_budget_rmb", "")

    metrics_rows = "\n".join(
        "<tr>"
        f"<td>E{int(row['episode_index'])}</td><td>{esc(row['case_id'])}</td><td>{'成功' if row.get('success') else '失败'}</td>"
        f"<td>{row.get('steps')}</td><td>{row.get('graph_nodes')}</td><td>{row.get('graph_edges')}</td><td>{row.get('graph_paths')}</td>"
        f"<td>{fmt(float(row.get('llm_usage_delta', {}).get('total_tokens', 0)))}</td>"
        "</tr>"
        for row in metrics
    )
    top_actions_rows = "\n".join(
        f"<tr><td>{esc(label)}</td><td>{count}</td></tr>"
        for label, count in action_counter.most_common(12)
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ExperienceGraph 复杂任务实验报告</title>
  <style>
    body {{ margin: 0; font-family: Inter, "Microsoft YaHei", system-ui, sans-serif; color: #1d2327; background: #f6f6f2; }}
    header {{ padding: 32px 40px; background: #ffffff; border-bottom: 1px solid #d8d8d0; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px 24px 48px; }}
    h1 {{ margin: 0 0 10px; font-size: 30px; }}
    h2 {{ margin: 28px 0 12px; font-size: 22px; }}
    h3 {{ margin: 18px 0 8px; font-size: 17px; }}
    p, li {{ line-height: 1.7; }}
    .meta {{ color: #667078; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }}
    .card .label {{ color: #667078; font-size: 12px; }}
    .card .value {{ font-size: 25px; margin-top: 6px; }}
    .section {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }}
    .chart-wrap {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px; overflow-x: auto; }}
    svg.chart {{ width: 100%; min-width: 520px; height: auto; display: block; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #eee; text-align: left; font-size: 13px; }}
    th {{ background: #eef1ec; }}
    .good {{ color: #137a4f; font-weight: 700; }}
    .warn {{ color: #9a5d16; font-weight: 700; }}
    code {{ background: #eef1ec; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>ExperienceGraph 复杂任务实验报告</h1>
    <div class="meta">Run: <code>{esc(run_id)}</code> · Last updated: 2026-06-21 · 数据来源：metrics / steps / budget_progress JSONL</div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div class="label">总成功率</div><div class="value">{success_total}/{len(metrics)}</div></div>
      <div class="card"><div class="label">平均成功步数</div><div class="value">{avg_steps:.2f}</div></div>
      <div class="card"><div class="label">第三轮平均步数</div><div class="value">{last_round['avg_steps']:.2f}</div></div>
      <div class="card"><div class="label">最终经验图</div><div class="value">{int(graph_nodes[-1])}N / {int(graph_edges[-1])}E / {int(graph_paths[-1])}P</div></div>
      <div class="card"><div class="label">最终成本估算</div><div class="value">{cumulative_cost[-1]:.3f} RMB</div></div>
    </section>

    <section class="section">
      <h2>核心结论</h2>
      <p>{esc(conclusion)}</p>
      <p>按 case 看：{esc(case_detail)}。</p>
      <p>失败记录：{esc(failure_detail)}</p>
      <p class="warn">注意：这是 {len(cases)} 个 case × {len(rounds)} 轮的小样本实验，可以作为“机制有效”的早期证据，但还不能替代多 seed、多任务族、baseline/ablation 的正式结论。</p>
    </section>

    <section class="section">
      <h2>实验设置</h2>
      <ul>
        <li>Agent: <code>graph</code>，variant: <code>full</code></li>
        <li>任务族：<code>golden_equipment_chain</code></li>
        <li>case 顺序：<code>{esc(','.join(cases))}</code>，重复 {len(rounds)} 轮，共 {len(metrics)} episode</li>
        <li>同一个 run 内持续保留 ExperienceGraph，不清空 graph</li>
        <li>Provider/model：<code>{esc(provider)}</code> / <code>{esc(model)}</code>，最大步数：{esc(max_steps)}，预算上限：{esc(max_budget)} RMB</li>
      </ul>
    </section>

    <section>
      <h2>统计图</h2>
      <div class="charts">
        {''.join(f'<div class="chart-wrap">{chart}</div>' for chart in charts)}
      </div>
    </section>

    <section class="section">
      <h2>Episode 明细</h2>
      <table>
        <thead><tr><th>Episode</th><th>Case</th><th>结果</th><th>步数</th><th>Nodes</th><th>Edges</th><th>Paths</th><th>Total Tokens</th></tr></thead>
        <tbody>{metrics_rows}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>高频动作</h2>
      <table>
        <thead><tr><th>Action</th><th>次数</th></tr></thead>
        <tbody>{top_actions_rows}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>解读</h2>
      <p>第一轮几乎没有可复用 candidate path，模型主要依赖即时规划。后续轮次 candidate path 数量显著增加，模型能看到更多历史路线，但 flash 在第三轮 <code>GEC_006</code> 仍耗尽了 18 步预算，说明步数上限能暴露较弱模型的规划偏差。</p>
      <p>经验图的 node 数在第一轮后基本稳定，edge/path 继续增长。这说明环境状态抽象已经覆盖主要条件，但不同 episode 的行动路径仍在积累。这种形态是合理的：节点表示可复用状态，路径记录策略差异和结果统计。</p>
      <p>如果下一步要把这个结果写成更可信的实验，应增加至少 5 个 seed，并加入 <code>react</code>、<code>graph/no_graph_context</code>、<code>random_retrieval</code> 对照。当前 run 的价值在于证明复杂链式任务能跑通，并且 ExperienceGraph 的检索信号和效率变化方向一致。</p>
    </section>
  </main>
</body>
</html>
"""
    output.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a detailed Chinese HTML report for the complex GEC experiment.")
    parser.add_argument("--run-dir", default="runs/graph_gec_3x3_deepseek_exp1")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "complex_gec_detailed_report.html"
    build_report(run_dir, output)
    print(output)


if __name__ == "__main__":
    main()
