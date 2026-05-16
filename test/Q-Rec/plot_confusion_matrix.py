"""Create an SVG heatmap and per-class metrics from a confusion matrix CSV."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot classifier confusion matrix as an SVG heatmap.")
    parser.add_argument("matrix_csv", type=Path, help="Confusion matrix CSV from train/eval output.")
    parser.add_argument("--output-svg", type=Path, default=None, help="Output SVG path.")
    parser.add_argument("--output-metrics", type=Path, default=None, help="Output per-class metrics CSV path.")
    return parser.parse_args()


def read_matrix(path: Path) -> tuple[list[str], list[list[int]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"Invalid confusion matrix CSV: {path}")
    class_names = rows[0][1:]
    matrix = [[int(value) for value in row[1:]] for row in rows[1:]]
    if len(matrix) != len(class_names):
        raise ValueError("Confusion matrix row count does not match class count.")
    for row in matrix:
        if len(row) != len(class_names):
            raise ValueError("Confusion matrix must be square.")
    return class_names, matrix


def compute_metrics(class_names: list[str], matrix: list[list[int]]) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    total = sum(sum(row) for row in matrix)
    rows: list[dict[str, float | int | str]] = []
    for idx, class_name in enumerate(class_names):
        tp = matrix[idx][idx]
        support = sum(matrix[idx])
        predicted_total = sum(matrix[row_idx][idx] for row_idx in range(len(class_names)))
        precision = 0.0 if predicted_total == 0 else tp / predicted_total
        recall = 0.0 if support == 0 else tp / support
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "class_id": idx,
                "class_name": class_name,
                "support": support,
                "predicted_total": predicted_total,
                "tp": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    correct = sum(matrix[idx][idx] for idx in range(len(class_names)))
    macro_precision = sum(float(row["precision"]) for row in rows) / len(rows)
    macro_recall = sum(float(row["recall"]) for row in rows) / len(rows)
    macro_f1 = sum(float(row["f1"]) for row in rows) / len(rows)
    weighted_f1 = 0.0 if total == 0 else sum(float(row["f1"]) * int(row["support"]) for row in rows) / total
    summary = {
        "overall_accuracy": 0.0 if total == 0 else correct / total,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    return rows, summary


def color_for(value: float) -> str:
    value = max(0.0, min(1.0, value))
    # Light blue to deep indigo.
    start = (239, 246, 255)
    end = (49, 46, 129)
    rgb = tuple(round(start[i] + (end[i] - start[i]) * value) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def text_color(value: float) -> str:
    return "#ffffff" if value >= 0.48 else "#111827"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_metrics(path: Path, metric_rows: list[dict[str, float | int | str]], summary: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["class_id", "class_name", "support", "predicted_total", "tp", "precision", "recall", "f1"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(
                {
                    **row,
                    "precision": pct(float(row["precision"])),
                    "recall": pct(float(row["recall"])),
                    "f1": pct(float(row["f1"])),
                }
            )
        writer.writerow({})
        writer.writerow({"class_name": "overall_accuracy", "precision": pct(summary["overall_accuracy"])})
        writer.writerow({"class_name": "macro_precision", "precision": pct(summary["macro_precision"])})
        writer.writerow({"class_name": "macro_recall", "precision": pct(summary["macro_recall"])})
        writer.writerow({"class_name": "macro_f1", "precision": pct(summary["macro_f1"])})
        writer.writerow({"class_name": "weighted_f1", "precision": pct(summary["weighted_f1"])})


def write_svg(
    path: Path,
    *,
    source_name: str,
    class_names: list[str],
    matrix: list[list[int]],
    metric_rows: list[dict[str, float | int | str]],
    summary: dict[str, float],
) -> None:
    n = len(class_names)
    cell = 34
    left = 92
    top = 120
    metric_left = left + n * cell + 22
    width = metric_left + 280
    height = top + n * cell + 92
    max_row_total = max(max(sum(row), 1) for row in matrix)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:11px}.tiny{font-size:9px}.label{font-size:12px;font-weight:600}.title{font-size:22px;font-weight:700}",
        ".metric{font-size:11px}.axis{font-size:13px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" class="title">Confusion Matrix Heatmap</text>',
        f'<text x="{left}" y="56" class="small">Source: {html.escape(source_name)}</text>',
        (
            f'<text x="{left}" y="78" class="small">'
            f'Overall Acc: {pct(summary["overall_accuracy"])} | '
            f'Macro Precision: {pct(summary["macro_precision"])} | '
            f'Macro Recall: {pct(summary["macro_recall"])} | '
            f'Macro F1: {pct(summary["macro_f1"])} | '
            f'Weighted F1: {pct(summary["weighted_f1"])}</text>'
        ),
        f'<text x="{left + n * cell / 2 - 48}" y="{top - 60}" class="axis">Predicted class</text>',
        f'<text x="18" y="{top + n * cell / 2 + 36}" class="axis" transform="rotate(-90 18 {top + n * cell / 2 + 36})">Actual class</text>',
    ]

    for idx, name in enumerate(class_names):
        x = left + idx * cell + cell / 2
        y = top - 12
        parts.append(
            f'<text x="{x}" y="{y}" class="small" text-anchor="start" transform="rotate(-45 {x} {y})">{html.escape(name)}</text>'
        )
        parts.append(f'<text x="{left - 10}" y="{top + idx * cell + 22}" class="label" text-anchor="end">{html.escape(name)}</text>')

    for row_idx, row in enumerate(matrix):
        row_total = max(sum(row), 1)
        for col_idx, count in enumerate(row):
            normalized = count / row_total
            x = left + col_idx * cell
            y = top + row_idx * cell
            stroke = "#111827" if row_idx == col_idx else "#d1d5db"
            stroke_width = "1.2" if row_idx == col_idx else "0.5"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color_for(normalized)}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}"><title>'
                f'actual={html.escape(class_names[row_idx])}, predicted={html.escape(class_names[col_idx])}, '
                f'count={count}, row_pct={pct(normalized)}</title></rect>'
            )
            if count > 0:
                parts.append(
                    f'<text x="{x + cell / 2}" y="{y + 15}" class="tiny" text-anchor="middle" '
                    f'fill="{text_color(normalized)}">{count}</text>'
                )
                parts.append(
                    f'<text x="{x + cell / 2}" y="{y + 27}" class="tiny" text-anchor="middle" '
                    f'fill="{text_color(normalized)}">{normalized * 100:.0f}%</text>'
                )

    parts.append(f'<text x="{metric_left}" y="{top - 24}" class="axis">Per-class metrics</text>')
    parts.append(f'<text x="{metric_left}" y="{top - 6}" class="metric">Class</text>')
    parts.append(f'<text x="{metric_left + 58}" y="{top - 6}" class="metric">Prec</text>')
    parts.append(f'<text x="{metric_left + 110}" y="{top - 6}" class="metric">Recall</text>')
    parts.append(f'<text x="{metric_left + 166}" y="{top - 6}" class="metric">F1</text>')
    parts.append(f'<text x="{metric_left + 215}" y="{top - 6}" class="metric">N</text>')

    for idx, row in enumerate(metric_rows):
        y = top + idx * cell + 22
        parts.append(f'<text x="{metric_left}" y="{y}" class="metric">{html.escape(str(row["class_name"]))}</text>')
        parts.append(f'<text x="{metric_left + 58}" y="{y}" class="metric">{float(row["precision"]) * 100:.1f}%</text>')
        parts.append(f'<text x="{metric_left + 110}" y="{y}" class="metric">{float(row["recall"]) * 100:.1f}%</text>')
        parts.append(f'<text x="{metric_left + 166}" y="{y}" class="metric">{float(row["f1"]) * 100:.1f}%</text>')
        parts.append(f'<text x="{metric_left + 215}" y="{y}" class="metric">{row["support"]}</text>')

    legend_x = left
    legend_y = top + n * cell + 38
    for idx in range(11):
        value = idx / 10
        parts.append(
            f'<rect x="{legend_x + idx * 34}" y="{legend_y}" width="34" height="12" fill="{color_for(value)}" stroke="#ffffff"/>'
        )
    parts.append(f'<text x="{legend_x}" y="{legend_y + 30}" class="small">0% of actual class</text>')
    parts.append(f'<text x="{legend_x + 318}" y="{legend_y + 30}" class="small">100%</text>')
    parts.append(
        f'<text x="{left}" y="{height - 18}" class="small">'
        f'Cell text shows count and row percentage. Heat color is normalized within each actual-class row. '
        f'Largest class support in a row: {max_row_total}.</text>'
    )
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    class_names, matrix = read_matrix(args.matrix_csv)
    metric_rows, summary = compute_metrics(class_names, matrix)
    output_svg = args.output_svg or args.matrix_csv.with_name(f"{args.matrix_csv.stem}_heatmap.svg")
    output_metrics = args.output_metrics or args.matrix_csv.with_name(f"{args.matrix_csv.stem}_metrics.csv")
    write_svg(
        output_svg,
        source_name=str(args.matrix_csv),
        class_names=class_names,
        matrix=matrix,
        metric_rows=metric_rows,
        summary=summary,
    )
    write_metrics(output_metrics, metric_rows, summary)
    print(f"SVG heatmap saved to {output_svg}")
    print(f"Per-class metrics saved to {output_metrics}")
    print(
        f"overall_accuracy={pct(summary['overall_accuracy'])} "
        f"macro_precision={pct(summary['macro_precision'])} "
        f"macro_recall={pct(summary['macro_recall'])} "
        f"macro_f1={pct(summary['macro_f1'])}"
    )


if __name__ == "__main__":
    main()
