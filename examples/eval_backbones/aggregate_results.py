"""Aggregate success/failure rollouts produced by examples/rollout_openha.py
into a per-task and overall success-rate summary.

Directory layout expected (as produced by rollout_openha.py):
  <record_path>/<model_id>-<output_mode>/<task_name>/<timestamp>-uuid/{success.json|loss.json}
"""
import argparse
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    per_task = defaultdict(lambda: {"success": 0, "total": 0, "frames_on_success": []})

    if os.path.isdir(args.record_path):
        for model_id_dir in os.listdir(args.record_path):
            model_id_path = os.path.join(args.record_path, model_id_dir)
            if not os.path.isdir(model_id_path):
                continue
            for task_name in os.listdir(model_id_path):
                task_path = os.path.join(model_id_path, task_name)
                if not os.path.isdir(task_path):
                    continue
                for run_dir in os.listdir(task_path):
                    run_path = os.path.join(task_path, run_dir)
                    if not os.path.isdir(run_path):
                        continue
                    success_file = os.path.join(run_path, "success.json")
                    loss_file = os.path.join(run_path, "loss.json")
                    if os.path.isfile(success_file):
                        per_task[task_name]["total"] += 1
                        per_task[task_name]["success"] += 1
                        try:
                            with open(success_file) as f:
                                per_task[task_name]["frames_on_success"].append(json.load(f).get("frames"))
                        except Exception:
                            pass
                    elif os.path.isfile(loss_file):
                        per_task[task_name]["total"] += 1

    overall_success = sum(v["success"] for v in per_task.values())
    overall_total = sum(v["total"] for v in per_task.values())

    summary = {
        "model_name": args.model_name,
        "per_task": {
            task: {
                "success": v["success"],
                "total": v["total"],
                "success_rate": (v["success"] / v["total"]) if v["total"] else None,
                "avg_frames_on_success": (
                    sum(f for f in v["frames_on_success"] if f is not None) / len(v["frames_on_success"])
                    if v["frames_on_success"] else None
                ),
            }
            for task, v in per_task.items()
        },
        "overall": {
            "success": overall_success,
            "total": overall_total,
            "success_rate": (overall_success / overall_total) if overall_total else None,
        },
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
