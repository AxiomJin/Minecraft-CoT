"""Aggregate success/failure rollouts produced by examples/rollout_openha.py
into a per-task and overall success-rate summary.

Directory layout expected (as produced by rollout_openha.py):
  <record_path>/<model_id>-<output_mode>/<task_name>/<timestamp>-uuid/{success.json|loss.json}
"""
import argparse
import json
import os
from collections import defaultdict


# 与论文 Embodied / GUI / Combat 三大类的对应关系（详见
# openagents/envs/tasks/task_manager.py 里 5 个任务生成器的命名前缀）：
#   Embodied = mine_block:*                              (导航+挖掘/砍伐，如"砍一棵树")
#   GUI      = craft_item:* / smelt_item:* / custom:interact_with_*  (crafting table/furnace 等GUI交互)
#   Combat   = kill_entity:*                             (击杀生物)
def classify_task_category(task_name: str) -> str:
    if task_name.startswith("mine_block:"):
        return "Embodied"
    if task_name.startswith("kill_entity:"):
        return "Combat"
    if task_name.startswith("craft_item:") or task_name.startswith("smelt_item:") or "interact_with" in task_name:
        return "GUI"
    return "Other"


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

    # 按 Embodied / GUI / Combat 三大类聚合，对应论文 Table 3 的分组统计
    # (avg_frames_on_success 对应论文的 "Steps" 列，success_rate 对应 "ASR" 列)。
    per_category = defaultdict(lambda: {"success": 0, "total": 0, "frames_on_success": [], "tasks": set()})
    for task, v in per_task.items():
        cat = classify_task_category(task)
        per_category[cat]["success"] += v["success"]
        per_category[cat]["total"] += v["total"]
        per_category[cat]["frames_on_success"].extend(v["frames_on_success"])
        per_category[cat]["tasks"].add(task)

    summary = {
        "model_name": args.model_name,
        "per_task": {
            task: {
                "category": classify_task_category(task),
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
        "by_category": {
            cat: {
                "num_distinct_tasks": len(v["tasks"]),
                "success": v["success"],
                "total": v["total"],
                "success_rate": (v["success"] / v["total"]) if v["total"] else None,
                "avg_frames_on_success": (
                    sum(f for f in v["frames_on_success"] if f is not None) / len(v["frames_on_success"])
                    if v["frames_on_success"] else None
                ),
            }
            for cat, v in per_category.items()
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
