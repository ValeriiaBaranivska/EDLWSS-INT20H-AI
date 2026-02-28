from dotenv import load_dotenv

load_dotenv(".env", override=True)

from core.pipeline import Pipeline

p = Pipeline()
result = p.run(seed=12345)

d = result.model_dump()
print("=== CHECKS ===")
print("messages:", len(d["messages"]))
print("turns:", [(m["role"], m["turn"]) for m in d["messages"]])
print("stages_run:", d["meta"]["stages_run"])
print("stages_skipped:", d["meta"]["stages_skipped"])
print("dirt_applied:", d["meta"]["dirt_applied"])
print("total_time_s:", d["meta"]["total_time_s"])
print()
print(result.model_dump_json(indent=2))
print()
print("Stage 2 ready")
