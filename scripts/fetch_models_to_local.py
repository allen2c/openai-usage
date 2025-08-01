import json
import pathlib

from openai_usage.extra.open_router import get_models

path = pathlib.Path("openai_usage/models.json")


if __name__ == "__main__":
    models_result = get_models()
    models_result.data.sort(key=lambda x: x.id)
    path.write_text(
        json.dumps(
            json.loads(models_result.model_dump_json()),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
