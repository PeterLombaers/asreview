import json
from pathlib import Path

import asreview as asr


def test_project_migration_1_to_2(tmpdir):
    asreview_v1_file = Path(
        "asreview",
        "webapp",
        "tests",
        "asreview-project-file-archive",
        "v1.5",
        "asreview-project-v1-5-startreview.asreview",
    )

    project = asr.Project.load(open(asreview_v1_file, "rb"), tmpdir, safe_import=True)

    assert project.config["version"].startswith("2.")

    with asr.open_state(project) as state:
        state.get_results_table()
        state.get_last_ranking_table()
        state.get_decision_changes()

    with open(
        Path(
            project.project_path,
            "reviews",
            project.reviews[0]["id"],
            "settings_metadata.json",
        )
    ) as f:
        data = json.load(f)

    cycle_data = asr.ActiveLearningCycleData(**data["current_value"])
    cycle = asr.ActiveLearningCycle.from_meta(cycle_data)

    assert isinstance(cycle.classifier.name, str)
