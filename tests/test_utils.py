from token_classification.utils import create_timestamped_run_directory


def test_timestamped_run_directories_preserve_same_second_runs(tmp_path):
    first = create_timestamped_run_directory(
        tmp_path,
        "experiment",
        timestamp="2026-09-02_10-30-25",
    )
    second = create_timestamped_run_directory(
        tmp_path,
        "experiment",
        timestamp="2026-09-02_10-30-25",
    )

    assert first.name == "experiment_2026-09-02_10-30-25"
    assert second.name == "experiment_2026-09-02_10-30-25_02"
