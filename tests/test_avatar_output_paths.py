import argparse

import pytest

from run_demo_avatar_single_audio_to_video import (
    build_output_paths,
    normalize_output_name,
    prepare_output_paths,
)


def test_default_output_name_is_preserved(tmp_path):
    paths = build_output_paths(tmp_path, None, "ai2v", 1)
    assert paths == [tmp_path / "ai2v_demo_1.mp4"]


@pytest.mark.parametrize(
    ("provided", "expected"),
    [("example", "example.mp4"), ("example.mp4", "example.mp4")],
)
def test_output_name_normalization(provided, expected):
    assert normalize_output_name(provided) == expected


def test_named_continuations_use_the_primary_stem(tmp_path):
    paths = build_output_paths(tmp_path, "example.mp4", "ai2v", 3)
    assert [path.name for path in paths] == [
        "example.mp4",
        "example_continue_2.mp4",
        "example_continue_3.mp4",
    ]


@pytest.mark.parametrize(
    "invalid_name",
    ["../example.mp4", "subdir/example.mp4", r"subdir\example.mp4", "", ".", ".."],
)
def test_invalid_output_names_are_rejected(invalid_name):
    with pytest.raises(argparse.ArgumentTypeError):
        normalize_output_name(invalid_name)


def test_existing_named_target_is_rejected(tmp_path):
    target = tmp_path / "example.mp4"
    target.touch()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_output_paths(tmp_path, "example.mp4", "ai2v", 1)


def test_missing_output_directory_is_created(tmp_path):
    output_dir = tmp_path / "missing" / "output"
    paths = prepare_output_paths(output_dir, "example", "ai2v", 1)

    assert output_dir.is_dir()
    assert paths == [output_dir / "example.mp4"]
