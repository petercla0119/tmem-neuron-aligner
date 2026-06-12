from pathlib import Path

from tmem_align.nd2_tools import build_manifest


def test_build_manifest_empty_folder(tmp_path: Path) -> None:
    output = tmp_path / "manifest.csv"
    df = build_manifest(tmp_path, output_csv=output)
    assert df.empty
    assert output.exists()
