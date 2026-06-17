"""Integration tests: verify pipeline output is consumable by FastAI ImageDataLoaders."""

import json
import shutil
from pathlib import Path

import pytest

from tidd.pipeline import GADFEncoder, PILWriter, Pipeline

TEST_DATA_FILE = Path("tests/data/ahup3020.12o_G20.txt")
LABELS_DICT = {
    "302": {
        "G04": {"start": 31400, "finish": 33200},
        "G07": {"start": 31160, "finish": 32960},
        "G08": {"start": 31900, "finish": 33700},
        "G10": {"start": 29900, "finish": 31700},
        "G20": {"start": 31150, "finish": 32950},
    }
}


@pytest.fixture
def pipeline_output(tmp_path):
    """Run the pipeline on a single test file and return the output path."""
    loc = tmp_path / "raw" / "test_loc" / "2012" / "302"
    loc.mkdir(parents=True)
    shutil.copy(TEST_DATA_FILE, loc / "ahup3020.12o_G20.txt")

    labels_file = tmp_path / "raw" / "test_loc" / "tid_start_finish_times.json"
    labels_file.write_text(json.dumps(LABELS_DICT))

    out_dir = tmp_path / "experiments" / "integration_test"

    pipe = Pipeline(encoder=GADFEncoder(), writer=PILWriter(), window_size=60, workers=1)
    pipe.run(
        raw_paths=[str(tmp_path / "raw" / "test_loc")],
        output_dir=str(out_dir),
        split="train",
        verbose=False,
    )

    return out_dir


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestPipelineIntegration:

    def test_output_directory_structure(self, pipeline_output):
        """Verify the pipeline produces the expected labeled directory layout."""
        labeled_dirs = list(pipeline_output.rglob("labeled"))
        assert len(labeled_dirs) > 0

        all_jpgs = list(pipeline_output.rglob("*.jpg"))
        assert len(all_jpgs) > 0

        classes = {p.parent.name for p in all_jpgs}
        assert "anomalous" in classes or "normal" in classes

    def test_fastai_image_data_loaders(self, pipeline_output):
        """Verify FastAI ImageDataLoaders can load the pipeline output."""
        from fastai.vision.all import ImageDataLoaders, Resize

        labeled_dir = list(pipeline_output.rglob("labeled"))[0]

        dls = ImageDataLoaders.from_folder(
            labeled_dir,
            item_tfms=Resize(224),
            valid_pct=0.2,
            bs=8,
        )

        assert dls is not None
        assert len(dls.vocab) >= 1

        batch = dls.one_batch()
        assert batch[0].shape[-1] == 224
        assert batch[0].shape[-2] == 224

    def test_filename_format(self, pipeline_output):
        """Verify generated filenames follow the expected pattern."""
        jpgs = list(pipeline_output.rglob("*.jpg"))
        assert len(jpgs) > 0

        for jpg in jpgs:
            parts = jpg.stem.split("_")
            assert parts[-1] == "GAF"
            assert len(parts) == 5
