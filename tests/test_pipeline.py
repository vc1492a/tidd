"""Unit tests for tidd.pipeline modules."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tidd.pipeline.discovery import FileInfo, discover_files
from tidd.pipeline.encoders import Encoder, GADFEncoder
from tidd.pipeline.labels import Labels
from tidd.pipeline.preprocess import Window, read_and_resample, split_events, window_events
from tidd.pipeline.runner import Pipeline, process_file
from tidd.pipeline.writers import MatplotlibWriter, PILWriter, Writer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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
def fixture_dir(tmp_path):
    """Create a minimal directory tree that mimics the raw data layout."""
    loc = tmp_path / "test_location" / "2012" / "302"
    loc.mkdir(parents=True)
    shutil.copy(TEST_DATA_FILE, loc / "ahup3020.12o_G20.txt")

    labels_file = tmp_path / "test_location" / "tid_start_finish_times.json"
    labels_file.write_text(json.dumps(LABELS_DICT))

    return tmp_path / "test_location"


@pytest.fixture
def file_info():
    return FileInfo(
        path=TEST_DATA_FILE,
        location="test_location",
        year=2012,
        day_of_year=302,
        station="ahup",
        satellite="G20",
    )


# ---------------------------------------------------------------------------
# discovery tests
# ---------------------------------------------------------------------------


class TestDiscovery:

    def test_discover_files(self, fixture_dir):
        files = discover_files(fixture_dir)
        assert len(files) == 1
        fi = files[0]
        assert isinstance(fi, FileInfo)
        assert fi.location == "test_location"
        assert fi.year == 2012
        assert fi.day_of_year == 302
        assert fi.station == "ahup"
        assert fi.satellite == "G20"

    def test_discover_files_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert discover_files(empty) == []

    def test_discover_files_nonexistent(self, tmp_path):
        assert discover_files(tmp_path / "no_such_dir") == []


# ---------------------------------------------------------------------------
# preprocess tests
# ---------------------------------------------------------------------------


class TestPreprocess:

    def test_read_and_resample(self, file_info):
        df = read_and_resample(file_info, resample_freq="1min")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.shape[0] > 0

    def test_split_events(self, file_info):
        df = read_and_resample(file_info, resample_freq="1min")
        events = split_events(df, min_length=100)
        assert len(events) > 0
        for ev in events:
            assert isinstance(ev, pd.DataFrame)
            assert ev.shape[0] >= 100

    def test_window_events(self, file_info):
        df = read_and_resample(file_info, resample_freq="1min")
        events = split_events(df, min_length=100)
        windows = window_events(events, window_size=60)
        assert len(windows) > 0
        for w in windows:
            assert isinstance(w, Window)
            assert len(w.array) == 60
            assert w.day_of_year == 302


# ---------------------------------------------------------------------------
# encoder tests
# ---------------------------------------------------------------------------


class TestEncoders:

    def test_gadf_encoder_shape(self):
        enc = GADFEncoder()
        series = np.random.randn(60)
        result = enc.encode(series)
        assert result.shape == (60, 60)

    def test_gadf_encoder_satisfies_protocol(self):
        assert isinstance(GADFEncoder(), Encoder)

    def test_custom_encoder_protocol(self):
        class DummyEncoder:
            def encode(self, series: np.ndarray) -> np.ndarray:
                return np.zeros((len(series), len(series)))

        assert isinstance(DummyEncoder(), Encoder)


# ---------------------------------------------------------------------------
# writer tests
# ---------------------------------------------------------------------------


class TestWriters:

    def test_pil_writer(self, tmp_path):
        writer = PILWriter(colormap="viridis", size=(224, 224))
        array = np.random.rand(60, 60)
        out = tmp_path / "test.jpg"
        writer.save(array, out)

        assert out.exists()
        from PIL import Image
        img = Image.open(out)
        assert img.size == (224, 224)

    def test_matplotlib_writer(self, tmp_path):
        writer = MatplotlibWriter()
        array = np.random.rand(60, 60)
        out = tmp_path / "test.jpg"
        writer.save(array, out)

        assert out.exists()
        from PIL import Image
        img = Image.open(out)
        assert img.size[0] > 0

    def test_pil_writer_satisfies_protocol(self):
        assert isinstance(PILWriter(), Writer)

    def test_matplotlib_writer_satisfies_protocol(self):
        assert isinstance(MatplotlibWriter(), Writer)


# ---------------------------------------------------------------------------
# labels tests
# ---------------------------------------------------------------------------


class TestLabels:

    def test_labels_load(self, fixture_dir):
        labels = Labels.from_raw_path(fixture_dir)
        assert labels._data == LABELS_DICT

    def test_classify_window_anomalous(self):
        labels = Labels(LABELS_DICT)
        result = labels.classify_window(day_of_year=302, satellite="G20", sod_end=31500.0)
        assert result == "anomalous"

    def test_classify_window_normal(self):
        labels = Labels(LABELS_DICT)
        result = labels.classify_window(day_of_year=302, satellite="G20", sod_end=10000.0)
        assert result == "normal"

    def test_classify_window_no_label(self):
        labels = Labels(LABELS_DICT)
        result = labels.classify_window(day_of_year=302, satellite="G99", sod_end=31500.0)
        assert result == "normal"

    def test_classify_window_unknown_doy(self):
        labels = Labels(LABELS_DICT)
        result = labels.classify_window(day_of_year=999, satellite="G20", sod_end=31500.0)
        assert result == "normal"


# ---------------------------------------------------------------------------
# runner tests
# ---------------------------------------------------------------------------


class TestRunner:

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_process_file(self, file_info, tmp_path):
        labels = Labels(LABELS_DICT)
        encoder = GADFEncoder()
        writer = PILWriter()

        count = process_file(
            file_info=file_info,
            labels=labels,
            encoder=encoder,
            writer=writer,
            output_dir=tmp_path,
            split="train",
            window_size=60,
            resample_freq="1min",
            min_sequence_length=100,
            is_validation=False,
        )

        assert count > 0

        jpgs = list(tmp_path.rglob("*.jpg"))
        assert len(jpgs) > 0

        classes = {p.parent.name for p in jpgs}
        assert "normal" in classes or "anomalous" in classes

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_pipeline_run(self, fixture_dir, tmp_path):
        pipe = Pipeline(
            encoder=GADFEncoder(),
            writer=PILWriter(),
            window_size=60,
            workers=1,
        )

        output = pipe.run(
            raw_paths=[str(fixture_dir)],
            output_dir=str(tmp_path / "output"),
            split="train",
            verbose=False,
        )

        jpgs = list(Path(output).rglob("*.jpg"))
        assert len(jpgs) > 0

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_pipeline_validation_creates_unlabeled(self, fixture_dir, tmp_path):
        pipe = Pipeline(encoder=GADFEncoder(), writer=PILWriter(), window_size=60, workers=1)
        output = pipe.run(
            raw_paths=[str(fixture_dir)],
            output_dir=str(tmp_path / "output"),
            split="validation",
            verbose=False,
        )

        unlabeled = list(Path(output).rglob("unlabeled/*.jpg"))
        assert len(unlabeled) > 0
