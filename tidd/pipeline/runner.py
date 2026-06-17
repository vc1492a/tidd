"""Pipeline orchestrator -- ties discovery, preprocessing, encoding, labeling, and writing together."""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Union

from tqdm import tqdm

from tidd.pipeline.discovery import FileInfo, discover_files
from tidd.pipeline.encoders import Encoder, GADFEncoder
from tidd.pipeline.labels import Labels
from tidd.pipeline.preprocess import read_and_resample, split_events, window_events
from tidd.pipeline.writers import PILWriter, Writer

logger = logging.getLogger(__name__)


def process_file(
    file_info: FileInfo,
    labels: Labels,
    encoder: Encoder,
    writer: Writer,
    output_dir: Path,
    split: str = "train",
    window_size: int = 60,
    resample_freq: str = "1min",
    min_sequence_length: int = 100,
    is_validation: bool = False,
) -> int:
    """
    Process a single raw sTEC file end-to-end: read, preprocess, encode, label, write.

    This is the unit of parallelism -- each file is fully independent.

    :returns: Number of images written.
    """

    try:
        df = read_and_resample(file_info, resample_freq=resample_freq)
        events = split_events(df, min_length=min_sequence_length)
        windows = window_events(events, window_size=window_size)
    except Exception:
        logger.warning("Failed to preprocess %s", file_info.path, exc_info=True)
        return 0

    count = 0
    for win in windows:
        try:
            image = encoder.encode(win.array)
        except Exception:
            logger.warning("Failed to encode window %s/%d", file_info.path, win.window_idx, exc_info=True)
            continue

        label = labels.classify_window(win.day_of_year, win.satellite, win.sod_end)

        pass_id = f"{win.station}__{win.satellite}"
        labeled_dir = output_dir / pass_id / "labeled" / label
        labeled_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{win.day_of_year}_{win.event_idx}_{win.window_idx}_{win.window_idx + window_size}_GAF.jpg"
        writer.save(image, labeled_dir / filename)

        if is_validation:
            unlabeled_dir = output_dir / pass_id / "unlabeled"
            unlabeled_dir.mkdir(parents=True, exist_ok=True)
            writer.save(image, unlabeled_dir / filename)

        count += 1

    return count


class Pipeline:
    """
    Orchestrate the full raw-to-image data pipeline.

    Example::

        from tidd.pipeline import Pipeline, GADFEncoder, PILWriter

        pipe = Pipeline(encoder=GADFEncoder(), writer=PILWriter())
        pipe.run(
            raw_paths=["data/hawaii"],
            labels_paths=["data/hawaii"],
            output_dir="data/experiments/my_experiment",
            split="train",
        )
    """

    def __init__(
        self,
        encoder: Encoder | None = None,
        writer: Writer | None = None,
        window_size: int = 60,
        resample_freq: str = "1min",
        min_sequence_length: int = 100,
        workers: int = -1,
    ):
        self.encoder = encoder or GADFEncoder()
        self.writer = writer or PILWriter()
        self.window_size = window_size
        self.resample_freq = resample_freq
        self.min_sequence_length = min_sequence_length
        self.workers = workers if workers > 0 else max(1, os.cpu_count() - 1)

    def run(
        self,
        raw_paths: list[Union[str, Path]],
        labels_paths: list[Union[str, Path]] | None = None,
        output_dir: Union[str, Path] = "data/experiments",
        split: str = "train",
        verbose: bool = True,
    ) -> Path:
        """
        Discover files, process in parallel, and write labeled images.

        :param raw_paths: List of raw data root directories (e.g. ``["data/hawaii"]``).
        :param labels_paths: Directories containing ``tid_start_finish_times.json``.
            Defaults to *raw_paths*.
        :param output_dir: Root output directory for generated images.
        :param split: Subdirectory name -- ``"train"`` or ``"validation"``.
        :param verbose: Show progress bar.
        :returns: Path to the output directory.
        """

        output_dir = Path(output_dir)
        is_validation = split == "validation"

        if labels_paths is None:
            labels_paths = raw_paths

        all_files: list[tuple[FileInfo, Labels]] = []
        for raw_path, labels_path in zip(raw_paths, labels_paths):
            labels = Labels.from_raw_path(labels_path)
            files = discover_files(raw_path)
            all_files.extend((fi, labels) for fi in files)

        if not all_files:
            logger.warning("No files discovered in %s", raw_paths)
            return output_dir

        logger.info("Discovered %d files to process", len(all_files))

        total_images = 0

        # ProcessPoolExecutor can't pickle Protocol instances, so fall back
        # to sequential when custom encoder/writer are used, or use the
        # simple loop for small workloads.
        if self.workers == 1 or len(all_files) <= 4:
            for fi, labels in tqdm(all_files, disable=not verbose, desc="Processing files"):
                n = process_file(
                    fi, labels, self.encoder, self.writer,
                    output_dir=output_dir / fi.location / split,
                    split=split,
                    window_size=self.window_size,
                    resample_freq=self.resample_freq,
                    min_sequence_length=self.min_sequence_length,
                    is_validation=is_validation,
                )
                total_images += n
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                for fi, labels in all_files:
                    fut = pool.submit(
                        process_file,
                        fi, labels, self.encoder, self.writer,
                        output_dir=output_dir / fi.location / split,
                        split=split,
                        window_size=self.window_size,
                        resample_freq=self.resample_freq,
                        min_sequence_length=self.min_sequence_length,
                        is_validation=is_validation,
                    )
                    futures[fut] = fi

                for fut in tqdm(as_completed(futures), total=len(futures), disable=not verbose, desc="Processing files"):
                    try:
                        total_images += fut.result()
                    except Exception:
                        logger.warning("Failed processing %s", futures[fut].path, exc_info=True)

        logger.info("Pipeline complete: %d images written to %s", total_images, output_dir)
        return output_dir
