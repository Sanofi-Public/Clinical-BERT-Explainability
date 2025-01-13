"""
This module contains classes to handle the loading of data for training a BERT model.
"""

import os
import numpy as np
import pandas as pd
import boto3
import torch
from pyarrow.parquet import ParquetFile
import itertools
import math
import random
import glob
import bisect
from torch.utils.data import IterableDataset

from config.config import PROJECT_DIR
from src.utils.utils import setup_logger, split_s3_file

logger = setup_logger()


class CustomDataset(IterableDataset):
    """
    An object that handles the loading of data
    from parquet files contraining tables formatted like:

    person_id | sorted_event_tokens | day_position_tokens  |
    0 | ["AGE:25", "ETHNICITY:UNK", "GENDER:F", "HCPCS:pr_a", "ICD10CM:dx_a"] | [0, 0, 0, 1, 2] |
    1 | ["AGE:26", "ETHNICITY:UNK", "GENDER:F", "HCPCS:pr_a", "ICD10CM:dx_a"] | [0, 0, 0, 1, 1] |
    ...
    """

    def __init__(
        self,
        data_location,
        tokenizer,
        max_length,
        tmp_data_location=os.path.join(PROJECT_DIR, "TMP_training_data"),
        subsample=None,
        include_person_ids=False,
        shuffle=True,
        code_filter=None,
        percentile_split=None,
        convert_to_ids=True,
    ):
        """
        Args:
            data_location (string):
                A directory containing the parquet files to load data from.
                The data will be loaded from all files matching
                {data_location}/*.parquet.
                This class also supports loading data directly to s3.
                The parquet file is assumed to have a table of the following form:
                person_id (int), sorted_event_tokens (array<string>),
                day_position_tokens (array<int>).
            tokenizer (tokenizers.models.Model):
                The tokenizer model to handle tokenization
            max_length (int):
                The sequence length to pad all outputs to. e.g.
                the max length for BERT is 512, so set it accordingly.
            tmp_data_location (string):
                If the data_location is on s3, and not local,
                then downlaod the data from s3 to this directory.
            subsample (float < 1.0):
                The fraction of the dataset to subsample
            include_person_ids:
                Whether to include person ids in the loaded data.
            shuffle:
                Whether to shuffle the dataframes before returning.
            code_filter:
                A comma-separated list of codes to use during training.
            percentile_split:
                The number of percentiles expected in the training data.
            convert_to_ids:
                Whether or not to apply the tokenizer to the output. i.e. to convert to input_ids
        """

        self.data_location = data_location
        assert self.data_location.endswith("/")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tmp_data_location = tmp_data_location
        self.subsample = subsample
        if not os.path.exists(tmp_data_location):
            os.makedirs(tmp_data_location)
        self.files = sorted([self.download_file_if_needed(f) for f in self.get_files()])
        self.include_person_ids = include_person_ids
        self.shuffle = shuffle
        self.code_filter = code_filter
        if self.code_filter is not None:
            self.codes_matched = self.code_filter.split(",")
            logger.info(f"Only using codes containing {self.codes_matched}")
        else:
            self.codes_matched = None

        self.percentile_split = percentile_split
        self.convert_to_ids = convert_to_ids
        super().__init__()

    def handle_multi_processing(self, files):
        """
        Determine which files this worker is responsible for.
        If running w/o multiprocessing, then just read all files.
        If running multiprocessing then check which worker
        this is using torch.utils.data.get_worker_info().
        Make sure not to read the same files as other workers
        to avoid duplicates. Check the documentation here:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(files)
        if worker_info is not None:  # in a worker process
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            logger.debug(
                f"Worker {worker_info.id} handling files {files[iter_start:iter_end]}"
            )
        else:
            iter_start = start
            iter_end = end
        return files[iter_start:iter_end]

    def get_files(self):
        """
        Return a list of all data parquet files found in the directory self.data_location
        """
        if self.data_location.startswith("s3://"):
            bucket, prefix = split_s3_file(self.data_location)
            s3 = boto3.resource("s3")
            my_bucket = s3.Bucket(bucket)
            files = [
                f"s3://{bucket}/{my_bucket_object.key}"
                for my_bucket_object in my_bucket.objects.filter(Prefix=prefix)
                if my_bucket_object.key.endswith(".parquet")
            ]
            logger.debug(f"Found files {files}")
        else:
            files = glob.glob(os.path.join(self.data_location, "*.parquet"))
        files = sorted(
            files
        )  # sort them to make sure each worker has the same files when handling multiprocessing
        return files

    def download_file_if_needed(self, data_location):
        """
        If file is not already downloaded, download it from S3.
        """
        if data_location.startswith("s3://"):
            local_fname = os.path.join(
                self.tmp_data_location, os.path.split(data_location)[-1]
            )
            if not os.path.exists(local_fname):
                logger.debug(f"downloading file {data_location}")
                bucket, key = split_s3_file(data_location)
                boto3.client("s3").download_file(bucket, key, local_fname)
            else:
                logger.debug(
                    f"Reading local copy of this data file {data_location} at {local_fname}"
                )
        else:  # this is local data file. Don't download it. Just read it.
            local_fname = data_location
        return local_fname

    def iterate_files(self):
        for f in self.handle_multi_processing(self.files):
            yield f

    def get_data_length(self):
        total = 0
        for local_fname in self.iterate_files():
            if self.subsample is None:
                total += ParquetFile(local_fname).metadata.num_rows
            else:
                total += int(
                    ParquetFile(local_fname).metadata.num_rows * self.subsample
                )
        return total

    def __len__(self):
        return self.get_data_length()

    def iterate_dataframes(self):
        for local_fname in self.iterate_files():
            df = pd.read_parquet(local_fname)
            if self.shuffle:
                df = df.sample(frac=1.0)
            elif self.subsample is not None:
                df = df.sample(frac=self.subsample)
            logger.debug(
                f"The dataframe after reading the file was this long {len(df)}"
            )
            if len(df) == 0:
                continue
            yield df

    def process_data(self, row):
        inputs = row["sorted_event_tokens"]
        position_ids = row["day_position_tokens"]

        # apply the filters, if they are defined
        if self.code_filter is not None:
            selection = [
                any(code in input for code in self.codes_matched) for input in inputs
            ]
            inputs = [input for i, input in enumerate(inputs) if selection[i]]
            position_ids = [
                position_id
                for i, position_id in enumerate(position_ids)
                if selection[i]
            ]

        if inputs[0] != "[CLS]":
            inputs = itertools.chain(["[CLS]"], inputs)  # add the cls token and encode
            inputs = list(inputs)
            position_ids = [0] + list(position_ids)  # account for the cls token

        if self.percentile_split is not None:
            percentiles = [
                el.split(":")[-1] if el.count(":") > 1 else None for el in inputs
            ]
            inputs = [":".join(el.split(":")[0:2]) for el in inputs]
            assert all(
                int(el.split("-")[0]) % self.percentile_split == 0
                for el in percentiles
                if el is not None
            )
            encoded_percentiles = [
                int(el.split("-")[1]) // self.percentile_split if el is not None else 0
                for el in percentiles
            ]
            encoded_percentiles = np.array(encoded_percentiles)
            assert len(encoded_percentiles) == len(inputs)

        to_return = {}
        if self.convert_to_ids:
            encoded_inputs = np.array(
                [self.tokenizer.tokenize(el)[0].id for el in inputs]
            )
            encoded_inputs = encoded_inputs[
                0 : min(self.max_length, len(encoded_inputs))
            ]
            to_return["input_ids"] = encoded_inputs
        else:
            encoded_inputs = inputs
            encoded_inputs = encoded_inputs[
                0 : min(self.max_length, len(encoded_inputs))
            ]
            to_return["inputs"] = encoded_inputs

        position_ids = position_ids[0 : min(self.max_length, len(encoded_inputs))]
        to_return["attention_mask"] = np.ones(len(encoded_inputs), np.int8)
        to_return["position_ids"] = position_ids

        if self.percentile_split is not None:
            to_return["token_type_ids"] = encoded_percentiles[
                0 : len(encoded_inputs)
            ]  # pad them to the max length

        if self.include_person_ids:
            to_return["person_id"] = [row["person_id"]]

        return to_return

    def __iter__(self):
        """
        Iterate through the entire dataset, returning entries one-by-one
        """

        for df in self.iterate_dataframes():
            for _, row in df.iterrows():
                to_return = self.process_data(row)

                yield to_return


class LabelledDataset(CustomDataset):
    """
    A class to iterate through a dataset of labelled dataset for training a BERT model.
    """

    def __init__(self, labelled_data_column_name, *args, **kwargs):
        self.labelled_data_column = labelled_data_column_name
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """
        Iterate through the entire dataset, returning entries one-by-one
        """
        for df in self.iterate_dataframes():
            for _, row in df.iterrows():
                to_return = self.process_data(row)
                to_return["labels"] = np.array([row[self.labelled_data_column]])
                yield to_return


class ProportionalDataset(IterableDataset):
    """
    This class is meant to load two datasets at fixed proportion.
    i.e. if proportion_1 is 1 and proportion_2 is 10, then 10 data
    point from dataset 2 will be returned for every one data point from dataset_1
    """

    def __init__(self, datasets, proportions):
        assert len(datasets) == len(proportions)
        self.datasets = datasets
        total_proportions = sum(proportions)
        self.proportions = proportions
        self.samplings = [proportion / total_proportions for proportion in proportions]
        self.iterators = [iter(data) for data in self.datasets]
        self.cumulative_sum = np.cumsum(self.samplings)

    def __len__(self):
        return int(
            min(len(data) for data in self.datasets) * sum(self.proportions)
        )  # This is how many iterations are needed to see the all samples of the smallest class.

    def __iter__(self):
        for _ in range(0, len(self)):
            index = bisect.bisect_right(self.cumulative_sum, random.uniform(0.0, 1.0))
            try:
                yield next(self.iterators[index])
            except StopIteration:
                self.iterators[index] = iter(self.datasets[index])
                yield next(self.iterators[index])


def get_vocab_from_file(vocab_file):
    """
    Reads a vocabulary file and returns a dictionary mapping each token to its index.

    Args:
        vocab_file (str): Path to the vocabulary file.

    Returns:
        dict: A dictionary where keys are tokens (str) and values are their indices (int).
    """
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = {el.strip(): i for i, el in enumerate(f.readlines())}
    return vocab
