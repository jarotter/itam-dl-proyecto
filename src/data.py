"""Preprocessing steps. tightly coupled to the Flickr8K dataset."""

import pickle
from configparser import ConfigParser
from typing import List, Tuple

import numpy as numpy
import pandas as pd
import tensorflow as tf
from google.cloud import storage

from transformers import RobertaTokenizer


class FileDownloader:
    """Downloads from Google Cloud Storage."""

    BUCKET = "tti-roberta-wcgan"

    def __init__(self, bucket: str = BUCKET):
        self.bucket_id = bucket
        self._init_bucket()

    def _init_bucket(self):
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_id)
        assert bucket is not None
        self.bucket = bucket

    def download_blob(self, path: str):
        return self.bucket.get_blob(path)

    def download_as_string(self, path: str):
        blob = self.download_blob(path)
        return blob.download_as_string().decode()


class Flickr8KFileLocations:
    """Location of the split train/eval/test datasets."""

    CFG_FILE = "cfg/flickr8.ini"

    def __init__(self, cfg_file: str = CFG_FILE):
        self.cfg_file = cfg_file
        self.downloader = FileDownloader()
        self._read_from_yaml()

    def _read_from_yaml(self):
        config = ConfigParser()
        cfg_file = self.downloader.download_as_string(self.cfg_file)
        config.read_string(cfg_file)
        self.train = config["file locations"]["train"]
        self.test = config["file locations"]["test"]
        self.eval = config["file locations"]["eval"]


class Flickr8K:
    """Base for both image and text datasets."""

    TOKENS_ID = "gs://tti-roberta-wcgan/data/text/Flickr8k.token.txt"

    def __init__(self, df: pd.DataFrame = None):
        self.file_locations = Flickr8KFileLocations()
        self.df = self._read_dataframe() if df is None else df
        self.downloader = FileDownloader()
        self._correct_image_name()
        self._train_test_split()
        self._assign_data()

    def _read_dataframe(self, path: str = TOKENS_ID) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t", names=["image_id", "text"])

    def _correct_image_name(self) -> str:
        splitter = lambda s: s.split("#")[0]
        self.df["image_id"] = self.df.image_id.apply(splitter)

    def _read_descriptions(self, dataset: str) -> List[str]:
        fpath = getattr(self.file_locations, dataset)
        texts_str = self.downloader.download_as_string(fpath)
        texts = [x.strip() for x in texts_str.split("\n")]
        return pd.Series(texts)

    def _train_test_split(self):
        assignments = pd.DataFrame()
        for dataset in ["train", "test", "eval"]:
            image_id = self._read_descriptions(dataset)
            df = pd.DataFrame({"image_id": image_id, "split": dataset})
            assignments = pd.concat((assignments, df))
        self.df = self.df.merge(assignments, how="outer")

    def _assign_data(self):
        for dataset in ["train", "test", "eval"]:
            values = self.df[self.df.split == dataset].text.to_list()
            setattr(self, dataset, values)


class RoBERTaTokenizedFlickr8K(Flickr8K):
    """Tokenized for RoBERTa."""

    PICKLE_LOCATION = "../data/text/roberta_tokenized.pickle"

    def __init__(
        self,
        roberta_model_name: str = "distilroberta-base",
        max_sentence_length: int = 25,
        batch_size: int = 36,
        data_object: Flickr8K = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.roberta_model_name = roberta_model_name
        self.max_sentence_length = max_sentence_length
        self._tokenize()
        self.data = data_object if data_object is not None else Flickr8K()

    def _tokenize(self):
        tokenizer = RobertaTokenizer.from_pretrained(self.roberta_model_name)
        for split in ["train", "test", "eval"]:
            values = getattr(self, split)
            tokenized = tokenizer.batch_encode_plus(
                values,
                pad_to_max_length=True,
                max_length=self.max_sentence_length,
                return_tensors="tf",
                return_attention_masks=False,
            )
            tokenized = tf.data.Dataset.from_tensor_slices(tokenized["input_ids"])
            tokenized = tokenized.batch(self.batch_size)
            setattr(self, split, tokenized)

    def save(self):
        with open(self.PICKLE_LOCATION, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str = PICKLE_LOCATION):
        with open(path, "rb") as f:
            return pickle.load(f)


class Flickr8KImages:
    """Stream from GCS."""

    FILE_LOCATION = "gs://tti-roberta-wcgan/data/img/"

    def __init__(
        self,
        data_object: Flickr8K = None,
        desired_image_shape: Tuple[int, int] = (64, 64),
        batch_size: int = 36,
    ):
        self.image_shape = desired_image_shape
        self.batch_size = batch_size
        self.data = Flickr8K() if data_object is None else data_object
        self._assign_data()

    def _assign_data(self):
        for split in ["train", "test", "eval"]:
            values = self.data.df[self.data.df.split == split].image_id.to_list()
            dataset = tf.data.Dataset.from_tensor_slices(values)
            dataset = dataset.map(self._process_name)
            dataset = dataset.batch(self.batch_size)
            setattr(self, split, dataset)

    def _process_name(self, image_name: str):
        filepath = tf.strings.join((self.FILE_LOCATION, image_name))
        return self._decode_image(filepath)

    def _decode_image(self, filepath: str):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, self.image_shape)
