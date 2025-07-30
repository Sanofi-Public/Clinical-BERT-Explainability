"""
This script generates word attributions for each class for the BERT classification model.
"""

import os
import warnings
import argparse
import re
from typing import Optional, Dict
import tarfile
import numpy as np
import pandas as pd
import torch
from transformers import (
    PreTrainedTokenizer,
    BertForSequenceClassification,
)
from torch.nn.modules.sparse import Embedding

from tqdm import tqdm
import yaml

from src.model_training.train_utils import LabelledDataset, get_vocab_from_file
from src.utils.utils import setup_logger

from transformers_interpret.explainers.text.sequence_classification import (
    SequenceClassificationExplainer,
)
from transformers_interpret.attributions import LIGAttributions

import sys
sys.path.append(os.path.abspath('./transformers_interpret'))

logger = setup_logger()


class FixedVocabTokenizer(PreTrainedTokenizer):
    """
    A tokenizer that uses a fixed vocabulary, NOT designed for natuaral language input.
    The tokenizer expects the input to be a concatenation of the following, separated by |.
        1. words/tokens
        2. position_ids
        3. (optional) token_type_ids,
    Example: "word1, word2, word3, word4 | 0, 1, 2, 3 | 0, 0, 1, 1"

    This tokenizer only adds a classification token at the beginning of the input,
    and truncates if desired.
    Code adapted from https://stackoverflow.com/questions/69531811/using-hugginface-transformers-and-tokenizers-with-a-fixed-vocabulary
    """

    def __init__(self, vocab: Dict[str, int], max_len: int = None):
        self._token_ids = vocab
        self._id_tokens: Dict[int, str] = {
            value: key for key, value in self._token_ids.items()
        }

        super().__init__(max_len=max_len)

        # Initialize special tokens
        max_id = len(vocab.values())
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.unk_token_id = self._token_ids.get(self.unk_token, max_id + 1)
        self.pad_token_id = self._token_ids.get(self.pad_token, max_id + 2)
        self.cls_token_id = self._token_ids.get(self.cls_token, max_id + 3)
        self.sep_token_id = self._token_ids.get(self.sep_token, max_id + 4)
        self.mask_token_id = self._token_ids.get(self.mask_token, max_id + 5)

    def _tokenize(self, text: str, **kwargs):
        # Remove spaces comma (,) colon (:) period (.) hyphen (-) and slash (/) except within words
        # (This is done because the transformers-interpret package
        # automatically adds spaces between tokens)
        text = re.sub(r"\s*([,:.\-/])\s*", r"\1", text)
        tokens = text.split(",")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        # Check in added tokens first, then in the original vocabulary
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._token_ids[token] if token in self._token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        # Check in added tokens first, then in the original vocabulary
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index]
        return self._id_tokens[index] if index in self._id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        # Combine the original vocabulary with the added tokens
        combined_vocab = self._token_ids.copy()
        combined_vocab.update(self.added_tokens_encoder)
        return combined_vocab

    @property
    def vocab_size(self) -> int:
        return len(self._token_ids)

    def __call__(
        self,
        texts: str,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ):
        texts = re.sub(r"\s*([:,])\s*", r"\1", texts)
        split_texts = texts.split("|")

        if max_length is None:
            max_length = self.model_max_length
        tokenized_texts = self._tokenize(split_texts[0])

        if not len(split_texts) in [2, 3]:
            raise ValueError(
                "The position ids should be included! Mark them with the | character."
            )

        input_ids = tokenized_texts
        position_ids = [int(el) for el in split_texts[1].split(",")]
        attention_mask = np.ones(len(input_ids), np.int8)

        if len(input_ids) != len(position_ids):
            raise ValueError(
                "You are missing input_ids or position_ids! They should be equal in number."
            )

        if len(split_texts) == 3:
            token_type_ids = [int(el) for el in split_texts[2].split(",")]
            if len(input_ids) != len(token_type_ids):
                raise ValueError(
                    "You are missing input_ids or token_type_ids! They should be equal in number."
                )
        else:
            token_type_ids = None

        if len(input_ids) > max_length and truncation is True:
            input_ids = input_ids[:max_length]
            position_ids = position_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:max_length]

        return {
            "input_ids": [self._convert_token_to_id(token) for token in input_ids],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "inputs": tokenized_texts,
            "token_type_ids": token_type_ids,
        }


class EHRSequenceClassificationExplainer(SequenceClassificationExplainer):
    """
    The default SequenceClassificationExplainer class assumes the position_ids to be sequential;
    this modificaiton instead uses a custom list of position_ids as specified by the input.

    This Explainer assumes the reference input and position embedding to be all-zero. EXCEPT demographic categorical tokens where every input has one at a fixed location.
    Example: ['AGE:43', 'ETHNICITY:Not Hispanic', 'GENDER:Female', 'RACE:Caucasian', 'REGION:Midwest', 'ICD10:1234', 'ICD10:56678', ...] <- every input will start with age, ehtnicity, gender, race, and region; people with unknown demographic information will have an unknown token.
    In this case, the reference input embedding is the average of all tokens that falls in that category. e.g. if GENDER has three tokens {GENDER:Male, GENDER: Female, GENDER: Unknown}, the reference input embedding is the average of the three.
    It is assumed that all tokens in the same category would have the same starters.
    """

    def __init__(
        self, model, tokenizer, categories_to_avg=None, middle_percentile=None
    ):
        super().__init__(model, tokenizer)
        if categories_to_avg is None:
            categories_to_avg = []
        self.categories_to_avg = categories_to_avg
        self.middle_percentile = middle_percentile

    def _calculate_attributions(
        self, embeddings: Embedding, index: int = None, class_name: str = None
    ):
        """
        A modification to the original function that changes how the input_ids, position_ids, and attention_mask are inputted.
        The actual attribution calculation aspect is the same.
        """
        tokenizer_output = self.tokenizer(self.text)
        input_ids, position_ids, attention_mask, inputs, token_type_ids = (
            tokenizer_output["input_ids"],
            tokenizer_output["position_ids"],
            tokenizer_output["attention_mask"],
            tokenizer_output["inputs"],
            tokenizer_output["token_type_ids"],
        )

        self.has_token_types = token_type_ids is not None
        self.has_position_ids = position_ids is not None

        self.input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        self.sep_idx = len(input_ids)
        ref_input_ids = [self.ref_token_id] * len(input_ids)

        categories_to_avg = self.categories_to_avg

        for category in categories_to_avg:
            avg_category_token_id = None
            avg_category_token = f"[AVG_{category}]"

            if avg_category_token not in self.tokenizer.get_vocab():
                logger.info(
                    f"{avg_category_token} is not found in the vocabulary; creating one"
                )

                # Get all tokens starting with category
                category_token_ids = [
                    token_id
                    for token, token_id in self.tokenizer.get_vocab().items()
                    if token.startswith(category)
                ]
                logger.info(
                    f"Found the following tokens for category {category}: {[self.tokenizer._convert_id_to_token(i) for i in category_token_ids]}"
                )

                # Compute the average embedding for the "avg_category" token
                token_embeddings = self.model.bert.embeddings.word_embeddings.weight
                category_embeddings = token_embeddings[category_token_ids]
                avg_category_embedding = torch.mean(category_embeddings, dim=0)

                # Expand the model's embeddings to accommodate the new token
                new_token_embeddings = torch.cat(
                    [token_embeddings, avg_category_embedding.unsqueeze(0)], dim=0
                )
                self.model.bert.embeddings.word_embeddings.weight = torch.nn.Parameter(
                    new_token_embeddings
                )

                logger.info(f"Existing token embedding shape: {token_embeddings.shape}")
                logger.info(f"Category embeddings shape: {category_embeddings.shape}")
                logger.info(
                    f"Average category embedding shape: {avg_category_embedding.shape}"
                )
                logger.info(f"New token embeddings shape: {new_token_embeddings.shape}")

                avg_category_token_id = new_token_embeddings.size(0) - 1
                self.tokenizer._token_ids[avg_category_token] = avg_category_token_id
                self.tokenizer._id_tokens[avg_category_token_id] = avg_category_token

                logger.info(f"Added {avg_category_token}: {avg_category_token_id}")

            else:
                avg_category_token_id = self.tokenizer._token_ids.get(
                    avg_category_token
                )

            if avg_category_token_id is None:
                raise ValueError(
                    f"The avg_category_token_id for {category} is not found."
                )

            for id, token in enumerate(inputs):
                if token.startswith(category):
                    ref_input_ids[id] = avg_category_token_id

        self.ref_input_ids = torch.tensor(ref_input_ids, device=self.device).unsqueeze(
            0
        )

        self.position_ids = torch.tensor(
            position_ids, dtype=torch.int, device=self.device
        ).unsqueeze(0)

        position_embeddings = self.model.bert.embeddings.position_embeddings.weight
        zero_row = torch.zeros(
            (1, position_embeddings.size(1)), device=position_embeddings.device
        )
        new_position_embeddings = torch.cat([position_embeddings, zero_row], dim=0)
        self.model.bert.embeddings.position_embeddings.weight = torch.nn.Parameter(
            new_position_embeddings
        )

        new_position_id = new_position_embeddings.size(0) - 1
        self.ref_position_ids = torch.full_like(self.input_ids, new_position_id)

        if token_type_ids is None:
            self.token_type_ids, self.ref_token_type_ids = None, None
        else:
            self.token_type_ids = torch.tensor(
                token_type_ids, device=self.device
            ).unsqueeze(0)
            self.ref_token_type_ids = torch.zeros(
                size=self.input_ids.size(),
                dtype=torch.int,
                device=self.device,
            )
            if self.middle_percentile is None:
                raise ValueError(
                    """This data has token type ids but the explainer does not know the middle percentile for reference!
                Please pass the middle_percentile keyword argument when initializing this class."""
                )
            self.ref_token_type_ids[
                self.token_type_ids > 0.5
            ] = (
                self.middle_percentile  # set these to be the average test result percentile for reference.
            )

        self.attention_mask = torch.tensor(
            attention_mask, dtype=torch.int, device=self.device
        ).unsqueeze(0)

        # The rest of the function below is adapted from transformers-interpret under the Apache License 2.0.
        # Original source: https://github.com/cdpierse/transformers-interpret
        # See the Apache License 2.0 at http://www.apache.org/licenses/LICENSE-2.0 for details.

        if index is not None:
            self.selected_index = index
        elif class_name is not None:
            if class_name in self.label2id.keys():
                self.selected_index = int(self.label2id[class_name])
            else:
                s = f"'{class_name}' is not found in self.label2id keys."
                s += "Defaulting to predicted index instead."
                warnings.warn(s)
                self.selected_index = int(self.predicted_class_index)
        else:
            self.selected_index = int(self.predicted_class_index)

        reference_tokens = [
            token.replace("Ġ", "") for token in self.decode(self.input_ids)
        ]

        lig = LIGAttributions(
            custom_forward=self._forward,
            embeddings=embeddings,
            tokens=reference_tokens,
            input_ids=self.input_ids,
            ref_input_ids=self.ref_input_ids,
            sep_id=self.sep_idx,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            ref_position_ids=self.ref_position_ids,
            token_type_ids=self.token_type_ids,
            ref_token_type_ids=self.ref_token_type_ids,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )

        lig.summarize()
        self.attributions = lig

    def _forward(self, *args):  # type: ignore
        self.accepts_token_type_ids = True
        self.accepts_position_ids = True

        input_ids = args[0]

        if self.has_position_ids and self.has_token_types:
            token_type_ids = args[1]
            position_ids = args[2]
        elif self.has_position_ids and not self.has_token_types:
            position_ids = args[1]
            token_type_ids = None
        elif not self.has_position_ids and self.has_token_types:
            token_type_ids = args[1]
            position_ids = None
        else:
            token_type_ids = None
            position_ids = None

        preds = self._get_preds(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        preds = preds[0]

        # if it is a single output node
        if len(preds[0]) == 1:
            self._single_node_output = True
            self.pred_probs = torch.sigmoid(preds)[0][0]
            return torch.sigmoid(preds)[:, :]

        self.pred_probs = torch.softmax(preds, dim=1)[0][self.selected_index]
        to_return = torch.softmax(preds, dim=1)[:, self.selected_index]

        return to_return


def generate_explanation(
    item, model_max_len, num_labels, internal_batch_size, cls_explainer
):
    """
    Given an input and the explainer, output the word attribution list for each class.
    """
    words_df = pd.DataFrame(item["inputs"], columns=["word"])

    inputs_str = ",".join(map(str, item["inputs"]))
    position_ids_str = ",".join(map(str, item["position_ids"]))
    if "token_type_ids" in item:
        token_type_ids_str = ",".join(map(str, item["token_type_ids"]))
    else:
        token_type_ids_str = ""

    # Concatenate into a single string
    concatenated_string = inputs_str + "|" + position_ids_str + "|" + token_type_ids_str
    concatenated_string = concatenated_string.rstrip("|")

    for i in range(num_labels):
        word_attributions = cls_explainer(
            concatenated_string,
            internal_batch_size=internal_batch_size,
            class_name=f"LABEL_{i}",
        )
        # add the second element of every tuple as a column to words_df, the column name is "score_{i}"
        words_df[f"score_{i}"] = [
            x[1] for x in word_attributions
        ]  # excluding the first item which is [CLS]

        # add a new column named "pred_prob_{i}" that is the predicted probability of the class and it is the same value for every row
        words_df[f"pred_prob_{i}"] = cls_explainer.pred_probs.item()

    words_df["person_id"] = item["person_id"][0]
    words_df["position_id"] = item["position_ids"]
    if "token_type_ids" in item:
        words_df["token_type_id"] = item["token_type_ids"]

    words_df["true_label"] = item["labels"][0]

    return words_df


def run_explanation_generation(
    config_path, model_location, input_dataset_path, output_path
):
    """
    A modification of the main function so that it can be run in Jupyter Notebook
    The input path, model, and output are all locally sourced/outputted.
    """

    with open(config_path, "r") as f:
        yaml_args = yaml.safe_load(f)

    with tarfile.open(model_location, "r") as tar:
        tar.extractall()
        folder_name = os.path.commonprefix(tar.getnames())

    model_max_len = yaml_args.get("model_max_len", 512)
    model_path = os.path.join(".", folder_name)

    model = BertForSequenceClassification.from_pretrained(model_path)
    num_labels = model.config.num_labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_config = model.config
    logger.info(f"Number of token type IDs: {model_config.type_vocab_size}")
    percentile_split = None
    middle_percentile = None

    # type_vocab_size is used to represent the number of ways the splitting/binning is done to categorize lab tokens.
    # For example, if token_vocab_size = 11, the range of values are split into 10 bins (decile split from 1 to 10; with 0 representing non-lab tokens), 
    # and the middle percentile is 5.
    if model_config.type_vocab_size > 2:
        percentile_split = model_config.type_vocab_size - 1 
        middle_percentile = percentile_split // 2

    vocab = get_vocab_from_file(os.path.join(model_path, "vocab.txt"))
    tokenizer = FixedVocabTokenizer(vocab, max_len=model_max_len)

    cls_explainer = EHRSequenceClassificationExplainer(
        model,
        tokenizer,
        categories_to_avg=yaml_args.get("demographic_token_starters", []),
        middle_percentile=middle_percentile,
    )

    iterable_test_dataset = LabelledDataset(
        data_location=input_dataset_path,
        tokenizer=tokenizer,
        max_length=model_max_len,
        tmp_data_location="./TMP_data/",
        include_person_ids=True,
        labelled_data_column_name="label",
        percentile_split=percentile_split,
        # percentile_split=None,
        convert_to_ids=False,
    )

    columns = ["person_id", "position_id", "true_label"]
    for i in range(num_labels):
        columns.append(f"score_{i}")
        columns.append(f"pred_prob_{i}")
    word_attribution_df = pd.DataFrame(columns=columns)

    for item in tqdm(iterable_test_dataset, desc="Processing items"):
        words_df = generate_explanation(
            item,
            model_max_len=model_max_len,
            num_labels=num_labels,
            internal_batch_size=1,
            cls_explainer=cls_explainer,
        )
        word_attribution_df = pd.concat(
            [word_attribution_df, words_df], ignore_index=True
        )

    word_attribution_df.to_parquet(
        os.path.join(output_path, "output.parquet"), index=False
    )


def main():
    """
    Main function to run explanation generation.
    """

    parser = argparse.ArgumentParser(description="Run explanation generation.")

    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument("model_location", type=str, help="Location of the model")
    parser.add_argument(
        "input_dataset_path", type=str, help="Path to the input dataset"
    )
    parser.add_argument("output_path", type=str, help="Path to save the output")

    args = parser.parse_args()

    run_explanation_generation(
        args.config_path, args.model_location, args.input_dataset_path, args.output_path
    )


if __name__ == "__main__":
    main()
