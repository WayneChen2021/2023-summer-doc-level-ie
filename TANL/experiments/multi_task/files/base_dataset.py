# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import json
import logging
import random
from typing import Dict, Generator, Tuple, List
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, torch_distributed_zero_first, default_data_collator
import gzip

from arguments import DataTrainingArguments
from input_example import InputFeatures, InputExample
from input_formats import INPUT_FORMATS
from output_formats import OUTPUT_FORMATS

import numpy as np


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None         # name of the dataset
    data_name = None    # name of the directory, if different from the name of the dataset
    # string to prepend to every input sentence if multitask=True (default is self.name)
    task_descriptor = None

    default_input_format = 'plain'
    default_output_format = None
    default_data_dir = 'data'

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_input_length: int,
            max_output_length: int,
            overwrite_cache: bool = False,
            mode: str = 'train',
            local_rank: int = -1,
            # a number < 1 is to use only a subset of training data (random)
            train_subset: float = 1,
            seed: int = None,
            shuffle: bool = True,
            data_args: DataTrainingArguments = None,
            is_eval: bool = False,
            same_input_output_trigs = False
    ):
        if seed is not None:
            # set random seed for repeatability
            random.seed(seed)

        self.data_args = data_args
        self.tokenizer = tokenizer

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.input_format_str = data_args.input_format if data_args.input_format is not None else self.default_input_format
        self.output_format_str = data_args.output_format if data_args.output_format is not None else self.default_output_format
        self.input_format = INPUT_FORMATS[self.input_format_str]()
        self.output_format = OUTPUT_FORMATS[self.output_format_str]()

        self.data_path = data_args.data_dir if data_args.data_dir is not None else self.default_data_dir

        self.is_eval = is_eval
        self.eval_nll = data_args.eval_nll

        cached_data_file = os.path.join(
            self.data_dir(),
            f"cached_{self.name}_{mode}_{tokenizer.__class__.__name__}_{max_input_length}_{max_output_length}"
            f"{'_multitask' if data_args.multitask else ''}.pth"
        )

        with torch_distributed_zero_first(local_rank):
            # make sure only the first process in distributed training processes the dataset,
            # and the others can use the cached version

            if os.path.exists(cached_data_file) and not overwrite_cache:
                self.load_cached_data(cached_data_file)

            else:
                self.load_schema()   # here the dataset can load information such as entity/relation types
                self.examples = self.load_data(mode=mode, seed=seed)

                # assign examples to this dataset
                for example in self.examples:
                    example.dataset = self

                self.features = self.compute_features(
                    max_input_length=max_input_length,
                    max_output_length=max_output_length,
                    multitask=data_args.multitask,
                )

                if local_rank in [-1, 0]:
                    # save data
                    self.save_data(cached_data_file)

            # shuffle indices
            self.indices = list(range(len(self.examples)))
            if seed is not None and shuffle:
                random.shuffle(self.indices)

            # compute effective size of the dataset
            self.effective_size = round(train_subset * len(self.examples))
            if train_subset != 1:
                logging.info(
                    f"Effective dataset size reduced to {self.effective_size} ({train_subset * 100:.0f}%)")

    def __repr__(self):
        return f'Dataset {self.name}'

    def __len__(self):
        return self.effective_size

    def __getitem__(self, i: int) -> InputFeatures:
        return self.features[self.indices[i]]

    def get_example(self, i: int) -> InputExample:
        return self.examples[self.indices[i]]

    def data_dir(self):
        suffix = ""
        if os.getenv("DataDirOverrideSuffix"):
            suffix = os.getenv("DataDirOverrideSuffix")

        if self.data_name is not None:
            return os.path.join(self.data_path, self.data_name + suffix)
        else:
            return os.path.join(self.data_path, self.name + suffix)

    def load_cached_data(self, cached_data_file: str):
        d = torch.load(cached_data_file)
        self.examples, self.features = d['examples'], d['features']

    def save_data(self, cached_data_file: str):
        torch.save({
            'examples': self.examples,
            'features': self.features,
        }, cached_data_file)

    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def load_data(self, mode: str, seed: int = None) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode

        for split in splits:
            examples += self.load_data_single_split(split, seed=seed)

        return examples

    def _warn_max_sequence_length(self, max_sequence_length: int, sentences: List[str], name: str):
        max_length_needed = max(len(self.tokenizer.tokenize(x))
                                for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f'Max sequence length is {max_sequence_length} but the longest {name} sequence is '
                f'{max_length_needed} long'
            )

    def compute_features(self, max_input_length: int, max_output_length: int, multitask: bool = False):
        input_sentences = [self.input_format.format_input(
            example, multitask=multitask) for example in self.examples]
        output_sentences = [self.output_format.format_output(
            example) for example in self.examples]
        print()
        print(input_sentences[:10])
        print()
        print(output_sentences[:10])
        l = []
        input_tok = self.tokenizer.batch_encode_plus(
            input_sentences,
            max_length=max_input_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        for enc in input_tok._encodings:
            l.append(enc.tokens)
        self._warn_max_sequence_length(
            max_input_length, input_sentences, "input")

        output_tok = self.tokenizer.batch_encode_plus(
            output_sentences,
            max_length=max_output_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(
            max_output_length, output_sentences, "output")

        def uniquify(path):
            filename, extension = os.path.splitext(path)
            counter = 1
            while os.path.exists(path):
                path = filename + "_" + str(counter) + extension
                counter += 1
            return path

        json.dump(l, open(uniquify("./tokens.json"), "w+"))
        assert len(input_sentences) == len(output_sentences)
        assert input_tok.input_ids.size(0) == output_tok.input_ids.size(0)

        features = []
        for sentence_input_ids, att_mask, label_input_ids in zip(input_tok.input_ids, input_tok.attention_mask,
                                                                 output_tok.input_ids):
            label_input_ids_list = label_input_ids.tolist()
            left, right = 0, 0
            is_in_bracket = False
            left_bracket_id, right_bracket_id = 784, 908
            left_curly_id, right_curly_id = 32100, 32101
            argument_mask = [-100] * len(label_input_ids_list)
            argument_mask_bool = [False] * len(label_input_ids_list)
            types_mask = [-100] * len(label_input_ids_list)
            types_mask_bool = [False] * len(label_input_ids_list)
            
            for i, tok_id in enumerate(label_input_ids_list):
                left += int(tok_id == left_bracket_id)
                if left > right:
                    argument_mask_bool[i] = True
                    argument_mask[i] = tok_id
                right += int(tok_id == right_bracket_id)

                if tok_id == left_curly_id:
                    is_in_bracket = True
                if is_in_bracket:
                    types_mask[i] = tok_id
                    types_mask_bool[i] = True
                if tok_id == right_curly_id:
                    is_in_bracket = False

            features.append(InputFeatures(
                input_ids=sentence_input_ids.tolist(),
                attention_mask=att_mask.tolist(),
                label_ids=label_input_ids_list,
                non_args_masked=argument_mask,
                types_mask=types_mask,
                non_args_masked_bool=argument_mask_bool,
                types_mask_bool=types_mask_bool,
                left_brackets=[left_bracket_id if not b else -100 for b in argument_mask],
                right_brackets=[right_bracket_id if not b else -100 for b in argument_mask]
            ))

        return features

    def generate_output_sentences(self, data_args: DataTrainingArguments, model, device, batch_size: int) \
            -> Generator[Tuple[InputExample, str], None, None]:
        """
        Generate pairs (example, output_sentence) for evaluation.
        """
        test_data_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        for i, inputs in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            predictions = model.generate(
                inputs['input_ids'].to(device),
                max_length=data_args.max_output_seq_length_eval,
                num_beams=data_args.num_beams if data_args.num_beams else 1,
                output_hidden_states=True
            )

            def tuple_to_numpy(x):
                return tuple(t.cpu().numpy() if not isinstance(t, tuple) else tuple_to_numpy(t) for t in x)

            def np_compress_save(file_name, data):
                f = gzip.GzipFile(f"{file_name}.npy.gz", "w")
                np.save(file=f, arr=data)
                f.close()
            if os.getenv("SaveHiddenState"):
                tmp_i = i

                folder_name = os.getenv("HIDDENSTATE_FOLDERNAME") if os.getenv(
                    "HIDDENSTATE_FOLDERNAME") else "hidden_state"
                os.makedirs(folder_name, exist_ok=True)

                prepad = os.getenv("PrePad") if os.getenv("PrePad") else ""
                prefix = os.path.join(
                    ".", folder_name, f"{prepad}output_sentence"+str(tmp_i))
                if not os.getenv("LAST_LAYER_ONLY"):
                    np_compress_save(prefix+"_encoder_hidden_states",
                                     tuple_to_numpy(predictions.encoder_hidden_states))
                else:
                    np_compress_save(
                        prefix+"_encoder_embedding", tuple_to_numpy(predictions.encoder_hidden_states)[-1])
                if not os.getenv("NO_DECODER_HIDDEN_STATE"):
                    np_compress_save(prefix+"_decoder_hidden_states",
                                     tuple_to_numpy(predictions.decoder_hidden_states))
                # if predictions.encoder_attentions:
                #    np.save(prefix+"_encoder_attentions.npy",tuple_to_numpy(predictions.encoder_attentions))
                # if predictions.cross_attentions:
                #    np.save(prefix+"_cross_attentions.npy",tuple_to_numpy(predictions.cross_attentions))
                # np.save(prefix+"_sequences.npy",tuple_to_numpy(predictions.sequences))

            for j, (input_ids, label_ids, prediction) in enumerate(
                    zip(inputs['input_ids'], inputs['labels'], predictions)):
                current_id = i * batch_size + j
                example = self.get_example(current_id)
                output_sentence = self.tokenizer.decode(prediction, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)

                yield example, output_sentence

    @abstractmethod
    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, returning the task-relevant metrics.
        """
        pass