# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Uses some code from
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py


import argparse
import configparser
import itertools
import json
import logging
import os
import shutil
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, Trainer, TrainerCallback

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from datasets import load_dataset
from evaluate import evaluate, get_avg_results, print_results
from utils import get_episode_indices

def main():
    # assert torch.cuda.is_available(), 'CUDA not available'

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('-c', '--config_file', type=str,
                        default='config.ini', help='configuration file')
    parser.add_argument('-e', '--eval', action='store_true',
                        default=False, help='run evaluation only')
    parser.add_argument('--evaluate_checkpoints', action='store_true', default=False,
                        help='evaluate intermediate checkpoints instead of the final model')
    parser.add_argument('--evaluate_last_checkpoint', action='store_true', default=False,
                        help='evaluate the last intermediate checkpoint instead of the final model')
    parser.add_argument('--evaluate_checkpoint_in_dir', type=str, default=None,
                        help='evaluate the checkpoint in the given directory')
    parser.add_argument('-a', '--evaluate_all', action='store_true', default=False,
                        help='evaluate intermediate checkpoints together with the final model')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='which GPU to use for evaluation')
    parser.add_argument('-v', '--verbose_results', action='store_true', default=False,
                        help='print results for each evaluation run')
    args, remaining_args = parser.parse_known_args()

    # read config file
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config

    # set defaults for other arguments
    defaults = {
        'overwrite_output_dir': True,
        'overwrite_cache': True,
        'per_device_eval_batch_size': 4,
        'learning_rate': 5e-4,
        'logging_steps': 100,     # do not log by default
        'save_steps': 0,        # do not save checkpoints by default
    }

    # the config file gives default values for the command line arguments
    defaults.update(dict(config.items(job)))
    for key in defaults:
        if defaults[key] in ['True', 'False']:
            # interpret True/False as boolean
            defaults[key] = config.getboolean(job, key)
        if defaults[key] == 'None':
            # interpret as None
            defaults[key] = None

    if args.eval:
        # run evaluation only
        defaults['do_train'] = False

    # parse remaining arguments and divide them into three categories
    second_parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    second_parser.set_defaults(**defaults)

    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(
        remaining_args)

    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass

    # process arguments related to max length
    if data_args.max_output_seq_length_eval is None:
        # defaults first to max_output_seq_length, then max_seq_length_eval, then max_seq_length
        data_args.max_output_seq_length_eval = data_args.max_output_seq_length \
            or data_args.max_seq_length_eval \
            or data_args.max_seq_length

    if data_args.max_output_seq_length is None:
        # defaults to max_seq_length
        data_args.max_output_seq_length = data_args.max_seq_length

    if data_args.max_seq_length_eval is None:
        # defaults to max_seq_length
        data_args.max_seq_length_eval = data_args.max_seq_length

    if data_args.chunk_size_eval is None:
        # defaults to chunk_size
        data_args.chunk_size_eval = data_args.chunk_size

    if data_args.chunk_overlap_eval is None:
        # defaults to chunk overlap
        data_args.chunk_overlap_eval = data_args.chunk_overlap

    # construct name for the output directory
    # for example: conll04-t5-base-ep200-len256-ratio0-b4-train
    output_dir = os.path.join(
        training_args.output_dir,
        f'{args.job}'
        f'-{model_args.model_name_or_path.split("/")[-1]}'
        f'-ep{round(training_args.num_train_epochs)}'
        f'-len{data_args.max_seq_length}'
    )

    if data_args.max_output_seq_length != data_args.max_seq_length:
        output_dir += f'-{data_args.max_output_seq_length}'

    if training_args.learning_rate != 5e-4:
        output_dir += f'-lr{training_args.learning_rate}'

    output_dir += f'-b{training_args.per_device_train_batch_size}' \
                  f'-{data_args.train_split}'

    if data_args.chunk_size != 128:
        output_dir += f'-chunk{data_args.chunk_size}'
    if data_args.chunk_overlap != 64:
        output_dir += f'-overlap{data_args.chunk_overlap}'

    if data_args.output_format is not None:
        output_dir += f'-{data_args.output_format}'
    if data_args.input_format is not None:
        output_dir += f'-{data_args.input_format}'
    if data_args.train_subset < 1:
        output_dir += f'-size{data_args.train_subset:.2f}'

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    # setup logging
    logging.basicConfig(
        filename=os.path.join(output_dir, 'logs.log'),
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # construct file name for the evaluation results
    evaluation_output_filename = f'results'
    if data_args.num_beams is not None:
        evaluation_output_filename += f'-{data_args.num_beams}beams'
    if data_args.max_seq_length_eval is not None:
        evaluation_output_filename += f'-len{data_args.max_seq_length_eval}'

    # create model config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )
    new_tokens = {"{", "}"} - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    # get list of dataset names
    dataset_names = data_args.datasets.split(',')

    # construct list of episode indices
    episode_indices = get_episode_indices(data_args.episodes)

    # episode loop
    # (note that the episode index is used as the random seed, so that each episode is reproducible)
    evaluation_results = defaultdict(list)
    for ep_idx in episode_indices:
        print()
        logging.info(
            f'Episode {ep_idx} ({len(episode_indices)} episodes total)')
        episode_output_dir = os.path.join(output_dir, f'episode{ep_idx}')

        try:
            os.mkdir(episode_output_dir)
        except FileExistsError:
            pass

        logging.info(f'Output directory: {episode_output_dir}')

        # checkpoints are saved in episode-specific directory
        training_args.output_dir = episode_output_dir

        # load pretrained model
        model = None
        if training_args.zero_shot or training_args.do_train:
            logging.info(f"Using model {model_args.model_name_or_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
            )

        # fine-tune the model
        if training_args.do_train:
            # load train dataset
            datasets = []
            for dataset_name in dataset_names:
                # logging.info(f'Process dataset {dataset_name} (train)')
                dataset = load_dataset(
                    dataset_name, data_args, split=data_args.train_split,
                    max_input_length=data_args.max_seq_length, max_output_length=data_args.max_output_seq_length,
                    tokenizer=tokenizer, seed=ep_idx, train_subset=data_args.train_subset,
                )
                datasets.append(dataset)

            train_dataset = torch.utils.data.ConcatDataset(
                datasets) if training_args.do_train else None

            num_gpus = args.gpu
            tracking_dataset_eval = load_dataset(
                'muc_ner_multiphase', data_args,
                max_input_length=data_args.max_seq_length_eval,
                max_output_length=data_args.max_output_seq_length_eval,
                tokenizer=tokenizer, split='dev', seed=ep_idx, shuffle=False, is_eval=True,
            )
            tracking_dataset_test = load_dataset(
                'muc_ner_multiphase', data_args,
                max_input_length=data_args.max_seq_length_eval,
                max_output_length=data_args.max_output_seq_length_eval,
                tokenizer=tokenizer, split='test', seed=ep_idx, shuffle=False, is_eval=True,
            )

            class CustomCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step % 500 == 0:
                        # Call your custom function here
                        model.eval()
                        device = torch.device(
                            "cuda", num_gpus) if torch.cuda.is_available() else torch.device("cpu")
                        model.to(device)
                        with open("train_predictions.txt", "a") as pred_file:
                            pred_file.write("EVAL PART\n")
                        tracking_dataset_eval.evaluate_dataset(
                            data_args=data_args, model=model, device=device, batch_size=training_args.per_device_eval_batch_size,
                            log_file="train_predictions.txt", is_multiphase=True)
                        with open("train_predictions.txt", "a") as pred_file:
                            pred_file.write("TEST PART\n")
                        tracking_dataset_test.evaluate_dataset(
                            data_args=data_args, model=model, device=device, batch_size=training_args.per_device_eval_batch_size,
                            log_file="train_predictions.txt", is_multiphase=True)
                        model.train()

            # construct trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                callbacks=[CustomCallback]
            )

            # start trainer
            logging.info('Start training')
            trainer.train(
                # model_path=model_args.model_name_or_path
            )
            with open("logs.json", "w") as log_file:
                log_file.write(json.dumps(trainer.state.log_history))

            # save model parameters
            if not os.path.exists("model_checkpoint"):
                os.mkdir("model_checkpoint")
            trainer.save_model("model_checkpoint")
        
        dev_dir = "data/muc_ner_multiphase/muc_ner_multiphase_dev.json"
        os.remove(dev_dir)
        shutil.copy("other_data/muc_ner_multiphase_dev.json", dev_dir)

        # run evaluation
        if not model:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "model_checkpoint",
                config=config,
                cache_dir=model_args.cache_dir,
            )
        model.eval()

        device = torch.device(
            "cuda", args.gpu) if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        dev_dataset = load_dataset(
            'muc_ner_multiphase', data_args,
            max_input_length=data_args.max_seq_length_eval,
            max_output_length=data_args.max_output_seq_length_eval,
            tokenizer=tokenizer, split='dev', seed=ep_idx, shuffle=False, is_eval=True,
        )
        _ = dev_dataset.evaluate_dataset(data_args=data_args, model=model, device=device, batch_size=training_args.per_device_eval_batch_size,
                                            log_file="dev_predictions.txt", is_multiphase=True)
        
        if args.do_test:
            test_dir = "data/muc_ner_multiphase/muc_ner_multiphase_test.json"
            os.remove(test_dir)
            shutil.copy("other_data/muc_ner_multiphase_test.json", test_dir)

            test_dataset = load_dataset(
                'muc_ner_multiphase', data_args,
                max_input_length=data_args.max_seq_length_eval,
                max_output_length=data_args.max_output_seq_length_eval,
                tokenizer=tokenizer, split='test', seed=ep_idx, shuffle=False, is_eval=True,
            )
            
            _ = test_dataset.evaluate_dataset(data_args=data_args, model=model, device=device, batch_size=training_args.per_device_eval_batch_size,
                                                log_file="test_predictions.txt", is_multiphase=True)


if __name__ == "__main__":
    main()
