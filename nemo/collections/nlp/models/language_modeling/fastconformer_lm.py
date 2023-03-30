# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from nemo.collections.common.losses import NLLLoss, SmoothedCrossEntropyLoss
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.nlp.modules.common import TokenClassifier
from typing import Any, Dict, List, Optional, Union

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from nemo.utils import logging
from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.nlp.modules.common.lm_utils import get_transformer
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
    TextGeneration,
)
from nemo.core.classes import ModelPT
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer


class FCModel(ModelPT, TextGeneration):
    """
    FastConformer-based LM pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        self.tokenizer = None
        super().__init__(cfg, trainer)

        #source: nemo/collections/nlp/models/language_modeling/megatron_base_model.py
        # build tokenizer (defaults to nemo supported tokenizers)
        self._build_tokenizer()
        # manipulate vocabulary (e.g., pad vocabulary for better efficiency)
        self._build_vocab()
        self.encoder = instantiate(self._cfg.encoder)
        self.wte = torch.nn.Embedding(self.padded_vocab_size, self._cfg.hidden_size)
        self.wpe = torch.nn.Embedding(self._cfg.max_position_embeddings, self._cfg.hidden_size)

        # Transformer decoder
        self._cfg.transformer_decoder.vocab_size = self.padded_vocab_size
        self.transformer_decoder = get_transformer(
            library=self._cfg.transformer_decoder.get("library", "nemo"),
            model_name=self._cfg.transformer_decoder.get("model_name", None),
            pretrained=self._cfg.transformer_decoder.get("pretrained", False),
            checkpoint_file=self._cfg.transformer_decoder.get("checkpoint_file", None),
            config_dict=self._cfg.transformer_decoder,
            encoder=False,
            pre_ln_final_layer_norm=self._cfg.transformer_decoder.get("pre_ln_final_layer_norm", False),
        )

        self.log_softmax = TokenClassifier(
            hidden_size=self._cfg.hidden_size,
            num_classes=self.padded_vocab_size,
            activation=self._cfg.head.activation,
            log_softmax=self._cfg.head.log_softmax,
            dropout=self._cfg.head.dropout,
            use_transformer_init=self._cfg.head.use_transformer_init,
        )
        self.log_softmax.mlp.layer0.weight = self.transformer_decoder.embedding.token_embedding.weight
        std_init_range = 1 / self.transformer_decoder.hidden_size ** 0.5
        self.transformer_decoder.apply(lambda module: transformer_weights_init(module, std_init_range))
        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))
        self.loss = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id if hasattr(self.tokenizer, "pad_id") else 0, label_smoothing=self._cfg.label_smoothing)
        self.val_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        if stage == 'predict':
            return
        else:
            # allowing restored models to optionally setup datasets
            self.build_train_valid_test_datasets()
            self.setup_training_data(self.cfg.data)
            self.setup_validation_data(self.cfg.data)
            self.setup_test_data(self.cfg.data)



    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        Override default Encoder-decoder tokenizer to use legacy=True for sentencepiece.
        """
        if hasattr(self._cfg.tokenizer, "sentencepiece_legacy"):
            legacy = self._cfg.tokenizer.sentencepiece_legacy
        else:
            legacy = True if self._cfg.tokenizer.library == 'sentencepiece' else False
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=self._cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("tokenizer.merge_file", self._cfg.tokenizer.merge_file),
            delimiter=self.cfg.tokenizer.get('delimiter', None),
            legacy=legacy,
        )

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        # TODO: add config to allow to disable it?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.tokenizer.vocab_size,
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    # @typecheck()
    def forward(self, input_ids, attention_mask, input_len):
        if self.mode == "byt5":
            encoded_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoded_len = input_len
            # encoded_input = [B, seq_len, hid_dim]
            # swap seq_len and hid_dim dimensions to get [B, hid_dim, seq_len]
            encoded_input = encoded_input.transpose(1, 2)
        elif self.mode == "conformer_bpe":
            input_embedding = self.embedding(input_ids)
            input_embedding = input_embedding.transpose(1, 2)
            encoded_input, encoded_len = self.encoder(audio_signal=input_embedding, length=input_len)
        else:
            raise ValueError(f"{self.mode} is not supported. Choose from {self.supported_modes}")

        log_probs = self.decoder(encoder_output=encoded_input)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, greedy_predictions, encoded_len

    # ===== Training Functions ===== #
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )

        loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        return super().training_epoch_end(outputs)

    # ===== Validation Functions ===== #
    def validation_step(self, batch, batch_idx, dataloader_idx=0, split="val"):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, greedy_predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )
        val_loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )

        self._wer.update(
            predictions=log_probs, targets=targets, target_lengths=target_lengths, predictions_lengths=encoded_len
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()

        self._per.update(
            predictions=log_probs, targets=targets, target_lengths=target_lengths, predictions_lengths=encoded_len
        )
        per, per_num, per_denom = self._per.compute()
        self._per.reset()

        self.log(f"{split}_loss", val_loss)
        return {
            f"{split}_loss": val_loss,
            f"{split}_wer_num": wer_num,
            f"{split}_wer_denom": wer_denom,
            f"{split}_wer": wer,
            f"{split}_per_num": per_num,
            f"{split}_per_denom": per_denom,
            f"{split}_per": per,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, dataloader_idx, split="test")



    # ===== Dataset Setup Functions ===== #
    def _setup_dataloader_from_config(self, cfg: DictConfig, name: str):
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")

        if not os.path.exists(cfg.manifest_filepath):
            raise ValueError(f"{cfg.dataset.manifest_filepath} not found")

        dataset = instantiate(
            cfg.dataset,
            manifest_filepath=cfg.manifest_filepath,
            phoneme_field=cfg.dataset.phoneme_field,
            grapheme_field=cfg.dataset.grapheme_field,
            tokenizer_graphemes=self.tokenizer_grapheme,
            do_lower=self._cfg.tokenizer_grapheme.do_lower,
            tokenizer_phonemes=self.tokenizer,
            labels=self.vocabulary,
            max_source_len=self.max_source_len,
            with_labels=True,
        )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg: DictConfig):
        if not cfg or cfg.manifest_filepath is None:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg, name="train")

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict] = None):
        if not val_data_config or val_data_config.manifest_filepath is None:
            self._validation_dl = None
            return
        super().setup_multiple_validation_data(val_data_config)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict] = None):
        if not test_data_config or test_data_config.manifest_filepath is None:
            self._test_dl = None
            return
        super().setup_multiple_test_data(test_data_config)

    def setup_validation_data(self, cfg: Optional[DictConfig]):
        if not cfg or cfg.manifest_filepath is None:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg, name="val")

    def setup_test_data(self, cfg: Optional[DictConfig]):
        if not cfg or cfg.manifest_filepath is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg, name="test")

    # ===== List Available Models - N/A =====$
    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
