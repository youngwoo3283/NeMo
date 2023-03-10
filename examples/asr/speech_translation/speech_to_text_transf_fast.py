# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""
# Training the model
```sh
python speech_to_text_transf.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.text.tar_files=<path to tar files with text data> \
    model.train_ds.text.metadata_file=<path to text metadata file> \
    model.train_ds.audio.tarred_audio_filepaths=<path to tar files with audio> \
    model.train_ds.audio_manifest_filepath=<path to audio data manifest> \
    model.validation_ds.manifest_filepath=<path to validation manifest> \
    model.test_ds.manifest_filepath=<path to test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.model_path=<path to speech tokenizer model> \
    model.tokenizer.type=<either bpe, wpe, or yttm> \
    model.encoder_tokenizer.tokenizer_model=<path to cmudict> \
    model.encoder_tokenizer.vocab_file=<path to heteronyms> \
    model.decoder_tokenizer.tokenizer_model=<path to decoder tokenizer model> \
    trainer.gpus=-1 \
    trainer.accelerator="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```


"""

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from collections import OrderedDict

from nemo.collections.asr.models import EncDecTransfModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="../conf/transformer_decoder/", config_name="speech_translation_with_tts_test")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecTransfModelBPE(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # # Initialize encoder with the weights from pre-trained ASR encoder
    # enc_weights = OrderedDict()
    # weights = torch.load("/train_data/models/asr/rnnt_bn_fast.ckpt")
    # for key in weights.keys():
    #     if key[:7]=="encoder":
    #         enc_weights[key[8:]]=weights[key]
    # asr_model.encoder.load_state_dict(enc_weights)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()