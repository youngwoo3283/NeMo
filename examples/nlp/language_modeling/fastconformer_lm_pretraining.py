# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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



from omegaconf.omegaconf import OmegaConf, open_dict
import pytorch_lightning as pl
import torch
from collections import OrderedDict

from nemo.collections.nlp.models.language_modeling.fastconformer_lm import FCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="fastconformer_lm_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = FCModel(cfg=cfg.model, trainer=trainer)
    # Initialize encoder with the weights from pre-trained ASR encoder
    enc_weights = OrderedDict()
    weights = torch.load("/media/ebakhturina/DATA/nmt/DEPENDENCY/models/asr/rnnt_bn_fast.ckpt")
    for key in weights.keys():
        if key[:7] == "encoder":
            enc_weights[key[8:]] = weights[key]
    model.encoder.load_state_dict(enc_weights)
    trainer.fit(model)




if __name__ == '__main__':
    main()
