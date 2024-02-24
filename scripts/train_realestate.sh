#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# export CUDA_VISIBLE_DEVICES=1

SCENE=scene_name
EXPERIMENT=realestate_lvl_encoded
DATA_DIR=/raid/ankit/srinath/jax-mipnerf360/data/scene_name
CHECKPOINT_DIR=data/"$EXPERIMENT"/"$SCENE"

#rm "$CHECKPOINT_DIR"/*
python -m train \
  --gin_configs=configs/realestate_mip360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr