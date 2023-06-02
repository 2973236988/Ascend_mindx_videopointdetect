#!/usr/bin/env python
# coding=utf-8

# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

import argparse

import torch
from torch.autograd import Variable

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_onnx(net, output_name):
    """
    Convert network from pytorch to onnx

    Args:
        net: The network which will be converted
        output_name: The name of output nodes

    """
    net_input = Variable(torch.randn(1, 3, 560, 560))
    input_names = ['data']
    output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                    'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']
    torch.onnx.export(net, net_input, output_name, verbose=True,
                      opset_version=11, input_names=input_names, output_names=output_names)
    print("convert to onnx complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/checkpoint_iter_370000.pth',
                        help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation.onnx',
                        help='name of output model in ONNX format')
    args = parser.parse_args()
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    load_state(net, checkpoint)
    convert_to_onnx(net, args.output_name)
