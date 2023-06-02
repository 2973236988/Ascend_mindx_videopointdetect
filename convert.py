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

import cv2

if __name__ == '__main__':
    file_name = "test.mp4"
    video_capture = cv2.VideoCapture(file_name)
    video_writer = cv2.VideoWriter(filename="processed_" + file_name.split('.')[0] + ".mp4",
                                  fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                  frameSize=(
                                      int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                  fps=int(video_capture.get(cv2.CAP_PROP_FPS)))
    time = 0
    while True:
        time += 1
        res, img = video_capture.read()
        if not res:
            break
        if time % 3 == 1:
            video_writer.write(img)
    video_writer.release()
    print('----------------')
    print('video conversion complete')
