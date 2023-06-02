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

import sys
import time
import cv2
import numpy as np

sys.path.append("./proto")
from StreamManagerApi import StreamManagerApi, StringVector
import mxpiOpenposeProto_pb2 as mxpiOpenposeProto

COCO_PAIRS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
              (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)]

COCO_PAIRS_RENDER = COCO_PAIRS[:-2]

COCO_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def draw_pose_bbox(input_img, person_list):
    """
    Draw person keypoints and skeletons on input image

    Args:
        input_img: input image
        person_list: MxpiPersonList object, each element of which is a MxpiPersonInfo object that stores data of person

    Returns:
        None

    """
    joints = []
    for person in person_list:
        skeletons = person.skeletonInfoVec
        x_coords, y_coords, centers = [], [], {}
        seen_idx = []
        # draw keypoints
        for skele in skeletons:
            part_idx1 = skele.cocoSkeletonIndex1
            part_idx2 = skele.cocoSkeletonIndex2
            if part_idx1 not in seen_idx:
                seen_idx.append(part_idx1)
                center = (int(skele.x0), int(skele.y0))
                centers[part_idx1] = center
                x_coords.append(center[0])
                y_coords.append(center[1])
                cv2.circle(input_img, center, 3, COCO_COLORS[part_idx1], thickness=3, lineType=8, shift=0)
            if part_idx2 not in seen_idx:
                seen_idx.append(part_idx2)
                center = (int(skele.x1), int(skele.y1))
                centers[part_idx2] = center
                x_coords.append(center[0])
                y_coords.append(center[1])
                cv2.circle(input_img, center, 3, COCO_COLORS[part_idx2], thickness=3, lineType=8, shift=0)

        # draw skeletons
        for pair_order, pair in enumerate(COCO_PAIRS_RENDER):
            if pair[0] not in seen_idx \
                    or pair[1] not in seen_idx:
                continue
            cv2.line(input_img, centers[pair[0]], centers[pair[1]], COCO_COLORS[pair_order], 3, cv2.LINE_AA)

        joints.append(centers)

    return input_img, joints


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("pipeline/openpose.pipeline", "rb") as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    stream_name = b"openpose_detection"
    in_plugin_id = 0

    infer_time_list = []
    times = 0
    frame_frequency = 1
    file_name = "test.mp4"
    video_capture = cv2.VideoCapture(file_name)
    video_writer = cv2.VideoWriter(filename=file_name.split('.')[0] + "_detect_result.mp4",
                                 fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),  # fourcc = -1,
                                 frameSize=(int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                 fps=int(video_capture.get(cv2.CAP_PROP_FPS) / frame_frequency))

    keys = [b"mxpi_openposepostprocess0"]
    key_vec = StringVector()
    for key in keys:
        key_vec.push_back(key)

    while True:
        times += 1
        start_time = time.perf_counter()
        res, img = video_capture.read()
        if not res:
            break
        if times % 3 == 1:
            infer_result = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, key_vec)
            if infer_result.size() == 0:
                print("infer_result is null")
                exit()
            if infer_result[0].errorCode != 0:
                print("infer_result error. errorCode=%d" % infer_result[0].errorCode)
                exit()
        end_time = time.perf_counter()
        infer_time_list.append(end_time - start_time)
        result_personlist = mxpiOpenposeProto.MxpiPersonList()
        result_personlist.ParseFromString(infer_result[0].messageBuf)
        detect_person_list = result_personlist.personInfoVec
        image_show = draw_pose_bbox(img, detect_person_list)[0]
        video_writer.write(image_show)
    video_writer.release()
    # destroy streams
    stream_manager_api.DestroyAllStreams()

    print('----------------')
    print("average fps: ", 1 / np.array(infer_time_list).mean())
