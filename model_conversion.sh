atc --model=./human-pose-estimation.onnx --framework=5 --output=openpose_pytorch_560 --soc_version=Ascend310 --input_shape="data:1, 3, 560, 560" --input_format=NCHW --insert_op_conf=./insert_op.cfg

# 删除除 om 模型外额外生成的文件

rm fusion_result.json
rm -rf kernel_meta 
