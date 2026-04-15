"""
ONNX转TensorRT引擎脚本
注意：必须在Jetson设备上运行此脚本！

使用方法：
python onnx_to_tensorrt.py --onnx onnx/fastspeech2.onnx --output fastspeech2.engine
"""

import argparse
import os
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def onnx_to_tensorrt(
    onnx_file_path,
    engine_file_path,
    max_batch_size=1,
    max_workspace_size=1 << 30,  # 1GB
    fp16_mode=True,
    int8_mode=False,
    dynamic_shapes=None,
):
    """
    将ONNX模型转换为TensorRT引擎
    
    Args:
        onnx_file_path: ONNX模型路径
        engine_file_path: 输出的TensorRT引擎路径
        max_batch_size: 最大批次大小
        max_workspace_size: 最大工作空间大小（字节）
        fp16_mode: 是否使用FP16精度
        int8_mode: 是否使用INT8精度
        dynamic_shapes: 动态形状配置
    """
    print("=" * 60)
    print("ONNX转TensorRT引擎")
    print("=" * 60)
    
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX文件不存在: {onnx_file_path}")
    
    print(f"输入ONNX文件: {onnx_file_path}")
    print(f"输出引擎文件: {engine_file_path}")
    print(f"最大批次大小: {max_batch_size}")
    print(f"最大工作空间: {max_workspace_size / (1 << 20):.2f} MB")
    print(f"FP16模式: {fp16_mode}")
    print(f"INT8模式: {int8_mode}")
    
    # 创建构建器
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析ONNX模型
    print("\n解析ONNX模型...")
    with open(onnx_file_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("解析ONNX模型失败!")
            for error in range(parser.num_errors):
                print(f"  错误 {error}: {parser.get_error(error)}")
            return False
    
    print("ONNX模型解析成功!")
    
    # 打印网络信息
    print(f"\n网络输入:")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"  {input_tensor.name}: {input_tensor.shape} ({input_tensor.dtype})")
    
    print(f"\n网络输出:")
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"  {output_tensor.name}: {output_tensor.shape} ({output_tensor.dtype})")
    
    # 配置构建器
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # 设置精度模式
    if fp16_mode and builder.platform_has_fast_fp16:
        print("\n启用FP16模式")
        config.set_flag(trt.BuilderFlag.FP16)
    
    if int8_mode and builder.platform_has_fast_int8:
        print("启用INT8模式")
        config.set_flag(trt.BuilderFlag.INT8)
        # 需要校准数据集
        # config.int8_calibrator = calibrator
    
    # 设置动态形状（如果需要）
    if dynamic_shapes:
        print("\n设置动态形状...")
        profile = builder.create_optimization_profile()
        for input_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            print(f"  {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
        config.add_optimization_profile(profile)
    
    # 构建引擎
    print("\n构建TensorRT引擎...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    import time
    start_time = time.time()
    
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("构建引擎失败!")
        return False
    
    build_time = time.time() - start_time
    print(f"引擎构建成功! 耗时: {build_time:.2f} 秒")
    
    # 保存引擎
    print(f"\n保存引擎到: {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    engine_size = os.path.getsize(engine_file_path) / (1 << 20)  # MB
    print(f"引擎文件大小: {engine_size:.2f} MB")
    
    print("=" * 60)
    print("转换完成!")
    print("=" * 60)
    
    return True


def test_tensorrt_engine(engine_file_path, test_input=None):
    """
    测试TensorRT引擎
    
    Args:
        engine_file_path: TensorRT引擎路径
        test_input: 测试输入数据
    """
    print("\n" + "=" * 60)
    print("测试TensorRT引擎")
    print("=" * 60)
    
    # 加载引擎
    print(f"加载引擎: {engine_file_path}")
    with open(engine_file_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 打印引擎信息
    print(f"\n引擎信息:")
    print(f"  批次大小: {engine.max_batch_size}")
    print(f"  设备内存: {engine.device_memory_size / (1 << 20):.2f} MB")
    
    # 获取输入输出信息
    bindings = []
    for i in range(engine.num_bindings):
        binding = engine[i]
        shape = engine.get_binding_shape(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print(f"  {binding}: shape={shape}, dtype={dtype}")
        bindings.append(binding)
    
    print("\n引擎加载成功!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX转TensorRT引擎")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX模型路径")
    parser.add_argument("--output", type=str, required=True, help="输出引擎路径")
    parser.add_argument("--batch_size", type=int, default=1, help="最大批次大小")
    parser.add_argument("--workspace", type=int, default=1, help="最大工作空间大小(GB)")
    parser.add_argument("--fp16", action="store_true", help="启用FP16模式")
    parser.add_argument("--int8", action="store_true", help="启用INT8模式")
    parser.add_argument("--test", action="store_true", help="转换后测试引擎")
    
    args = parser.parse_args()
    
    # 动态形状配置（根据FastSpeech2模型调整）
    dynamic_shapes = {
        'texts': (
            (1, 1),      # 最小形状
            (1, 50),     # 优化形状
            (1, 200),    # 最大形状
        ),
    }
    
    # 转换模型
    success = onnx_to_tensorrt(
        args.onnx,
        args.output,
        max_batch_size=args.batch_size,
        max_workspace_size=args.workspace << 30,
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        dynamic_shapes=dynamic_shapes,
    )
    
    if success and args.test:
        test_tensorrt_engine(args.output)
