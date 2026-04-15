# camera_agent

一个面向后续墙面坐标生成、工件识别、激光投影的 Python/OpenCV 实验项目。

当前项目已经具备两条可运行闭环：

1. 相机 / Aruco 闭环
图像或摄像头输入 -> Aruco 检测 -> 墙面坐标 -> 投影目标点 -> 执行队列 -> 执行播放验证

2. ILDA / laser 文件闭环
`.ild` 文件输入 -> ILDA 解析 -> 预览图 -> 执行队列 -> 执行播放验证

当前阶段不依赖 RealSense SDK，不处理深度，不依赖训练模型。
当前项目仍然使用普通 OpenCV/V4L2 相机读取，不依赖 RealSense SDK。
在当前 Linux 机器上，RealSense 可通过 `camera index 8` 使用。
当前 ArUco 检测建议优先使用打印出来的纯净 marker，而不是带 UI 元素的手机屏幕。

## 1. 项目结构

项目根目录当前主要脚本如下：

- [`camera_input.py`](/Users/ruirenmei/camera_agent/camera_input.py)
  摄像头打开、预览、截图保存。

- [`aruco_detect.py`](/Users/ruirenmei/camera_agent/aruco_detect.py)
  Aruco 检测，支持图片和摄像头，支持导出检测 JSON 和墙面坐标 JSON。

- [`aruco_to_wall_coords.py`](/Users/ruirenmei/camera_agent/aruco_to_wall_coords.py)
  将 Aruco 检测结果转换为墙面平面坐标。

- [`wall_coords_viewer.py`](/Users/ruirenmei/camera_agent/wall_coords_viewer.py)
  将墙面坐标或目标点 JSON 渲染成 2D 平面预览。

- [`projection_targets.py`](/Users/ruirenmei/camera_agent/projection_targets.py)
  将墙面坐标整理为激光投影友好的目标点列表。

- [`projection_simulator.py`](/Users/ruirenmei/camera_agent/projection_simulator.py)
  在 2D 墙面平面上模拟投影落点和投影顺序。

- [`projection_executor_stub.py`](/Users/ruirenmei/camera_agent/projection_executor_stub.py)
  将目标点转换成硬件无关的执行队列。

- [`projection_executor_player.py`](/Users/ruirenmei/camera_agent/projection_executor_player.py)
  在终端按步骤播放执行队列，模拟执行流程。

- [`camera_pipeline.py`](/Users/ruirenmei/camera_agent/camera_pipeline.py)
  相机 / Aruco 统一入口。

- [`ild_loader.py`](/Users/ruirenmei/camera_agent/ild_loader.py)
  解析 `.ild` 文件、导出摘要、生成预览图。

- [`ild_to_execution_queue.py`](/Users/ruirenmei/camera_agent/ild_to_execution_queue.py)
  将 `.ild` 文件转换成统一执行队列。

- [`laser_pipeline.py`](/Users/ruirenmei/camera_agent/laser_pipeline.py)
  ILDA / laser 统一入口。

## 2. 环境要求

- Python 3.9+
- macOS / Linux / Windows 均可，当前项目在 macOS 上开发
- OpenCV contrib 版本

## 3. 安装依赖

建议使用当前 Python 环境直接安装：

```bash
python3 -m pip install opencv-contrib-python
```

如果你想确认安装是否成功：

```bash
python3 -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco'))"
```

预期输出类似：

```text
4.13.0
True
```

## 4. 输出目录说明

运行过程中会自动生成以下目录：

- `outputs/captured_frames/`
  摄像头截图。

- `outputs/aruco_detect/`
  Aruco 检测结果图、检测 JSON。

- `outputs/wall_coords/`
  墙面坐标 JSON。

- `outputs/wall_coords_viewer/`
  墙面坐标平面图。

- `outputs/projection_targets/`
  投影目标点 JSON。

- `outputs/projection_simulator/`
  投影模拟图。

- `outputs/projection_executor/`
  执行队列 JSON。

- `outputs/ilda/`
  ILDA 摘要 JSON、ILDA 预览图。

## 5. 快速开始

### 5.1 先测试摄像头输入

运行：

```bash
python3 camera_input.py
```

在当前 Linux + RealSense 环境中，建议显式使用：

```bash
python3 camera_input.py --camera 8
```

操作：

- 按 `s` 保存当前帧
- 按 `q` 退出

这一步主要确认：

- 摄像头是否能正常打开
- OpenCV 窗口是否能正常弹出
- 图像是否能保存到 `outputs/captured_frames/`

### 5.2 先测试 ILDA 文件

项目当前默认使用这个文件：

`/Users/ruirenmei/Desktop/bluelaser.ild`

运行：

```bash
python3 ild_loader.py
```

如果只想解析，不弹窗口：

```bash
python3 ild_loader.py --no-window
```

这一步会：

- 解析 `.ild` 文件
- 打印 frame 数、点数、格式信息
- 保存摘要 JSON
- 保存预览图

## 6. 相机 / Aruco 闭环

这是当前最重要的一条图像链路。

### 6.1 单独运行 Aruco 检测

图片模式：

```bash
python3 aruco_detect.py --image path/to/image.jpg
```

摄像头模式：

```bash
python3 aruco_detect.py --camera 8
```

带目标 ID 白名单和连续稳定确认：

```bash
python3 aruco_detect.py --camera 8 --target-marker-ids 37 --min-stable-frames 3
```

如果想同时导出墙面坐标：

```bash
python3 aruco_detect.py \
  --image path/to/image.jpg \
  --marker-size-mm 50 \
  --origin-marker-id 23
```

说明：

- `--marker-size-mm`
  参考 Aruco 实际边长，单位 mm。

- `--origin-marker-id`
  用作墙面坐标原点的标签 ID。

- `--origin`
  原点放在参考标签的 `top_left` 或 `center`，默认 `top_left`。

运行结果：

- 图上会绘制标签边框、四角点、中心点、ID
- 终端会打印 marker 信息
- 检测 JSON 会保存到 `outputs/aruco_detect/`
- 如果提供了 `--marker-size-mm`，墙面坐标 JSON 会保存到 `outputs/wall_coords/`

### 6.2 单独把检测结果转成墙面坐标

```bash
python3 aruco_to_wall_coords.py \
  --input outputs/aruco_detect/your_detection.json \
  --marker-size-mm 50 \
  --origin-marker-id 23
```

如果想让原点放在参考标签中心：

```bash
python3 aruco_to_wall_coords.py \
  --input outputs/aruco_detect/your_detection.json \
  --marker-size-mm 50 \
  --origin-marker-id 23 \
  --origin center
```

### 6.3 可视化墙面坐标

```bash
python3 wall_coords_viewer.py --input outputs/wall_coords/your_wall_coords.json
```

操作：

- 按 `s` 保存预览图
- 按 `q` 退出

### 6.4 从墙面坐标生成投影目标点

默认导出中心点：

```bash
python3 projection_targets.py \
  --input outputs/wall_coords/your_wall_coords.json
```

导出四角点和中心点：

```bash
python3 projection_targets.py \
  --input outputs/wall_coords/your_wall_coords.json \
  --target-type all
```

只导出指定 marker：

```bash
python3 projection_targets.py \
  --input outputs/wall_coords/your_wall_coords.json \
  --target-type all \
  --marker-ids 23 42
```

### 6.5 模拟投影目标点

```bash
python3 projection_simulator.py \
  --input outputs/projection_targets/your_projection_targets.json \
  --connect-order \
  --show-labels
```

说明：

- `--connect-order`
  按目标点顺序连线，模拟投影路径。

- `--show-labels`
  显示每个 target 的完整标签。

### 6.6 把目标点转换成执行队列

```bash
python3 projection_executor_stub.py \
  --input outputs/projection_targets/your_projection_targets.json
```

可调参数示例：

```bash
python3 projection_executor_stub.py \
  --input outputs/projection_targets/your_projection_targets.json \
  --dwell-ms 800 \
  --travel-ms 200 \
  --settle-ms 100 \
  --repeat 2 \
  --laser-power 0.7
```

### 6.7 播放执行队列

快速 dry run：

```bash
python3 projection_executor_player.py \
  --input outputs/projection_executor/your_execution_queue.json \
  --dry-run
```

只播放前 10 步：

```bash
python3 projection_executor_player.py \
  --input outputs/projection_executor/your_execution_queue.json \
  --dry-run \
  --max-steps 10
```

真实按时长播放：

```bash
python3 projection_executor_player.py \
  --input outputs/projection_executor/your_execution_queue.json
```

### 6.8 相机 / Aruco 一条命令跑完整闭环

图片模式：

```bash
python3 camera_pipeline.py \
  --image path/to/image.jpg \
  --marker-size-mm 50 \
  --origin-marker-id 23 \
  --target-type all \
  --export-execution-queue
```

摄像头模式：

```bash
python3 camera_pipeline.py \
  --camera 8 \
  --marker-size-mm 50 \
  --origin-marker-id 23 \
  --target-type centers \
  --export-execution-queue
```

说明：

- `camera_pipeline.py` 在实时摄像头模式下，需要在窗口中按 `s` 才会导出当前结果。
- 按下 `s` 后会导出当前帧对应的 `aruco_detect`、`wall_coords`、`projection_targets`，如果启用了 `--export-execution-queue`，还会继续导出 `execution_queue`。
- `--origin-marker-id` 只用于定义 wall coordinate origin，不会限制保留哪些 marker。
- 如果没有显式传 `--target-marker-ids`，则会保留所有通过检测与稳定判定的 marker。
- 更换 `--origin-marker-id` 只会改变坐标系原点，不会改变 marker 集合。

当前 Linux + RealSense 环境推荐命令：

```bash
python camera_pipeline.py \
  --camera 8 \
  --dict DICT_4X4_50 \
  --width 1280 \
  --height 720 \
  --marker-size-mm 50 \
  --origin-marker-id 37 \
  --origin top_left \
  --target-type centers \
  --export-execution-queue
```

当左上角出现多个稳定 ID 时，按 `s` 保存当前完整结果，按 `q` 退出。

执行队列时间参数也可以在这里直接设置：

```bash
python3 camera_pipeline.py \
  --image path/to/image.jpg \
  --marker-size-mm 50 \
  --origin-marker-id 23 \
  --target-type all \
  --export-execution-queue \
  --dwell-ms 800 \
  --travel-ms 200 \
  --settle-ms 100 \
  --repeat 2 \
  --laser-power 0.7
```

## 7. ILDA / laser 文件闭环

这是当前第二条完整闭环，面向后续真实 laser 文件输入。

### 7.1 单独解析 ILDA 文件

默认就是桌面的 `bluelaser.ild`：

```bash
python3 ild_loader.py
```

### 7.2 单独把 ILDA 转成执行队列

```bash
python3 ild_to_execution_queue.py
```

带参数示例：

```bash
python3 ild_to_execution_queue.py \
  --point-step-ms 20 \
  --blank-step-ms 6 \
  --repeat 2 \
  --laser-power 0.8
```

### 7.3 播放 ILDA 生成的执行队列

```bash
python3 projection_executor_player.py \
  --input outputs/projection_executor/your_ilda_execution_queue.json \
  --dry-run \
  --max-steps 20
```

### 7.4 ILDA / laser 一条命令跑完整闭环

最简单：

```bash
python3 laser_pipeline.py
```

不弹窗：

```bash
python3 laser_pipeline.py --no-window
```

直接生成并播放前 10 步：

```bash
python3 laser_pipeline.py --no-window --play --dry-run --max-steps 10
```

更接近真实时序：

```bash
python3 laser_pipeline.py --play
```

带参数示例：

```bash
python3 laser_pipeline.py \
  --point-step-ms 20 \
  --blank-step-ms 6 \
  --repeat 2 \
  --laser-power 0.8 \
  --play --dry-run --max-steps 20
```

## 8. 建议的完整验证顺序

如果你现在想完整验证项目，建议按下面顺序来。

### 8.1 先跑 ILDA / laser 闭环

```bash
python3 laser_pipeline.py --no-window --play --dry-run --max-steps 20
```

检查点：

- 是否成功解析 `bluelaser.ild`
- 是否生成预览图
- 是否生成执行队列 JSON
- `projection_executor_player.py` 的播放顺序是否合理

### 8.2 再跑相机 / Aruco 闭环

```bash
python3 camera_pipeline.py \
  --image path/to/your/test.jpg \
  --marker-size-mm 50 \
  --origin-marker-id 23 \
  --target-type all \
  --export-execution-queue
```

检查点：

- Aruco 是否被正确检测到
- 墙面坐标是否合理
- 目标点是否生成成功
- 执行队列是否生成成功

### 8.3 再单独看投影模拟

```bash
python3 projection_simulator.py \
  --input outputs/projection_targets/your_projection_targets.json \
  --connect-order --show-labels
```

### 8.4 最后播放相机链路生成的执行队列

```bash
python3 projection_executor_player.py \
  --input outputs/projection_executor/your_execution_queue.json \
  --dry-run
```

## 9. 数据格式概览

### 9.1 Aruco 检测 JSON

主要字段：

- `source`
- `dictionary`
- `image_size`
- `marker_count`
- `markers`

每个 marker 下有：

- `id`
- `corners`
- `center`

### 9.2 墙面坐标 JSON

主要字段：

- `reference_marker`
- `coordinate_system`
- `marker_count`
- `markers`

每个 marker 下有：

- `image`
- `wall_mm`

### 9.3 投影目标点 JSON

主要字段：

- `target_type`
- `target_count`
- `targets`

每个 target 下有：

- `label`
- `marker_id`
- `point_type`
- `wall_mm`

### 9.4 执行队列 JSON

主要字段：

- `device_name`
- `queue_config`
- `step_count`
- `estimated_total_duration_ms`
- `steps`

每个 step 下有：

- `step_index`
- `cycle_index`
- `action`
- `target_label`
- `wall_mm`
- `duration_ms`

## 10. 当前假设和限制

当前版本是最小可用闭环，存在一些明确假设：

- 墙面坐标阶段默认参考 Aruco 和目标点都位于同一平面上
- 没有使用相机标定参数
- 没有使用 RealSense 深度
- 没有接真实激光硬件
- `projection_executor_stub.py` 和 `projection_executor_player.py` 目前都是逻辑验证工具
- `ild_loader.py` 目前只支持 ILDA format `0` 和 `1`

## 11. 常见问题

### 11.1 `ModuleNotFoundError: No module named 'cv2'`

安装：

```bash
python3 -m pip install opencv-contrib-python
```

### 11.2 `当前 OpenCV 环境缺少 aruco 模块`

说明你安装的不是 contrib 版本，需要重新安装：

```bash
python3 -m pip install opencv-contrib-python
```

### 11.3 没检测到 Aruco 标签

优先检查：

- 字典是否匹配，默认是 `DICT_4X4_50`
- 图像是否清晰
- 标签是否完整出现在画面中
- 光照是否过暗或过曝

### 11.4 墙面坐标导出失败

优先检查：

- 是否提供了 `--marker-size-mm`
- `--origin-marker-id` 是否真出现在检测结果里
- 参考 marker 是否完整、稳定

### 11.5 执行队列播放太慢

可以用：

```bash
python3 projection_executor_player.py --input your_queue.json --speed 5
```

或者先用：

```bash
python3 projection_executor_player.py --input your_queue.json --dry-run
```

## 12. 后续可扩展方向

后续如果继续推进，这个项目最自然的扩展方向包括：

- 加入相机标定和去畸变
- 将图像坐标到墙面坐标的映射从单 marker 扩展到多 marker 稳定估计
- 接入真实 laser 控制接口
- 加入工件识别模块
- 将 `camera_pipeline.py` 和 `laser_pipeline.py` 进一步统一成单一调度入口

## 13. 当前推荐命令

如果你今天只想最快验证整个项目，优先跑这两条：

相机 / Aruco 闭环：

```bash
python3 camera_pipeline.py \
  --image path/to/your/test.jpg \
  --marker-size-mm 50 \
  --origin-marker-id 23 \
  --target-type all \
  --export-execution-queue
```

ILDA / laser 闭环：

```bash
python3 laser_pipeline.py --no-window --play --dry-run --max-steps 20
```
