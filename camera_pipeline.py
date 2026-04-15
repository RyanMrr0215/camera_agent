import argparse
import sys
import time
from pathlib import Path

import cv2

from aruco_detect import (
    OUTPUT_DIR as DETECT_OUTPUT_DIR,
    WINDOW_NAME,
    annotate_detection_result,
    build_detection_payload,
    collect_marker_results,
    detect_markers,
    draw_live_info,
    filter_and_enrich_results,
    format_results_text,
    get_aruco_dictionary,
    get_pose_inputs,
    load_image,
    save_detection_json,
    save_result_image,
    validate_image_path,
)
from aruco_runtime import StableMarkerTracker
from aruco_to_wall_coords import (
    OUTPUT_DIR as WALL_OUTPUT_DIR,
    convert_detection_to_wall_payload,
    print_wall_results,
    save_wall_json_with_prefix,
)
from camera_source import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    get_camera_debug_info,
    open_camera,
    read_bgr_frame,
)
from projection_executor_stub import (
    OUTPUT_DIR as EXECUTION_OUTPUT_DIR,
    convert_projection_to_execution_payload,
    print_execution_summary,
    save_execution_json_with_prefix,
)
from projection_targets import (
    OUTPUT_DIR as PROJECTION_OUTPUT_DIR,
    convert_wall_to_projection_payload,
    print_projection_targets,
    save_projection_json_with_prefix,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full camera pipeline: detection, wall coords, and projection targets."
    )
    parser.add_argument(
        "--image",
        help="Path to the input image, for example: images/test.jpg",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="Camera index for real-time pipeline. Default: 8",
    )
    parser.add_argument(
        "--dict",
        default="DICT_4X4_50",
        help="Aruco dictionary name. Default: DICT_4X4_50",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help="Camera width for live mode. Default: 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help="Camera height for live mode. Default: 480",
    )
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        required=True,
        help="Physical size of the reference Aruco marker in millimeters",
    )
    parser.add_argument(
        "--origin-marker-id",
        type=int,
        help="Marker ID used as wall origin. Default: first detected stable marker",
    )
    parser.add_argument(
        "--origin",
        choices=["top_left", "center"],
        default="top_left",
        help="Wall coordinate origin on the reference marker. Default: top_left",
    )
    parser.add_argument(
        "--target-type",
        choices=["centers", "corners", "all"],
        default="centers",
        help="Projection target type. Default: centers",
    )
    parser.add_argument(
        "--target-marker-ids",
        nargs="+",
        type=int,
        help="Optional marker IDs to keep as valid detections and projection targets",
    )
    parser.add_argument(
        "--min-stable-frames",
        type=int,
        default=3,
        help="Frames required before a detection is considered stable. Default: 3",
    )
    parser.add_argument(
        "--stable-center-threshold-px",
        type=float,
        default=30.0,
        help="Max allowed center movement between frames. Default: 30 px",
    )
    parser.add_argument(
        "--label-prefix",
        default="target",
        help="Prefix used when naming projection targets. Default: target",
    )
    parser.add_argument(
        "--export-execution-queue",
        action="store_true",
        help="Also export a projection execution queue JSON",
    )
    parser.add_argument(
        "--dwell-ms",
        type=int,
        default=500,
        help="Execution queue dwell time per target in milliseconds. Default: 500",
    )
    parser.add_argument(
        "--travel-ms",
        type=int,
        default=150,
        help="Execution queue travel time between targets in milliseconds. Default: 150",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=80,
        help="Execution queue settling time after movement in milliseconds. Default: 80",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Execution queue repeat count. Default: 1",
    )
    parser.add_argument(
        "--laser-power",
        type=float,
        default=1.0,
        help="Execution queue normalized laser power in [0.0, 1.0]. Default: 1.0",
    )
    parser.add_argument(
        "--device-name",
        default="laser_projector_stub",
        help="Execution queue device name. Default: laser_projector_stub",
    )
    parser.add_argument(
        "--camera-matrix",
        help="Optional path to camera matrix JSON for future pose estimation",
    )
    parser.add_argument(
        "--dist-coeffs",
        help="Optional path to distortion coefficients JSON for future pose estimation",
    )

    args = parser.parse_args()
    if args.image is not None and "--camera" in sys.argv:
        parser.error("--image 和 --camera 不能同时使用。")
    return args


def ensure_output_dir(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_pipeline_outputs(detection_payload, image_to_save, prefix, args):
    detect_output_dir = ensure_output_dir(DETECT_OUTPUT_DIR)
    wall_output_dir = ensure_output_dir(WALL_OUTPUT_DIR)
    projection_output_dir = ensure_output_dir(PROJECTION_OUTPUT_DIR)
    execution_output_dir = ensure_output_dir(EXECUTION_OUTPUT_DIR)

    save_result_image(image_to_save, prefix, detect_output_dir)
    save_detection_json(detection_payload, prefix, detect_output_dir)

    wall_payload = convert_detection_to_wall_payload(
        detection_payload=detection_payload,
        marker_size_mm=args.marker_size_mm,
        origin_marker_id=args.origin_marker_id,
        origin_mode=args.origin,
    )
    print_wall_results(wall_payload)
    save_wall_json_with_prefix(wall_payload, prefix, wall_output_dir)

    projection_payload = convert_wall_to_projection_payload(
        wall_payload=wall_payload,
        target_type=args.target_type,
        marker_ids=args.target_marker_ids,
        label_prefix=args.label_prefix,
    )
    print_projection_targets(projection_payload)
    save_projection_json_with_prefix(projection_payload, prefix, projection_output_dir)

    if args.export_execution_queue:
        execution_payload = convert_projection_to_execution_payload(
            projection_payload=projection_payload,
            dwell_ms=args.dwell_ms,
            travel_ms=args.travel_ms,
            settle_ms=args.settle_ms,
            repeat=args.repeat,
            laser_power=args.laser_power,
            device_name=args.device_name,
        )
        print_execution_summary(execution_payload)
        save_execution_json_with_prefix(
            execution_payload, prefix, execution_output_dir
        )


def run_image_mode(args):
    image_path = validate_image_path(args.image)
    dictionary = get_aruco_dictionary(args.dict)
    pose_inputs = get_pose_inputs(args)
    image = load_image(image_path)

    corners, ids, _ = detect_markers(image, dictionary)
    raw_results = collect_marker_results(corners, ids)
    effective_results = filter_and_enrich_results(raw_results, args, pose_inputs)
    annotated_image = annotate_detection_result(image, effective_results)
    detection_payload = build_detection_payload(
        results=effective_results,
        frame_shape=image.shape,
        dictionary_name=args.dict,
        source=str(image_path),
        marker_size_mm=args.marker_size_mm,
        camera_matrix=pose_inputs["camera_matrix"],
        dist_coeffs=pose_inputs["dist_coeffs"],
    )

    print(format_results_text(effective_results))
    export_pipeline_outputs(
        detection_payload=detection_payload,
        image_to_save=annotated_image,
        prefix=image_path.stem,
        args=args,
    )

    print("[INFO] 按 'q' 退出窗口，按 's' 重新保存当前结果。")
    while True:
        cv2.imshow(WINDOW_NAME, annotated_image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit.")
            break
        if key == ord("s"):
            export_pipeline_outputs(
                detection_payload=detection_payload,
                image_to_save=annotated_image,
                prefix=image_path.stem,
                args=args,
            )

    cv2.destroyAllWindows()


def run_camera_mode(args):
    dictionary = get_aruco_dictionary(args.dict)
    pose_inputs = get_pose_inputs(args)

    cap = open_camera(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
    )
    debug_info = get_camera_debug_info(cap, args.camera)
    print(f"requested camera index = {debug_info['requested_camera_index']}")
    print(f"cap.isOpened() = {debug_info['opened']}")
    print(f"actual width = {debug_info['actual_width']}")
    print(f"actual height = {debug_info['actual_height']}")
    print(f"actual fourcc = {debug_info['actual_fourcc']}")

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.camera}")

    tracker = StableMarkerTracker(
        min_stable_frames=args.min_stable_frames,
        max_center_jump_px=args.stable_center_threshold_px,
    )
    frame_count = 0
    start_time = time.time()
    last_result_text = None

    print("[INFO] 实时 pipeline 已启动，按 'q' 退出，按 's' 导出当前整套结果。")

    try:
        while True:
            ret, frame = read_bgr_frame(cap)
            if not ret or frame is None:
                print("[ERROR] 读取摄像头帧失败")
                break

            frame_count += 1
            corners, ids, _ = detect_markers(frame, dictionary)
            raw_results = collect_marker_results(corners, ids)
            filtered_results = filter_and_enrich_results(raw_results, args, pose_inputs)
            stable_results, events = tracker.update(filtered_results)

            for event in events:
                print(event)

            stable_origin_result = None
            if args.origin_marker_id is None and stable_results:
                stable_origin_result = stable_results[0]
            elif args.origin_marker_id is not None:
                stable_origin_result = next(
                    (result for result in stable_results if result["id"] == args.origin_marker_id),
                    None,
                )

            if stable_origin_result is not None and args.origin_marker_id is None:
                args.origin_marker_id = stable_origin_result["id"]

            if stable_origin_result is not None:
                result_text = format_results_text(
                    stable_results,
                    title="[INFO] 当前稳定检测结果",
                )
                if result_text != last_result_text:
                    print(result_text)
                    print(
                        f"[INFO] origin marker 已稳定: ID {stable_origin_result['id']}"
                    )
                    last_result_text = result_text

            annotated_frame = annotate_detection_result(frame, filtered_results)
            display_frame = draw_live_info(
                annotated_frame.copy(),
                frame_count,
                start_time,
                stable_results=stable_results,
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] Quit.")
                break

            if key == ord("s"):
                if stable_origin_result is None:
                    print("[INFO] origin marker 尚未稳定，跳过当前导出。")
                    continue

                detection_payload = build_detection_payload(
                    results=stable_results,
                    frame_shape=frame.shape,
                    dictionary_name=args.dict,
                    source=f"camera:{args.camera}",
                    marker_size_mm=args.marker_size_mm,
                    camera_matrix=pose_inputs["camera_matrix"],
                    dist_coeffs=pose_inputs["dist_coeffs"],
                )
                try:
                    export_pipeline_outputs(
                        detection_payload=detection_payload,
                        image_to_save=display_frame,
                        prefix="camera",
                        args=args,
                    )
                except Exception as exc:
                    print(f"[WARN] 当前帧导出失败: {exc}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released, windows closed.")


def main():
    args = parse_args()

    try:
        if args.image is not None:
            run_image_mode(args)
        else:
            run_camera_mode(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
