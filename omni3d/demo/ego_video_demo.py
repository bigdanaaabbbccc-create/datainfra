#!/usr/bin/env python3
"""Run Cube R-CNN on EgoRaw undistorted videos with matching calibration.

Example (run with the project's Python environment and an accessible GPU):
  python demo/ego_video_demo.py \
    --input-dir /mnt/workspace/ego_demo_preview/cups \
                /mnt/workspace/ego_demo_preview/storage_box \
    --output-dir output/ego_demo --start 5 --duration 30 --fps 5

Each input directory contains head_left_camera_undistorted.mp4 and
head_left_camera_params.json, either directly or under head_left_camera/.
Outputs are overlay.mp4, scene.mp4, predictions.jsonl, preview JPGs, and
summary.json. Predictions remain in each frame's camera coordinates; the
source world-to-camera pose is preserved without claiming world alignment
or object tracking. The original demo and model code are not changed.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def sample_indices(frame_count, source_fps, start, duration, output_fps):
    """Uniform output times mapped to nearest source frame (zero based)."""
    if not all(math.isfinite(x) for x in (source_fps, start, duration, output_fps)):
        raise ValueError("Timing parameters must be finite")
    if source_fps <= 0 or not 0 < output_fps <= source_fps:
        raise ValueError("Require 0 < output fps <= source fps")
    if start < 0 or duration <= 0 or frame_count <= 0:
        raise ValueError("Invalid start, duration, or video frame count")
    end = min(start + duration, frame_count / source_fps)
    indices = []
    for i in range(max(0, int(math.ceil((end - start) * output_fps)))):
        timestamp = start + i / output_fps
        frame = int(math.floor(timestamp * source_fps + 0.5))
        if timestamp < end and frame < frame_count:
            indices.append(frame)
    if not indices:
        raise ValueError("Requested interval contains no frames")
    if len(set(indices)) != len(indices):
        raise ValueError("Sampling would repeat source frames")
    return indices


def load_sample(directory, args):
    directory = Path(directory).resolve()
    camera_dir = directory / "head_left_camera"
    if not camera_dir.is_dir():
        camera_dir = directory
    video = camera_dir / "head_left_camera_undistorted.mp4"
    params_path = camera_dir / "head_left_camera_params.json"
    params = read_json(params_path)
    intrinsics = params["undistorted_intrinsics"]
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Invalid undistorted camera intrinsics")
    if params.get("undistorted_distortion_model", "none") != "none":
        raise ValueError("This adapter requires an undistorted pinhole video")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video: {}".format(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    indices = sample_indices(frame_count, fps, args.start, args.duration, args.fps)
    poses = {int(p["frame"]): p for p in params["frames"]}
    # These EgoRaw files use one-based pose frame numbers, not zero-based.
    if len(poses) != len(params["frames"]) or set(poses) != set(range(1, frame_count + 1)):
        raise ValueError("Pose frame IDs must match video frames 1..N exactly")
    for index in indices:
        pose = poses[index + 1]
        rotation = np.asarray(pose["R_w2c"], dtype=float)
        translation = np.asarray(pose["t_w2c"], dtype=float)
        if (rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()):
            raise ValueError("Invalid source camera pose at frame {}".format(index + 1))
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        devices = read_json(metadata_path).get("capture_devices", {}).get("devices", [])
        for device in devices:
            if device.get("device_type") == "head_left_camera":
                resolution = device.get("resolution", {})
                if resolution != {"width": width, "height": height}:
                    raise ValueError("Video resolution differs from camera metadata")
    return dict(name=directory.name, video=video, params_path=params_path, K=K,
                poses=poses, fps=fps, frame_count=frame_count, width=width,
                height=height, indices=indices)


def decode_frames(sample):
    cap = cv2.VideoCapture(str(sample["video"]))
    try:
        first = sample["indices"][0]
        if first and not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise RuntimeError("Video decoder cannot seek to start frame")
        current = first
        for index in sample["indices"]:
            while current < index:
                if not cap.grab():
                    raise RuntimeError("Decode failed before frame {}".format(index))
                current += 1
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Decode failed at frame {}".format(index))
            current += 1
            yield index, frame
    finally:
        cap.release()


def build_predictor(args):
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import transforms as T
    from cubercnn.config import get_cfg_defaults
    from cubercnn.modeling.proposal_generator import RPNWithIgnore  # noqa: F401
    from cubercnn.modeling.roi_heads import ROIHeads3D  # noqa: F401
    from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa: F401
    from cubercnn.modeling.meta_arch import build_model
    from cubercnn import util

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is unavailable. Run in the GPU-enabled DSW environment.")
    torch.set_num_threads(args.cpu_threads)
    def local_path(path):
        if path.startswith(util.CubeRCNNHandler.PREFIX):
            return util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, path)
        return path

    config_path = local_path(args.config_file)
    category_path = args.category_meta or str(Path(config_path).parent / "category_meta.json")
    # Resolve metadata separately if the config was obtained from the model zoo.
    if not Path(category_path).is_file() and not args.category_meta:
        category_path = local_path(args.config_file.rsplit("/", 1)[0] + "/category_meta.json")
    classes = read_json(category_path)["thing_classes"]
    cfg = get_cfg()
    get_cfg_defaults(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = local_path(args.weights)
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    resize = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
    return model, resize, classes, cfg


def infer_frame(model, resize, classes, cfg, frame, K, threshold):
    import torch

    model_image = frame[:, :, ::-1] if cfg.INPUT.FORMAT == "RGB" else frame
    transformed = resize.get_transform(model_image).apply_image(model_image)
    inputs = dict(image=torch.as_tensor(np.ascontiguousarray(transformed.transpose(2, 0, 1))).cuda(),
                  height=frame.shape[0], width=frame.shape[1], K=K)
    with torch.no_grad():
        instances = model([inputs])[0]["instances"].to("cpu")
    records = []
    rejected = 0
    for i in range(len(instances)):
        score = float(instances.scores[i])
        if not math.isfinite(score):
            rejected += 1
            continue
        if score < threshold:
            continue
        record = dict(category_id=int(instances.pred_classes[i]), score=score,
                      bbox2D_xyxy=instances.pred_boxes.tensor[i].tolist(),
                      center_cam=instances.pred_center_cam[i].tolist(),
                      dimensions_whl=instances.pred_dimensions[i].tolist(),
                      R_cam=instances.pred_pose[i].tolist(),
                      bbox3D_cam=instances.pred_bbox3D[i].tolist())
        arrays = [np.asarray(record[k]) for k in
                  ("bbox2D_xyxy", "center_cam", "dimensions_whl", "R_cam", "bbox3D_cam")]
        if (not all(np.isfinite(a).all() for a in arrays)
                or min(record["dimensions_whl"]) <= 0 or record["center_cam"][2] <= 0):
            rejected += 1
            continue
        record["category"] = classes[record["category_id"]]
        records.append(record)
    return records, rejected


def render_frame(frame, K, records, args):
    import torch
    from cubercnn import util, vis

    # Inference and stored K retain original resolution. Only rendering is resized.
    width = min(args.output_width, frame.shape[1])
    width -= width % 2
    height = int(round(frame.shape[0] * width / frame.shape[1] / 2)) * 2
    rendered = cv2.resize(frame, (width, height))
    render_K = K.copy()
    render_K[0] *= width / frame.shape[1]
    render_K[1] *= height / frame.shape[0]
    scene = np.full((args.scene_size, args.scene_size, 3), 245, dtype=np.uint8)
    if records:
        meshes, labels = [], []
        for record in records:
            color = [c / 255.0 for c in util.get_color(record["category_id"])]
            meshes.append(util.mesh_cuboid(record["center_cam"] + record["dimensions_whl"],
                                           record["R_cam"], color=color))
            labels.append("{} {:.2f}".format(record["category"], record["score"]))
        with torch.no_grad():
            rendered, novel, _ = vis.draw_scene_view(
                rendered, render_K, meshes, text=labels, scale=args.scene_size,
                blend_weight=0.5, blend_weight_overlay=0.85)
        scene = cv2.resize(novel, (args.scene_size, args.scene_size))
    else:
        cv2.putText(scene, "No detections", (20, args.scene_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return np.clip(rendered, 0, 255).astype(np.uint8), np.clip(scene, 0, 255).astype(np.uint8)


def open_writer(path, fps, frame):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (frame.shape[1], frame.shape[0]))
    if not writer.isOpened():
        raise RuntimeError("Cannot initialize MP4 encoder: {}".format(path))
    return writer


def verify_video(path, expected_count, expected_fps):
    cap = cv2.VideoCapture(str(path))
    try:
        count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        if count != expected_count or abs(fps - expected_fps) > 0.001:
            raise RuntimeError("Output video verification failed: {} ({} frames, {} fps)".format(
                path, count, fps))
    finally:
        cap.release()


def run_sample(sample, args, model, resize, classes, cfg):
    out = Path(args.output_dir).resolve() / sample["name"]
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    counts = Counter()
    frames_with_detections = rejected_total = written = 0
    writers = []
    preview_ids = {0, len(sample["indices"]) // 2, len(sample["indices"]) - 1}
    try:
        with open(out / "predictions.jsonl", "w", encoding="utf-8") as handle:
            for output_index, (source_index, frame) in enumerate(decode_frames(sample)):
                records, rejected = infer_frame(model, resize, classes, cfg, frame,
                                                 sample["K"], args.threshold)
                overlay, scene = render_frame(frame, sample["K"], records, args)
                label = "frame {} | {:.2f}s | {} objects".format(
                    source_index + 1, source_index / sample["fps"], len(records))
                for image in (overlay, scene):
                    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (245, 245, 245), -1)
                    cv2.putText(image, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (20, 20, 20), 1, cv2.LINE_AA)
                if not writers:
                    writers.append(open_writer(out / "overlay.mp4", args.fps, overlay))
                    writers.append(open_writer(out / "scene.mp4", args.fps, scene))
                writers[0].write(overlay)
                writers[1].write(scene)
                row = dict(source_frame_index=source_index, source_frame_number=source_index + 1,
                           timestamp_s=source_index / sample["fps"], output_frame_index=output_index,
                           output_timestamp_s=output_index / args.fps, K=sample["K"].tolist(),
                           source_pose=sample["poses"][source_index + 1],
                           coordinate_system="camera", units="meters", detections=records,
                           rejected_invalid_predictions=rejected)
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                counts.update(record["category"] for record in records)
                frames_with_detections += bool(records)
                rejected_total += rejected
                written += 1
                if output_index in preview_ids:
                    for suffix, image in (("boxes", overlay), ("novel", scene)):
                        path = out / "frame_{:06d}_{}.jpg".format(source_index + 1, suffix)
                        if not cv2.imwrite(str(path), image):
                            raise RuntimeError("Cannot save preview: {}".format(path))
                if written == 1 or written % 10 == 0 or written == len(sample["indices"]):
                    print("{}: {}/{} frames, {} detections, {:.1f}s".format(
                        sample["name"], written, len(sample["indices"]), sum(counts.values()),
                        time.time() - started), flush=True)
    finally:
        for writer in writers:
            writer.release()
    for filename in ("overlay.mp4", "scene.mp4"):
        verify_video(out / filename, written, args.fps)
    summary = dict(status="complete", source_video=str(sample["video"]),
                   source_camera_params=str(sample["params_path"]), source_fps=sample["fps"],
                   source_frame_count=sample["frame_count"],
                   source_resolution=[sample["width"], sample["height"]],
                   K=sample["K"].tolist(), requested_start_s=args.start,
                   requested_duration_s=args.duration, output_fps=args.fps,
                   output_frame_count=written, output_duration_s=written / args.fps,
                   frames_with_detections=frames_with_detections,
                   detection_counts_by_class=dict(counts), threshold=args.threshold,
                   rejected_invalid_predictions=rejected_total, elapsed_s=time.time() - started,
                   config_file=args.config_file, weights=args.weights, video_codec="mp4v",
                   coordinate_system="per-frame camera", units="meters",
                   scene_view="Per-frame auto-fit novel view; not a fused world map",
                   source_pose_note="Copied as provided; world conventions and alignment not validated",
                   tracking=False, videos_verified_by_full_decode=True)
    write_json(out / "summary.json", summary)
    print("Saved {}".format(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-file", default="cubercnn://omni3d/cubercnn_DLA34_FPN.yaml")
    parser.add_argument("--weights", default="cubercnn://omni3d/cubercnn_DLA34_FPN.pth")
    parser.add_argument("--category-meta", default=None)
    parser.add_argument("--start", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--output-width", type=int, default=960)
    parser.add_argument("--scene-size", type=int, default=640)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--inspect-only", action="store_true", help="Check inputs without loading the model or writing outputs")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    if args.output_width < 64 or args.scene_size < 64 or args.scene_size % 2 or args.cpu_threads < 1:
        parser.error("Require output width >= 64, even scene size >= 64, and positive cpu threads")
    samples = [load_sample(directory, args) for directory in args.input_dir]
    if len({sample["name"] for sample in samples}) != len(samples):
        parser.error("Input directory basenames must be unique")
    for sample in samples:
        print("{}: {}x{}, {} fps, {} source frames -> {} sampled frames; K={}".format(
            sample["name"], sample["width"], sample["height"], sample["fps"],
            sample["frame_count"], len(sample["indices"]), sample["K"].tolist()), flush=True)
        destination = Path(args.output_dir) / sample["name"]
        if not args.inspect_only and destination.exists() and any(destination.iterdir()):
            parser.error("Output already contains files; choose a new output directory: {}".format(destination))
    if args.inspect_only:
        return
    model, resize, classes, cfg = build_predictor(args)
    for sample in samples:
        run_sample(sample, args, model, resize, classes, cfg)


if __name__ == "__main__":
    main()
