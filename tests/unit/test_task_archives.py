import types
from pathlib import Path

import depthai as dai
import pytest

from ros_packages.camera.oak_d_lite.task_archives import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    YOLO26N_MODEL_ID,
    YOLO26N_POSE_MODEL_ID,
    YOLOV6N_COCO_LABELS,
    YOLOV6N_MODEL_ID,
    YUNET_MODEL_ID,
    create_archive,
    labels_for_model,
    yolo26n_archive_config,
    yolo26n_pose_archive_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
YUNET_BLOB = (
    REPO_ROOT / "models/face_detection_yunet_160x120/face_detection_yunet_160x120.blob"
)
YOLOV6N_BLOB = REPO_ROOT / "models/yolov6n_coco_640x640/yolov6n_coco_640x640.blob"
YOLO26N_BLOB = REPO_ROOT / "models/yolo26n_coco_512x288/yolo26n_coco_512x288.blob"
YOLO26N_POSE_BLOB = (
    REPO_ROOT / "models/yolo26n_pose_coco_512x288/yolo26n_pose_coco_512x288.blob"
)


def _require_blob(path: Path) -> None:
    if not path.is_file():
        pytest.skip(
            f"{path.name} is not in git; it ships in the models-2026-09-15 release asset"
        )


def test_real_yunet_blob_builds_archive_from_its_tensor_metadata(tmp_path):
    _require_blob(YUNET_BLOB)
    blob = dai.OpenVINO.Blob(YUNET_BLOB)
    archive = create_archive(YUNET_MODEL_ID, YUNET_BLOB, tmp_path)

    assert isinstance(archive, dai.NNArchive)
    assert (archive.getInputWidth(), archive.getInputHeight()) == (160, 120)
    [head] = list(archive.getConfig().model.heads)
    assert head.parser == "YuNetParser"
    assert set(head.outputs) == set(blob.networkOutputs)
    assert head.metadata.confThreshold == 0.8
    assert head.metadata.iouThreshold == 0.3


def test_model_without_registered_parser_keeps_the_plain_network_path():
    assert create_archive("facemesh_192x192", "/unused.blob") is None


def test_real_yolov6n_blob_builds_archive_from_its_tensor_metadata(tmp_path):
    _require_blob(YOLOV6N_BLOB)
    blob = dai.OpenVINO.Blob(YOLOV6N_BLOB)
    archive = create_archive(YOLOV6N_MODEL_ID, YOLOV6N_BLOB, tmp_path)

    assert isinstance(archive, dai.NNArchive)
    assert (archive.getInputWidth(), archive.getInputHeight()) == (640, 640)
    [model_input] = list(archive.getConfig().model.inputs)
    [blob_input_name] = tuple(blob.networkInputs)
    assert model_input.name == blob_input_name
    archive_outputs = {
        output.name: list(output.shape) for output in archive.getConfig().model.outputs
    }
    blob_outputs = {
        name: list(tensor.dims) for name, tensor in blob.networkOutputs.items()
    }
    assert archive_outputs == blob_outputs
    [head] = list(archive.getConfig().model.heads)
    assert head.parser == "YOLOExtendedParser"
    assert set(head.outputs) == set(blob.networkOutputs)
    assert head.metadata.nClasses == 80
    assert head.metadata.subtype == "yolov6"


def test_labels_for_model_returns_eighty_coco_names_in_model_order():
    labels = labels_for_model(YOLOV6N_MODEL_ID)
    assert labels is YOLOV6N_COCO_LABELS
    assert len(labels) == 80
    assert labels[0] == "person"
    assert labels[2] == "car"
    assert labels[-1] == "toothbrush"


def _archive_matches_blob(archive, blob):
    [model_input] = list(archive.getConfig().model.inputs)
    [blob_input_name] = tuple(blob.networkInputs)
    assert model_input.name == blob_input_name
    archive_outputs = {
        output.name: list(output.shape) for output in archive.getConfig().model.outputs
    }
    blob_outputs = {
        name: list(tensor.dims) for name, tensor in blob.networkOutputs.items()
    }
    assert archive_outputs == blob_outputs


def test_real_yolo26n_blob_builds_archive_from_its_tensor_metadata(tmp_path):
    _require_blob(YOLO26N_BLOB)
    blob = dai.OpenVINO.Blob(YOLO26N_BLOB)
    archive = create_archive(YOLO26N_MODEL_ID, YOLO26N_BLOB, tmp_path)

    assert (archive.getInputWidth(), archive.getInputHeight()) == (512, 288)
    _archive_matches_blob(archive, blob)
    [head] = list(archive.getConfig().model.heads)
    assert head.parser == "YOLOExtendedParser"
    assert head.metadata.nClasses == 80
    # The one-to-many head: three feature maps, decoded with NMS.
    assert head.metadata.subtype == "yolov8"
    assert labels_for_model(YOLO26N_MODEL_ID) is YOLOV6N_COCO_LABELS


def test_real_yolo26n_pose_blob_builds_archive_with_named_keypoints(tmp_path):
    _require_blob(YOLO26N_POSE_BLOB)
    blob = dai.OpenVINO.Blob(YOLO26N_POSE_BLOB)
    archive = create_archive(YOLO26N_POSE_MODEL_ID, YOLO26N_POSE_BLOB, tmp_path)

    assert (archive.getInputWidth(), archive.getInputHeight()) == (512, 288)
    _archive_matches_blob(archive, blob)
    [head] = list(archive.getConfig().model.heads)
    assert head.parser == "YOLOExtendedParser"
    assert head.metadata.subtype == "yolo26"
    assert head.metadata.nKeypoints == 17
    extra = head.metadata.extraParams
    assert extra["keypoint_label_names"] == list(COCO_KEYPOINT_NAMES)
    assert labels_for_model(YOLO26N_POSE_MODEL_ID) == ("person",)


def test_a_blob_with_other_outputs_is_rejected(tmp_path):
    # The YOLOv6n blob under the YOLO26n id is rejected, not mis-parsed.
    _require_blob(YOLOV6N_BLOB)
    with pytest.raises(ValueError, match="expected"):
        create_archive(YOLO26N_MODEL_ID, YOLOV6N_BLOB, tmp_path)


# The archive configs are pure functions of a blob's tensor metadata, so they
# are tested without the blobs (which are a release asset, not in git).


def _fake_blob(inputs, outputs):
    tensor = lambda dims: types.SimpleNamespace(dims=dims)
    return types.SimpleNamespace(
        networkInputs={name: tensor(dims) for name, dims in inputs.items()},
        networkOutputs={name: tensor(dims) for name, dims in outputs.items()},
    )


YOLO26N_TENSORS = (
    {"images": [512, 288, 3, 1]},
    {
        "output3_yolov6r2": [16, 9, 85, 1],
        "output2_yolov6r2": [32, 18, 85, 1],
        "output1_yolov6r2": [64, 36, 85, 1],
    },
)
YOLO26N_POSE_TENSORS = (
    {"images": [512, 288, 3, 1]},
    {"output_yolo26": [6, 3024, 1], "kpt_output": [51, 3024, 1]},
)


def test_yolo26n_config_without_a_blob_parses_three_maps_with_nms():
    config = yolo26n_archive_config(_fake_blob(*YOLO26N_TENSORS))

    model = config["model"]
    assert model["inputs"][0]["shape"] == [1, 3, 288, 512]
    [head] = model["heads"]
    assert head["parser"] == "YOLOExtendedParser"
    assert head["metadata"]["subtype"] == "yolov8"
    assert head["metadata"]["n_classes"] == 80
    assert head["metadata"]["classes"] == list(YOLOV6N_COCO_LABELS)
    # the output order is the one the parser expects, not the blob's dict order
    assert head["outputs"] == ["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"]
    assert [o["name"] for o in model["outputs"]] == head["outputs"]


def test_yolo26n_pose_config_without_a_blob_names_the_keypoints():
    config = yolo26n_pose_archive_config(_fake_blob(*YOLO26N_POSE_TENSORS))

    [head] = config["model"]["heads"]
    meta = head["metadata"]
    assert head["parser"] == "YOLOExtendedParser" and meta["subtype"] == "yolo26"
    assert meta["n_classes"] == 1 and meta["classes"] == ["person"]
    assert meta["n_keypoints"] == 17
    assert meta["keypoint_label_names"] == list(COCO_KEYPOINT_NAMES)
    assert len(COCO_KEYPOINT_NAMES) == 17
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in COCO_SKELETON)
    assert meta["skeleton_edges"] == [list(edge) for edge in COCO_SKELETON]
    assert head["outputs"] == ["output_yolo26", "kpt_output"]


@pytest.mark.parametrize("config_builder,tensors", [
    (yolo26n_archive_config, YOLO26N_POSE_TENSORS),        # pose blob as detector
    (yolo26n_pose_archive_config, YOLO26N_TENSORS),        # detector blob as pose
])
def test_a_blob_with_the_wrong_outputs_is_rejected_without_the_blob_files(
    config_builder, tensors
):
    with pytest.raises(ValueError, match="expected"):
        config_builder(_fake_blob(*tensors))


def test_a_blob_with_another_input_size_is_rejected():
    inputs, outputs = YOLO26N_TENSORS
    with pytest.raises(ValueError, match="input dimensions"):
        yolo26n_archive_config(_fake_blob({"images": [640, 640, 3, 1]}, outputs))


def test_yolo26n_and_yolov6n_share_the_class_order():
    assert labels_for_model(YOLO26N_MODEL_ID) == labels_for_model(YOLOV6N_MODEL_ID)
