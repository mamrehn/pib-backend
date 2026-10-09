import sys
import types

from ros_packages.camera.oak_d_lite.parsed_detections import (
    translate_detection,
    translate_detections,
)


def _point(x, y, z=0.0, name="", **extra):
    return types.SimpleNamespace(
        imageCoordinates=types.SimpleNamespace(x=x, y=y, z=z),
        labelName=name,
        **extra,
    )


def test_translates_label_score_box_keypoints_and_scalars_to_ros_contract(monkeypatch):
    datatypes = types.ModuleType("datatypes")
    datatypes_msg = types.ModuleType("datatypes.msg")

    class Detection:
        pass

    datatypes_msg.Detection = Detection
    datatypes.msg = datatypes_msg
    monkeypatch.setitem(sys.modules, "datatypes", datatypes)
    monkeypatch.setitem(sys.modules, "datatypes.msg", datatypes_msg)

    box = types.SimpleNamespace(
        center=types.SimpleNamespace(x=0.5, y=0.5),
        size=types.SimpleNamespace(width=0.5, height=0.25),
    )
    parsed = types.SimpleNamespace(
        label=0,
        labelName="",
        confidence=0.875,
        getBoundingBox=lambda: box,
        getKeypoints=lambda: [
            _point(0.25, 0.5, name="left_eye"),
            _point(0.75, 0.25),
        ],
        scalar_names=["yaw_deg"],
        scalar_values=[12.5],
    )

    detection = translate_detection(parsed, ("Face",), 640, 480)

    assert detection.label == "Face"
    assert detection.score == 0.875
    assert (
        detection.x_min,
        detection.y_min,
        detection.x_max,
        detection.y_max,
    ) == (160, 180, 480, 300)
    assert detection.keypoint_names == ["left_eye", "landmark_1"]
    assert detection.keypoint_x == [160.0, 480.0]
    assert detection.keypoint_y == [240.0, 120.0]
    assert detection.keypoint_z == [0.0, 0.0]
    # These keypoints carry no confidence at all, so there are no scores.
    assert detection.keypoint_score == []
    assert detection.scalar_names == ["yaw_deg"]
    assert detection.scalar_values == [12.5]


def test_translates_two_coco_classes_to_pixel_boxes_without_keypoints(monkeypatch):
    datatypes = types.ModuleType("datatypes")
    datatypes_msg = types.ModuleType("datatypes.msg")

    class Detection:
        pass

    datatypes_msg.Detection = Detection
    datatypes.msg = datatypes_msg
    monkeypatch.setitem(sys.modules, "datatypes", datatypes)
    monkeypatch.setitem(sys.modules, "datatypes.msg", datatypes_msg)

    person = types.SimpleNamespace(
        label=0,
        labelName="",
        confidence=0.91,
        getBoundingBox=lambda: types.SimpleNamespace(
            center=types.SimpleNamespace(x=0.25, y=0.25),
            size=types.SimpleNamespace(width=0.5, height=0.5),
        ),
        scalar_names=(),
        scalar_values=(),
    )
    bicycle = types.SimpleNamespace(
        label=1,
        labelName="",
        confidence=0.42,
        getBoundingBox=lambda: types.SimpleNamespace(
            center=types.SimpleNamespace(x=0.75, y=0.75),
            size=types.SimpleNamespace(width=0.5, height=0.5),
        ),
        scalar_names=(),
        scalar_values=(),
    )

    detections = translate_detections(
        types.SimpleNamespace(detections=[person, bicycle]),
        ("person", "bicycle"),
        640,
        640,
    )

    assert [item.label for item in detections] == ["person", "bicycle"]
    assert [item.score for item in detections] == [0.91, 0.42]
    assert (
        detections[0].x_min,
        detections[0].y_min,
        detections[0].x_max,
        detections[0].y_max,
    ) == (0, 0, 320, 320)
    assert (
        detections[1].x_min,
        detections[1].y_min,
        detections[1].x_max,
        detections[1].y_max,
    ) == (320, 320, 640, 640)
    assert detections[0].keypoint_names == []
    assert detections[1].keypoint_names == []


def _stub_detection_message(monkeypatch):
    datatypes = types.ModuleType("datatypes")
    datatypes_msg = types.ModuleType("datatypes.msg")

    class Detection:
        pass

    datatypes_msg.Detection = Detection
    datatypes.msg = datatypes_msg
    monkeypatch.setitem(sys.modules, "datatypes", datatypes)
    monkeypatch.setitem(sys.modules, "datatypes.msg", datatypes_msg)


def _person(*points):
    box = types.SimpleNamespace(
        center=types.SimpleNamespace(x=0.5, y=0.5),
        size=types.SimpleNamespace(width=0.5, height=0.5),
    )
    return types.SimpleNamespace(
        label=0,
        labelName="person",
        confidence=0.9,
        getBoundingBox=lambda: box,
        getKeypoints=lambda: list(points),
    )


def test_keypoint_confidences_are_published_as_scores(monkeypatch):
    _stub_detection_message(monkeypatch)
    parsed = _person(
        _point(0.1, 0.1, name="nose", confidence=0.95),
        _point(0.2, 0.2, name="left_ankle", confidence=0.0),
    )

    detection = translate_detection(parsed, ("person",), 100, 100)

    assert detection.keypoint_score == [0.95, 0.0]


def test_a_keypoint_without_confidence_empties_the_scores(monkeypatch):
    # DepthAI marks a missing confidence with -1; scores for only some points
    # would be misread by index, so there are none at all.
    _stub_detection_message(monkeypatch)
    parsed = _person(
        _point(0.1, 0.1, name="nose", confidence=0.95),
        _point(0.2, 0.2, name="left_eye", confidence=-1.0),
    )

    detection = translate_detection(parsed, ("person",), 100, 100)

    assert detection.keypoint_score == []


def test_a_score_correction_is_applied_and_clamped(monkeypatch):
    _stub_detection_message(monkeypatch)
    parsed = _person(
        _point(0.1, 0.1, name="nose", confidence=0.5),
        _point(0.2, 0.2, name="left_eye", confidence=0.25),
    )

    [detection] = translate_detections(
        types.SimpleNamespace(detections=[parsed]),
        ("person",),
        100,
        100,
        keypoint_score=lambda score: 4 * score - 1,
    )

    assert detection.keypoint_score == [1.0, 0.0]
