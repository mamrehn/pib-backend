"""Unit tests for the ai_cam_topics camera node additions.

Covers the compatibility contract that lets the DepthAI v3 AI/IMU node keep
serving the legacy upstream topics (``camera_topic``, ``face_center``, the
``*_topic`` control subscriptions) alongside the new ``camera/*`` namespace.
"""

import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ros_packages/camera/oak_d_lite")
    ),
)

depthai = pytest.importorskip("depthai")
rclpy = pytest.importorskip("rclpy")

try:
    import datatypes.srv as _datatypes_srv
except ImportError:
    _datatypes = types.ModuleType("datatypes")
    _datatypes_srv = types.ModuleType("datatypes.srv")
    sys.modules["datatypes"] = _datatypes
    sys.modules["datatypes.srv"] = _datatypes_srv

    class _DummySrv:
        class Request:
            pass

        class Response:
            pass

    _datatypes_srv.GetCameraImage = _DummySrv
    _datatypes_srv.GetDepthFrame = _DummySrv
    _datatypes_srv.GetDistanceAtPx = _DummySrv
    _datatypes_srv.SwitchModel = _DummySrv
else:
    for _name in ("GetDepthFrame", "GetDistanceAtPx", "SwitchModel"):
        if not hasattr(_datatypes_srv, _name):

            class _DummySrv:
                class Request:
                    pass

                class Response:
                    pass

            setattr(_datatypes_srv, _name, _DummySrv)

import numpy as np

from ros_packages.camera.oak_d_lite.stereo import AVAILABLE_MODELS, CameraNode


def _make_node():
    with patch.object(CameraNode, "init_pipeline", return_value=True):
        node = CameraNode()
    node.publish_face_center = MagicMock()
    return node


def _set_counts(node, legacy=0, rgb=0, binary=0):
    node.publisher_ = MagicMock()
    node.publisher_.get_subscription_count.return_value = legacy
    node.rgb_pub = MagicMock()
    node.rgb_pub.get_subscription_count.return_value = rgb
    node.camera_image_pub = MagicMock()
    node.camera_image_pub.get_subscription_count.return_value = binary
    return node


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestColourTopicFanout(unittest.TestCase):
    """One JPEG encode must feed every subscribed colour topic."""

    def test_no_encode_when_nothing_subscribed(self, mock_exists, mock_dai, mock_casc):
        node = _set_counts(_make_node())
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with patch(
            "ros_packages.camera.oak_d_lite.stereo.cv2.imencode"
        ) as mock_imencode:
            node._publish_color_frame(frame)
            mock_imencode.assert_not_called()

        node.publisher_.publish.assert_not_called()
        node.rgb_pub.publish.assert_not_called()
        node.camera_image_pub.publish.assert_not_called()

    def test_legacy_subscriber_alone_gets_base64(self, mock_exists, mock_dai, mock_c):
        node = _set_counts(_make_node(), legacy=1)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        node._publish_color_frame(frame)

        node.publisher_.publish.assert_called_once()
        self.assertIsInstance(node.publisher_.publish.call_args[0][0].data, str)
        node.rgb_pub.publish.assert_not_called()
        node.camera_image_pub.publish.assert_not_called()

    def test_binary_subscriber_alone_skips_base64(self, mock_exists, mock_dai, mock_c):
        node = _set_counts(_make_node(), binary=1)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        node._publish_color_frame(frame)

        node.camera_image_pub.publish.assert_called_once()
        published = node.camera_image_pub.publish.call_args[0][0]
        self.assertEqual(published.format, "jpeg")
        self.assertIsInstance(published.data, bytes)
        node.publisher_.publish.assert_not_called()
        node.rgb_pub.publish.assert_not_called()

    def test_all_three_share_a_single_encode(self, mock_exists, mock_dai, mock_casc):
        node = _set_counts(_make_node(), legacy=1, rgb=1, binary=1)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        real_imencode = __import__("cv2").imencode
        with patch(
            "ros_packages.camera.oak_d_lite.stereo.cv2.imencode",
            side_effect=real_imencode,
        ) as mock_imencode:
            node._publish_color_frame(frame)
            self.assertEqual(mock_imencode.call_count, 1)

        node.publisher_.publish.assert_called_once()
        node.rgb_pub.publish.assert_called_once()
        node.camera_image_pub.publish.assert_called_once()
        # camera_topic and camera/rgb/image carry the identical payload.
        self.assertEqual(
            node.publisher_.publish.call_args[0][0].data,
            node.rgb_pub.publish.call_args[0][0].data,
        )


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestOnDemandPipeline(unittest.TestCase):
    """AI and IMU branches are built only while something subscribes."""

    def _node_with_demand(self, ai=0, imu=0):
        node = _make_node()
        node.ai_pub = MagicMock()
        node.ai_pub.get_subscription_count.return_value = ai
        for pub in ("imu_pub", "imu_accel_pub", "imu_gyro_pub"):
            mock = MagicMock()
            mock.get_subscription_count.return_value = imu
            setattr(node, pub, mock)
        node._restart_pipeline = MagicMock(return_value=True)
        return node

    def test_idle_demand_does_not_rebuild(self, mock_exists, mock_dai, mock_casc):
        node = self._node_with_demand()
        node.check_demand()
        node._restart_pipeline.assert_not_called()

    def test_ai_subscriber_triggers_rebuild(self, mock_exists, mock_dai, mock_casc):
        node = self._node_with_demand(ai=1)
        node.check_demand()
        node._restart_pipeline.assert_called_once_with({"ai": True, "imu": False})

    def test_imu_subscriber_triggers_rebuild(self, mock_exists, mock_dai, mock_casc):
        node = self._node_with_demand(imu=1)
        node.check_demand()
        node._restart_pipeline.assert_called_once_with({"ai": False, "imu": True})

    def test_force_rebuild_flag_is_consumed(self, mock_exists, mock_dai, mock_casc):
        node = self._node_with_demand()
        node._force_rebuild = True

        node.check_demand()
        node._restart_pipeline.assert_called_once()
        self.assertFalse(node._force_rebuild)

        node._restart_pipeline.reset_mock()
        node.check_demand()
        node._restart_pipeline.assert_not_called()


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestModelSwitching(unittest.TestCase):

    def test_unknown_model_is_rejected(self, mock_exists, mock_dai, mock_casc):
        node = _make_node()
        request = MagicMock()
        request.model_name = "not-a-model"
        response = MagicMock()

        node.switch_model_callback(request, response)

        self.assertFalse(response.success)
        self.assertIn("Unknown model", response.message)

    def test_switch_without_ai_subscribers_defers_the_load(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = _make_node()
        node._restart_pipeline = MagicMock(return_value=True)
        target = "person"
        self.assertIn(target, AVAILABLE_MODELS)

        request = MagicMock()
        request.model_name = target
        response = MagicMock()

        node.switch_model_callback(request, response)

        self.assertTrue(response.success)
        self.assertEqual(node.current_model_name, target)
        # No AI subscriber yet, so the device is not disturbed.
        node._restart_pipeline.assert_not_called()

    def test_switching_to_the_active_model_is_a_no_op(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = _make_node()
        node._restart_pipeline = MagicMock(return_value=True)

        request = MagicMock()
        request.model_name = node.current_model_name
        response = MagicMock()

        node.switch_model_callback(request, response)

        self.assertTrue(response.success)
        self.assertIn("Already using", response.message)
        node._restart_pipeline.assert_not_called()


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestImuFrequencyValidation(unittest.TestCase):

    def test_requested_frequency_snaps_to_a_supported_rate(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = _make_node()
        self.assertEqual(node._validate_imu_frequency(100), 100)
        self.assertEqual(node._validate_imu_frequency(120), 100)
        self.assertEqual(node._validate_imu_frequency(1000), 400)
        self.assertEqual(node._validate_imu_frequency(1), 25)


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestFailedFeatureIsNotRetriedEveryTick(unittest.TestCase):
    """A feature that cannot be built must not rebuild the pipeline in a loop.

    init_pipeline drops AI/IMU from the config when they fail, but the
    subscriber that asked for them is still there, so without a latch
    check_demand would see a mismatch and tear colour and depth down again on
    the very next timer tick.
    """

    def _node(self, ai=0, imu=0):
        node = _make_node()
        node.ai_pub = MagicMock()
        node.ai_pub.get_subscription_count.return_value = ai
        for pub in ("imu_pub", "imu_accel_pub", "imu_gyro_pub"):
            mock = MagicMock()
            mock.get_subscription_count.return_value = imu
            setattr(node, pub, mock)
        return node

    def test_failed_ai_model_is_not_rebuilt_until_it_changes(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._node(ai=1)

        def fail_ai(config):
            node.pipeline_config = {"ai": False, "imu": False}
            node._ai_failed_model = node.current_model_name
            node._model_load_error = "boom"
            return True

        node._restart_pipeline = MagicMock(side_effect=fail_ai)

        node.check_demand()
        self.assertEqual(node._restart_pipeline.call_count, 1)

        # Subsequent ticks must leave the device alone.
        for _ in range(5):
            node.check_demand()
        self.assertEqual(node._restart_pipeline.call_count, 1)

        # Selecting a different model clears the latch, so the new one is
        # attempted exactly once; failing again re-latches on that model.
        request = MagicMock()
        request.model_name = "person"
        node.pipeline_config = {"ai": True, "imu": False}
        node.switch_model_callback(request, MagicMock())

        self.assertEqual(node._restart_pipeline.call_count, 2)
        self.assertEqual(node._ai_failed_model, "person")

        for _ in range(5):
            node.check_demand()
        self.assertEqual(node._restart_pipeline.call_count, 2)

    def test_unavailable_imu_is_not_rebuilt_until_reconfigured(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._node(imu=1)

        def fail_imu(config):
            node.pipeline_config = {"ai": False, "imu": False}
            node._imu_unavailable = True
            return True

        node._restart_pipeline = MagicMock(side_effect=fail_imu)

        node.check_demand()
        self.assertEqual(node._restart_pipeline.call_count, 1)

        for _ in range(5):
            node.check_demand()
        self.assertEqual(node._restart_pipeline.call_count, 1)

        # A new frequency is a fresh attempt.
        msg = MagicMock()
        msg.data = json.dumps({"frequency": 200})
        node.imu_config_callback(msg)
        self.assertFalse(node._imu_unavailable)


class TestModelRegistryConsistency(unittest.TestCase):
    """The node registry and the pre-download script must not drift apart."""

    def _script_slugs(self):
        sys.path.insert(
            0,
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")),
        )
        from download_oak_models import DEFAULT_MODELS, MODEL_CATEGORIES, MODEL_SLUGS

        return MODEL_SLUGS, MODEL_CATEGORIES, DEFAULT_MODELS

    def test_slugs_match_the_node_registry(self):
        slugs, _, _ = self._script_slugs()
        node_slugs = {name: info["slug"] for name, info in AVAILABLE_MODELS.items()}
        self.assertEqual(node_slugs, slugs)

    def test_every_model_is_categorised_and_defaults_exist(self):
        slugs, categories, defaults = self._script_slugs()
        self.assertEqual(set(slugs), set(categories))
        self.assertTrue(set(defaults).issubset(set(slugs)))


if __name__ == "__main__":
    unittest.main()
