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
        node._restart_pipeline.assert_called_once_with(
            {"depth": False, "ai": True, "imu": False}
        )

    def test_imu_subscriber_triggers_rebuild(self, mock_exists, mock_dai, mock_casc):
        node = self._node_with_demand(imu=1)
        node.check_demand()
        node._restart_pipeline.assert_called_once_with(
            {"depth": True, "ai": False, "imu": True}
        )

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
            node.pipeline_config = {"depth": True, "ai": False, "imu": False}
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
        node.pipeline_config = {"depth": False, "ai": True, "imu": False}
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
            node.pipeline_config = {"depth": True, "ai": False, "imu": False}
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


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestDepthAiArbitration(unittest.TestCase):
    """Depth and AI cannot share the OAK-D Lite, so they swap on demand."""

    def _node(self, ai=0):
        node = _make_node()
        node.ai_pub = MagicMock()
        node.ai_pub.get_subscription_count.return_value = ai
        for pub in ("imu_pub", "imu_accel_pub", "imu_gyro_pub"):
            m = MagicMock()
            m.get_subscription_count.return_value = 0
            setattr(node, pub, m)

        def apply(cfg):
            node.pipeline_config = dict(cfg)
            node._pipeline_ok = True
            return True

        node._restart_pipeline = MagicMock(side_effect=apply)
        return node

    def test_depth_is_the_resting_state(self, mock_exists, mock_dai, mock_casc):
        node = self._node()
        node.check_demand()
        self.assertTrue(node.pipeline_config["depth"])
        self.assertFalse(node.pipeline_config["ai"])

    def test_ai_subscription_suspends_depth_and_unsubscribing_restores_it(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._node(ai=1)
        node.check_demand()
        self.assertEqual(
            node.pipeline_config, {"depth": False, "ai": True, "imu": False}
        )

        # switching back on the fly is just the demand going away
        node.ai_pub.get_subscription_count.return_value = 0
        node.check_demand()
        self.assertEqual(
            node.pipeline_config, {"depth": True, "ai": False, "imu": False}
        )
        self.assertEqual(node._restart_pipeline.call_count, 2)

    def test_depth_can_be_disabled_by_operator_override(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._node()
        msg = MagicMock()
        msg.data = json.dumps({"depth": False})
        node.camera_config_callback(msg)

        self.assertFalse(node._depth_enabled)
        self.assertFalse(node.pipeline_config["depth"])

    def test_unavailable_depth_is_not_retried(self, mock_exists, mock_dai, mock_c):
        node = self._node()
        node._depth_unavailable = True
        node.check_demand()
        self.assertFalse(node.pipeline_config["depth"])


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestPipelineStartFailureRecovery(unittest.TestCase):
    """SHAVE exhaustion is only reported at pipeline.start()."""

    def _bare(self, ai=True):
        with patch.object(CameraNode, "__init__", lambda self: None):
            node = CameraNode()
        node.get_logger = MagicMock()
        node.pipeline_config = {"depth": not ai, "ai": ai, "imu": False}
        node.current_model_name = "yolov6n"
        node.current_depth = "stale"
        node._depth_enabled = True
        node._depth_unavailable = False
        node._imu_unavailable = False
        node._model_loading = False
        node._model_load_error = None
        node._ai_failed_model = None
        node._pipeline_ok = False
        node._init_ai = MagicMock()
        node._init_stereo_depth = MagicMock()
        node._init_imu = MagicMock()
        return node

    def test_ai_start_failure_falls_back_to_a_running_pipeline(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._bare()
        pipeline = MagicMock()
        mock_dai.Pipeline.return_value = pipeline
        pipeline.start.side_effect = [
            RuntimeError(
                "NeuralNetwork: Blob compiled for 8 shaves, "
                "but only 6 are available in current configuration."
            ),
            None,
        ]

        ok = node.init_pipeline()

        # the camera must survive a model that cannot be placed
        self.assertTrue(ok)
        self.assertTrue(node._pipeline_ok)
        self.assertEqual(pipeline.start.call_count, 2)
        # the failed model is latched on the outer path, ending the retry loop
        self.assertEqual(node._ai_failed_model, "yolov6n")
        # and depth reclaims the cores the model was going to use
        self.assertEqual(
            node.pipeline_config, {"depth": True, "ai": False, "imu": False}
        )

    def test_total_failure_reports_not_ok_without_a_third_attempt(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._bare()
        pipeline = MagicMock()
        mock_dai.Pipeline.return_value = pipeline
        pipeline.start.side_effect = RuntimeError("device gone")

        ok = node.init_pipeline()

        self.assertFalse(ok)
        self.assertFalse(node._pipeline_ok)
        self.assertEqual(pipeline.start.call_count, 2)
        self.assertIsNone(node.queue)

    def test_stale_depth_is_dropped_when_depth_leaves_the_pipeline(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._bare(ai=False)
        node.pipeline_config = {"depth": False, "ai": False, "imu": False}
        mock_dai.Pipeline.return_value = MagicMock()

        node.init_pipeline()

        self.assertIsNone(node.current_depth)


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestDeadPipelineSelfHeals(unittest.TestCase):
    """A failed pipeline used to stay dead until the container was restarted."""

    def _node(self):
        node = _make_node()
        node.ai_pub = MagicMock()
        node.ai_pub.get_subscription_count.return_value = 0
        for pub in ("imu_pub", "imu_accel_pub", "imu_gyro_pub"):
            m = MagicMock()
            m.get_subscription_count.return_value = 0
            setattr(node, pub, m)
        node._restart_pipeline = MagicMock(return_value=False)
        return node

    def test_down_pipeline_is_retried_even_though_demand_is_unchanged(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._node()
        # exactly the state the old code got stuck in: config matches demand,
        # but nothing is running
        node._pipeline_ok = False
        node._next_recovery_attempt = 0.0
        node.pipeline_config = {"depth": True, "ai": False, "imu": False}

        node.check_demand()
        node._restart_pipeline.assert_called_once()

    def test_retries_are_rate_limited(self, mock_exists, mock_dai, mock_casc):
        node = self._node()
        node._pipeline_ok = False
        node._next_recovery_attempt = 0.0
        node.pipeline_config = {"depth": True, "ai": False, "imu": False}

        node.check_demand()
        for _ in range(20):
            node.check_demand()
        self.assertEqual(node._restart_pipeline.call_count, 1)

        # once the backoff expires it tries again
        node._next_recovery_attempt = 0.0
        node.check_demand()
        self.assertEqual(node._restart_pipeline.call_count, 2)

    def test_healthy_pipeline_is_left_alone(self, mock_exists, mock_dai, mock_c):
        node = self._node()
        node._pipeline_ok = True
        node._next_recovery_attempt = 0.0
        node.pipeline_config = {"depth": True, "ai": False, "imu": False}

        for _ in range(10):
            node.check_demand()
        node._restart_pipeline.assert_not_called()


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
