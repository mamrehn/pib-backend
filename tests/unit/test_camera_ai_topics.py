"""Unit tests for the ai_cam_topics camera node additions.

Covers the compatibility contract that lets the DepthAI v3 AI/IMU node keep
serving the legacy upstream topics (``camera_topic``, ``face_center``, the
``*_topic`` control subscriptions) alongside the new ``camera/*`` namespace.
"""

import json
import os
import sys
import time
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

from ros_packages.camera.oak_d_lite import stereo as stereo_module
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


def _demand_node(ai=0):
    node = _make_node()
    node.ai_pub = MagicMock()
    node.ai_pub.get_subscription_count.return_value = ai
    for pub in ("imu_pub", "imu_accel_pub", "imu_gyro_pub"):
        m = MagicMock()
        m.get_subscription_count.return_value = 0
        setattr(node, pub, m)
    return node


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestFrameWatchdog(unittest.TestCase):
    """DepthAI auto-reconnects a crashed device, hiding a dead camera.

    On the robot the OAK-D Lite crash-looped every ~8 s for 90 minutes while the
    node reported pipeline_ok and published nothing. Health must come from
    frames.
    """

    def _silent_node(self, config, ai=0):
        node = _demand_node(ai=ai)
        node._restart_pipeline = MagicMock(return_value=True)
        node.queue = MagicMock()
        node._pipeline_ok = True
        node.pipeline_config = dict(config)
        stale = time.monotonic() - stereo_module.FRAME_WATCHDOG_SECONDS - 1
        node._pipeline_started_at = stale
        node._last_colour_frame_at = 0.0
        return node

    def test_silent_pipeline_with_depth_drops_depth(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._silent_node({"depth": True, "ai": False, "imu": False})

        node.check_demand()

        self.assertTrue(node._depth_unavailable)
        node._restart_pipeline.assert_called_once_with(
            {"depth": False, "ai": False, "imu": False}
        )

    def test_silent_pipeline_with_ai_unloads_the_model(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = self._silent_node({"depth": False, "ai": True, "imu": False}, ai=1)

        node.check_demand()

        self.assertEqual(node._ai_failed_model, node.current_model_name)
        self.assertFalse(node._depth_unavailable)
        node._restart_pipeline.assert_called_once()

    def test_silent_colour_only_pipeline_is_marked_down_and_rebuilt(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = _demand_node()
        node._restart_pipeline = MagicMock(return_value=True)
        node.queue = MagicMock()
        node._pipeline_ok = True
        node._depth_unavailable = True
        node.pipeline_config = {"depth": False, "ai": False, "imu": False}
        node._pipeline_started_at = (
            time.monotonic() - stereo_module.FRAME_WATCHDOG_SECONDS - 1
        )

        node.check_demand()

        node._restart_pipeline.assert_called_once()

    def test_fresh_frames_keep_the_pipeline(self, mock_exists, mock_dai, mock_c):
        node = self._silent_node({"depth": True, "ai": False, "imu": False})
        node._last_colour_frame_at = time.monotonic()

        for _ in range(5):
            node.check_demand()

        self.assertFalse(node._depth_unavailable)
        node._restart_pipeline.assert_not_called()

    def test_new_pipeline_gets_a_grace_period(self, mock_exists, mock_dai, mock_c):
        node = self._silent_node({"depth": True, "ai": False, "imu": False})
        node._pipeline_started_at = time.monotonic()

        node.check_demand()

        node._restart_pipeline.assert_not_called()

    def test_timer_records_colour_frames(self, mock_exists, mock_dai, mock_casc):
        node = _demand_node()
        node.publish_face_center = MagicMock()
        node._publish_color_frame = MagicMock()
        image = MagicMock()
        image.getCvFrame.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
        node.queue = MagicMock()
        node.queue.tryGet.return_value = image
        before = time.monotonic()

        node.timer_callback()

        self.assertGreaterEqual(node._last_colour_frame_at, before)

    def test_explicit_depth_request_retries_after_a_detected_fault(
        self, mock_exists, mock_dai, mock_casc
    ):
        node = _demand_node()
        node._restart_pipeline = MagicMock(return_value=True)
        node._depth_unavailable = True
        msg = MagicMock()
        msg.data = json.dumps({"depth": True})

        node.camera_config_callback(msg)

        self.assertFalse(node._depth_unavailable)
        node._restart_pipeline.assert_called_once()


@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestPipelineConstructionOnHardware(unittest.TestCase):
    """Calls whose correct form was established on the OAK-D Lite itself."""

    def _bare(self):
        with patch.object(CameraNode, "__init__", lambda self: None):
            node = CameraNode()
        node.get_logger = MagicMock()
        node.queues = {}
        node.preview_width, node.preview_height = 1280, 720
        node.current_model_name = "pose_yolo"
        return node

    def test_aligned_stereo_sets_an_output_size_divisible_by_16(
        self, mock_exists, mock_dai
    ):
        node = self._bare()
        node.pipeline = MagicMock()
        stereo = MagicMock()
        node.pipeline.create.side_effect = lambda kind: (
            stereo if kind is mock_dai.node.StereoDepth else MagicMock()
        )

        node._init_stereo_depth()

        stereo.setOutputSize.assert_called_once_with(*stereo_module.DEPTH_OUTPUT_SIZE)
        width, height = stereo_module.DEPTH_OUTPUT_SIZE
        self.assertEqual(width % 16, 0)
        self.assertAlmostEqual(width / height, 2104 / 1560, places=1)

    def test_parsing_network_is_created_by_the_pipeline(self, mock_exists, mock_dai):
        node = self._bare()
        node.pipeline = MagicMock()
        node._get_model_description = MagicMock(return_value="desc")
        parser_cls = MagicMock(name="ParsingNeuralNetwork")
        camera = MagicMock()

        with patch.object(
            stereo_module, "_depthai_nodes_parser", return_value=parser_cls
        ):
            node._init_ai(camera)

        # build() on the class itself raised TypeError on the robot.
        parser_cls.build.assert_not_called()
        node.pipeline.create.assert_any_call(parser_cls)
        node.pipeline.create.return_value.build.assert_called_once_with(camera, "desc")
        self.assertIn("nn", node.queues)


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestAiResultFormatting(unittest.TestCase):
    """Formatters match depthai-nodes 0.5.2 and native ImgDetections."""

    @staticmethod
    def _point(x, y):
        return types.SimpleNamespace(x=x, y=y)

    def test_lines_use_start_and_end_points(self, mock_exists, mock_casc):
        node = _make_node()
        msg = types.SimpleNamespace(
            lines=[
                types.SimpleNamespace(
                    start_point=self._point(0.1, 0.2),
                    end_point=self._point(0.3, 0.4),
                    confidence=0.9,
                )
            ]
        )

        result = node._format_lines(msg)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["lines"][0]["start"], {"x": 0.1, "y": 0.2})
        self.assertEqual(result["lines"][0]["end"], {"x": 0.3, "y": 0.4})
        self.assertNotIn("error", result)

    def test_predictions_are_regression_values(self, mock_exists, mock_casc):
        node = _make_node()
        msg = types.SimpleNamespace(
            predictions=[types.SimpleNamespace(prediction=v) for v in (0.25, -1.5)]
        )

        self.assertEqual(
            node._format_predictions(msg), {"predictions": [0.25, -1.5], "count": 2}
        )

    def test_keypoints_message_reads_its_keypoints_list(self, mock_exists, mock_c):
        node = _make_node()
        kp = types.SimpleNamespace(
            imageCoordinates=self._point(0.5, 0.6), confidence=0.7
        )
        msg = types.SimpleNamespace(
            keypoints_list=types.SimpleNamespace(getKeypoints=lambda: [kp])
        )

        result = node._format_keypoints(msg)

        self.assertEqual(result["keypoints"], [{"x": 0.5, "y": 0.6, "confidence": 0.7}])

    def test_detections_carry_pose_keypoints(self, mock_exists, mock_casc):
        node = _make_node()
        kp = types.SimpleNamespace(
            imageCoordinates=self._point(0.1, 0.9), confidence=1.0
        )
        det = types.SimpleNamespace(
            label=0,
            confidence=0.8,
            xmin=0.1,
            ymin=0.2,
            xmax=0.3,
            ymax=0.4,
            getKeypoints=lambda: [kp],
        )
        msg = types.SimpleNamespace(detections=[det])

        result = node._format_detections(msg)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["detections"][0]["keypoints"][0]["x"], 0.1)

    def test_mask_mode_encodes_the_segmentation_mask(self, mock_exists, mock_casc):
        node = _make_node()
        node.segmentation_mode = "mask"
        mask = np.array([[255, 255, 0], [0, 1, 1]], dtype=np.uint8)
        msg = types.SimpleNamespace(detections=[], getCvSegmentationMask=lambda: mask)

        result = node._format_detections(msg)

        self.assertEqual(result["mask_rle"]["shape"], [2, 3])
        self.assertEqual(result["mask_rle"]["values"], [255, 0, 1])
        self.assertEqual(result["mask_rle"]["runs"], [2, 2, 2])

    def test_dispatches_depthai_nodes_messages(self, mock_exists, mock_casc):
        node = _make_node()

        class Keypoints:
            keypoints_list = types.SimpleNamespace(getKeypoints=lambda: [])

        class Lines:
            lines = []

        class Predictions:
            predictions = []

        with patch.object(
            stereo_module,
            "_depthai_nodes_messages",
            return_value=(Keypoints, Lines, Predictions, type("C", (), {})),
        ):
            self.assertIn("keypoints", node._format_ai_result(Keypoints(), "", "", {}))
            self.assertIn("lines", node._format_ai_result(Lines(), "", "", {}))
            self.assertIn(
                "predictions", node._format_ai_result(Predictions(), "", "", {})
            )


@patch("ros_packages.camera.oak_d_lite.stereo.dai")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestRegistryModelsBuild(unittest.TestCase):
    """Registry fixes established by running every model on the OAK-D Lite."""

    def _bare(self, model):
        with patch.object(CameraNode, "__init__", lambda self: None):
            node = CameraNode()
        node.get_logger = MagicMock()
        node.queues = {}
        node.pipeline = MagicMock()
        node.current_model_name = model
        node._get_model_description = MagicMock(return_value="desc")
        return node

    def test_scrfd_and_yunet_use_the_parsing_network(self, mock_exists, mock_dai):
        # DetectionNetwork rejects them: no YOLO or SSD detection head.
        for name, info in AVAILABLE_MODELS.items():
            if any(k in info["slug"] for k in ("scrfd", "yunet")):
                self.assertEqual(info["node_type"], "ParsingNeuralNetwork", name)

    def test_multi_head_model_reads_the_synced_outputs(self, mock_exists, mock_dai):
        node = self._bare("gaze")
        parsing_node = MagicMock()
        type(parsing_node).out = property(
            lambda self: (_ for _ in ()).throw(
                RuntimeError("Property out is only available ... 2 heads")
            )
        )
        node.pipeline.create.return_value.build.return_value = parsing_node

        with patch.object(
            stereo_module, "_depthai_nodes_parser", return_value=MagicMock()
        ):
            node._init_ai(MagicMock())

        parsing_node.outputs.createOutputQueue.assert_called_once()
        self.assertIs(
            node.queues["nn"], parsing_node.outputs.createOutputQueue.return_value
        )


@patch("ros_packages.camera.oak_d_lite.stereo.cv2.CascadeClassifier")
@patch("ros_packages.camera.oak_d_lite.stereo.os.path.exists", return_value=True)
class TestMultiHeadFormatting(unittest.TestCase):

    def test_classifications(self, mock_exists, mock_casc):
        node = _make_node()
        msg = types.SimpleNamespace(
            classes=["left", "right"],
            scores=np.array([0.2, 0.8]),
            top_class="right",
            top_score=0.8,
        )

        self.assertEqual(
            node._format_classifications(msg),
            {
                "classes": ["left", "right"],
                "scores": [0.2, 0.8],
                "top_class": "right",
                "top_score": 0.8,
            },
        )

    def test_message_group_formats_each_head(self, mock_exists, mock_casc):
        node = _make_node()

        class Predictions:
            def __init__(self, value):
                self.predictions = [types.SimpleNamespace(prediction=value)]

        class Group:
            def __init__(self, messages):
                self._messages = messages

            def getMessageNames(self):
                return list(self._messages)

            def __getitem__(self, name):
                return self._messages[name]

        fake_dai = types.SimpleNamespace(
            ImgDetections=type("D", (), {}), MessageGroup=Group
        )
        group = Group({"1": Predictions(-0.5), "0": Predictions(0.25)})

        with (
            patch.object(stereo_module, "dai", fake_dai),
            patch.object(
                stereo_module,
                "_depthai_nodes_messages",
                return_value=(
                    type("K", (), {}),
                    type("L", (), {}),
                    Predictions,
                    type("C", (), {}),
                ),
            ),
        ):
            result = node._format_ai_result(group, "gaze", "", {})

        self.assertEqual(list(result["heads"]), ["0", "1"])
        self.assertEqual(result["heads"]["0"]["predictions"], [0.25])
        self.assertEqual(result["heads"]["1"]["predictions"], [-0.5])


class TestKeypointConfidenceSentinel(unittest.TestCase):
    """YuNet and the hand landmarker report confidence -1 (none available)."""

    def test_negative_confidence_is_omitted(self):
        kp = types.SimpleNamespace(
            imageCoordinates=types.SimpleNamespace(x=0.26, y=0.17), confidence=-1.0
        )
        source = types.SimpleNamespace(getKeypoints=lambda: [kp])

        self.assertEqual(stereo_module._keypoints(source), [{"x": 0.26, "y": 0.17}])

    def test_real_confidence_is_kept(self):
        kp = types.SimpleNamespace(
            imageCoordinates=types.SimpleNamespace(x=0.46, y=0.0), confidence=0.506
        )
        source = types.SimpleNamespace(getKeypoints=lambda: [kp])

        self.assertEqual(
            stereo_module._keypoints(source),
            [{"x": 0.46, "y": 0.0, "confidence": 0.506}],
        )


class TestRunLengthEncoding(unittest.TestCase):

    @staticmethod
    def _naive(mask):
        flat = mask.ravel().tolist()
        runs, values = [], []
        for v in flat:
            if values and values[-1] == v:
                runs[-1] += 1
            else:
                values.append(v)
                runs.append(1)
        return {"runs": runs, "values": values, "shape": list(mask.shape)}

    def test_matches_a_reference_encoder(self):
        rng = np.random.default_rng(7)
        for shape in [(1, 1), (4, 5), (288, 512)]:
            mask = rng.integers(0, 3, size=shape).astype(np.uint8)
            mask[: shape[0] // 2] = 255
            self.assertEqual(stereo_module.rle_encode(mask), self._naive(mask))

    def test_empty_mask(self):
        empty = np.zeros((0, 3), dtype=np.uint8)
        self.assertEqual(
            stereo_module.rle_encode(empty), {"runs": [], "values": [], "shape": [0, 3]}
        )


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
