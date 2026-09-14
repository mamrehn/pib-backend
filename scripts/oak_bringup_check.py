#!/usr/bin/env python3
"""
OAK-D Lite bring-up check for the pib camera container.

The camera node's unit tests mock DepthAI entirely, so they cannot say whether
the device actually streams. This script answers that on the hardware itself,
one question per test, each in its own process so a native crash or a hang is
reported instead of taking the whole run down:

- does the device enumerate, on which USB speed, with which cameras, and has
  its firmware stored a crash dump;
- does the documented v3 preview path (requestOutput) stream;
- does the ISP output path the camera node uses stream, and at what size;
- does adding StereoDepth, aligned to the colour camera, stall either stream;
- does polling one frame per 0.1 s, as the node's timer does, keep up;
- can a pipeline be rebuilt in-process, both the documented way and the way
  the node currently does it.

Run it inside the ros-camera image with the camera node stopped, because the
OAK-D Lite serves one pipeline at a time:

    docker stop multirepo-ros-camera-1
    docker run --rm --privileged -v /dev:/dev -v "$PWD/scripts:/diag" \\
        --entrypoint python3 ros-camera:latest /diag/oak_bringup_check.py
    docker start multirepo-ros-camera-1
"""

import argparse
import subprocess
import sys
import time

import depthai as dai

SAMPLE_SECONDS = 10.0
TEST_TIMEOUT_SECONDS = 90
DEVICE_SETTLE_SECONDS = 4.0
EXPECTED_SOCKETS = {"CAM_A", "CAM_B", "CAM_C"}


# ---------------------------------------------------------------- helpers


def describe(msg, started):
    try:
        size = f"{msg.getWidth()}x{msg.getHeight()}"
    except Exception:
        size = type(msg).__name__
    return f"{size} after {time.monotonic() - started:.1f}s"


def drain(queues, seconds, poll_period=0.005, one_per_poll=False):
    """Count messages per queue and describe the first one of each."""
    counts = {name: 0 for name in queues}
    first = {}
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        for name, queue in queues.items():
            while True:
                msg = queue.tryGet()
                if msg is None:
                    break
                counts[name] += 1
                first.setdefault(name, describe(msg, started))
                if one_per_poll:
                    break
        time.sleep(poll_period)
    return counts, first


def colour_request_output(pipeline):
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    return {"colour": cam.requestOutput((1280, 720)).createOutputQueue()}


def colour_isp(pipeline):
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    return {"colour": cam.requestIspOutput().createOutputQueue()}


def with_stereo(pipeline, queues):
    """Add StereoDepth exactly as stereo.py's _init_stereo_depth does."""
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    try:
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    except Exception:
        pass
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    left.requestFullResolutionOutput().link(stereo.left)
    right.requestFullResolutionOutput().link(stereo.right)
    queues["depth"] = stereo.depth.createOutputQueue()
    return queues


def sample_fresh_device(build, **drain_kwargs):
    """Open a device, run one pipeline, shut down the documented way."""
    device = dai.Device()
    try:
        pipeline = dai.Pipeline(device)
        queues = build(pipeline)
        pipeline.start()
        counts, first = drain(queues, SAMPLE_SECONDS, **drain_kwargs)
        running = pipeline.isRunning()
        pipeline.stop()
        pipeline.wait()
    finally:
        device.close()
    ok = all(count > 0 for count in counts.values())
    return (
        ok,
        f"frames in {SAMPLE_SECONDS:.0f}s {counts}; first {first}; running={running}",
    )


# ---------------------------------------------------------------- tests


def test_device():
    infos = dai.Device.getAllAvailableDevices()
    if not infos:
        return False, "no OAK device found (check the USB cable, port and power)"

    device = dai.Device()
    try:
        speed = str(device.getUsbSpeed())
        sockets = {str(s).split(".")[-1] for s in device.getConnectedCameras()}
        crashed = device.hasCrashDump()
        crash_excerpt = ""
        if crashed:
            try:
                crash_excerpt = device.getCrashDump().serializeToJson()[:600]
            except Exception as exc:
                crash_excerpt = f"(could not read dump: {exc})"
    finally:
        device.close()

    notes = [f"devices={[repr(i) for i in infos]}", f"usb={speed}"]
    if "HIGH" in speed:
        notes.append("USB2 link: full-resolution streams will be bandwidth-limited")
    notes.append(f"cameras={sorted(sockets)}")
    if crashed:
        notes.append(f"STORED CRASH DUMP: {crash_excerpt}")

    ok = EXPECTED_SOCKETS <= sockets and not crashed
    return ok, "; ".join(notes)


def test_colour_request_output():
    return sample_fresh_device(colour_request_output)


def test_colour_isp():
    return sample_fresh_device(colour_isp)


def test_request_output_with_stereo():
    return sample_fresh_device(lambda p: with_stereo(p, colour_request_output(p)))


def test_node_pipeline():
    return sample_fresh_device(lambda p: with_stereo(p, colour_isp(p)))


def test_node_pipeline_polled_like_timer():
    return sample_fresh_device(
        lambda p: with_stereo(p, colour_isp(p)), poll_period=0.1, one_per_poll=True
    )


def test_rebuild_documented():
    """Two pipelines in turn on one Device, each stopped with stop() + wait()."""
    device = dai.Device()
    results = []
    try:
        for attempt in (1, 2):
            pipeline = dai.Pipeline(device)
            queues = colour_request_output(pipeline)
            pipeline.start()
            counts, _ = drain(queues, 5.0)
            pipeline.stop()
            pipeline.wait()
            results.append(counts["colour"])
            print(f"build {attempt}: {counts}", flush=True)
    finally:
        device.close()
    return all(results), f"colour frames per build {results}"


def test_rebuild_like_node():
    """What stereo.py does today: a new dai.Pipeline() and stop() without wait()."""
    results = []
    for attempt in (1, 2):
        pipeline = dai.Pipeline()
        queues = colour_isp(pipeline)
        pipeline.start()
        counts, _ = drain(queues, 5.0)
        pipeline.stop()
        results.append(counts["colour"])
        print(f"build {attempt}: {counts}", flush=True)
    return True, f"process survived both rebuilds; colour frames per build {results}"


PLAN = [
    ("device", test_device, "enumeration, USB speed, cameras, crash dump"),
    (
        "colour_request_output",
        test_colour_request_output,
        "documented v3 preview path, 1280x720 scaled on-device",
    ),
    ("colour_isp", test_colour_isp, "the ISP output path stereo.py uses"),
    (
        "request_output_with_stereo",
        test_request_output_with_stereo,
        "documented preview path plus StereoDepth aligned to CAM_A",
    ),
    (
        "node_pipeline",
        test_node_pipeline,
        "exactly the node's pipeline: ISP plus StereoDepth",
    ),
    (
        "node_pipeline_polled_like_timer",
        test_node_pipeline_polled_like_timer,
        "the node's pipeline read one frame per 0.1 s, as timer_callback does",
    ),
    (
        "rebuild_documented",
        test_rebuild_documented,
        "two pipelines on one Device, stop() + wait() between them",
    ),
    (
        "rebuild_like_node",
        test_rebuild_like_node,
        "two new dai.Pipeline() objects, stop() without wait(), as stereo.py does",
    ),
]
TESTS = {name: fn for name, fn, _ in PLAN}


# ---------------------------------------------------------------- driver


def run_child(name):
    try:
        ok, detail = TESTS[name]()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    print(f"RESULT {'PASS' if ok else 'FAIL'} {detail}", flush=True)
    return 0


def as_text(data):
    if data is None:
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else data


def interpret(status):
    def failed(name):
        return status.get(name) == "FAIL"

    def passed(name):
        return status.get(name) == "PASS"

    # Later tests build on earlier ones. When a foundation fails, every test
    # after it fails for the same reason, so conclusions drawn from those later
    # failures would be wrong. Stop at the first broken foundation.
    if failed("device"):
        return [
            "Device-level problem: unplug and replug the camera, use a USB3 port "
            "and cable, and rerun. Nothing later in this report can be trusted "
            "until this passes."
        ]
    if failed("colour_request_output"):
        return [
            "Even the simplest documented stream fails, so the fault is below "
            "stereo.py: hardware, power, or a device wedged by earlier crashes. "
            "Replug the camera and rerun. Later results cannot be trusted yet."
        ]

    hints = []
    if failed("colour_isp"):
        hints.append(
            "requestIspOutput() is the fault. stereo.py should request a scaled "
            "output with requestOutput((w, h)) instead."
        )
    if passed("colour_isp") and failed("node_pipeline"):
        hints.append(
            "The ISP path works alone but stalls once StereoDepth is aligned to "
            "it: depth alignment against the ISP frame is the fault."
        )
    if passed("request_output_with_stereo") and failed("node_pipeline"):
        hints.append(
            "StereoDepth works with a scaled colour output but not with the ISP "
            "output: switching the colour output fixes both streams."
        )
    if passed("node_pipeline") and failed("node_pipeline_polled_like_timer"):
        hints.append(
            "The pipeline streams, but one read per 0.1 s cannot keep up with "
            "the default queue: the node must drain its queues."
        )
    if passed("rebuild_documented") and failed("rebuild_like_node"):
        hints.append(
            "In-process rebuilds need stop() + wait() on an explicit Device; "
            "stereo.py's current teardown is the fault."
        )
    if failed("rebuild_documented"):
        # colour_request_output passed above, so this is about the rebuild.
        hints.append(
            "A Device cannot host a second pipeline: rebuilds must close the "
            "Device and open a new one."
        )
    if all(v == "PASS" for v in status.values()):
        hints.append(
            "Everything streams and rebuilds here, so the fault is in stereo.py's "
            "own handling rather than the hardware or the DepthAI calls."
        )
    return hints or ["No single failure pattern matched; send the full output."]


def run_parent():
    print(f"depthai {dai.__version__}, python {sys.version.split()[0]}", flush=True)
    status = {}
    for name, _, description in PLAN:
        print(f"\n=== {name}: {description}", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--test", name],
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_SECONDS,
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, returncode = as_text(exc.stdout), as_text(exc.stderr), None
            timed_out = True

        result = None
        for line in stdout.splitlines():
            if line.startswith("RESULT "):
                result = line
            else:
                print(f"  | {line}", flush=True)

        if timed_out:
            verdict, detail = "FAIL", f"hung for more than {TEST_TIMEOUT_SECONDS}s"
        elif returncode is not None and returncode < 0:
            verdict = "FAIL"
            detail = f"process killed by signal {-returncode} (native crash)"
        elif result is None:
            verdict = "FAIL"
            detail = f"exited {returncode} without a result; stderr: {stderr[-600:]}"
        else:
            _, verdict, detail = result.split(" ", 2)

        status[name] = verdict
        print(f"[{verdict}] {name}: {detail}", flush=True)
        time.sleep(DEVICE_SETTLE_SECONDS)

    print("\n=== summary", flush=True)
    for name, _, _ in PLAN:
        print(f"  {status[name]:4}  {name}", flush=True)
    print("\n=== interpretation", flush=True)
    for hint in interpret(status):
        print(f"  - {hint}", flush=True)
    return 0 if all(v == "PASS" for v in status.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--test", choices=sorted(TESTS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    return run_child(args.test) if args.test else run_parent()


if __name__ == "__main__":
    sys.exit(main())
