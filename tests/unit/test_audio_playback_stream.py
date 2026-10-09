"""The /audio_playback listener of the voice assistant's audio player.

The node module needs ROS and the voice stack to import, so the listener is
compiled on its own from the source with stand-ins for what it uses.
"""

import array
import ast
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYER = REPO_ROOT / "ros_packages/voice_assistant/voice_assistant/audio_player.py"


def _listener(**namespace):
    tree = ast.parse(PLAYER.read_text(encoding="utf-8"))
    [node_class] = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AudioPlayerNode"
    ]
    [function] = [
        node
        for node in node_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "receive_audio_stream_listener"
    ]
    module = ast.Module(body=[function], type_ignores=[])
    scope = {"array": array, "Int16MultiArray": object, **namespace}
    exec(compile(module, str(PLAYER), "exec"), scope)
    return scope["receive_audio_stream_listener"]


class _Node:
    def __init__(self):
        self.queued = []
        self.errors = []
        self.playback_queue = types.SimpleNamespace(
            put=lambda item, block: self.queued.append((item, block))
        )
        self._order = 7

    def counter_next(self):
        self._order += 1
        return self._order

    def get_logger(self):
        return types.SimpleNamespace(error=self.errors.append)


def test_the_player_subscribes_to_audio_playback():
    source = PLAYER.read_text(encoding="utf-8")
    assert '"audio_playback"' in source
    assert "self.receive_audio_stream_listener" in source


def test_samples_are_queued_as_one_gapless_chunk_in_order():
    encoding = object()
    listener = _listener(
        PlaybackItem=lambda *args: args, SPEECH_ENCODING=encoding
    )
    node = _Node()

    listener(node, types.SimpleNamespace(data=[0, 1, -1, 32767]))

    [((data, item_encoding, pause, order), block)] = node.queued
    assert data == [array.array("h", [0, 1, -1, 32767]).tobytes()]
    assert item_encoding is encoding
    assert pause == 0.0
    assert order == 8
    assert block is True


def test_samples_outside_int16_are_reported_and_dropped():
    listener = _listener(PlaybackItem=lambda *args: args, SPEECH_ENCODING=object())
    node = _Node()

    listener(node, types.SimpleNamespace(data=[40000]))

    assert node.queued == []
    assert len(node.errors) == 1
