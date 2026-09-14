"""Unit tests for the idempotent seeding path and the reset_db command.

seed_db refuses to touch a database that already has rows, so restoring the
shipped defaults used to mean dropping the database and losing every chat,
program and pose with it. reset_db re-applies the defaults in place instead,
which only works if every seeding step upserts rather than inserts.
"""

from __future__ import annotations

from click.testing import CliRunner

from app.app import db
from commands import reset_db
from default_pose_constants import CALIBRATION_POSE_NAME, STARTUP_POSE_NAME
from model.assistant_model import AssistantModel
from model.bricklet_model import Bricklet
from model.bricklet_pin_model import BrickletPin
from model.button_program_model import ButtonProgram
from model.chat_message_model import ChatMessage
from model.chat_model import Chat
from model.motor_model import Motor
from model.motor_position_model import MotorPosition
from model.personality_model import Personality
from model.pose_model import Pose
from model.program_model import Program

_SEEDED_MODELS = (
    AssistantModel,
    Bricklet,
    BrickletPin,
    ButtonProgram,
    Chat,
    ChatMessage,
    Motor,
    MotorPosition,
    Personality,
    Pose,
    Program,
)


def _counts() -> dict[str, int]:
    return {model.__name__: model.query.count() for model in _SEEDED_MODELS}


def _run_reset() -> None:
    result = CliRunner().invoke(reset_db, [])
    if result.exception:
        raise result.exception
    assert result.exit_code == 0, result.output
    db.session.commit()


def test_reset_on_a_seeded_db_does_not_duplicate_rows(app_ctx):
    """The original failure mode: a second seed hit a UNIQUE constraint."""
    before = _counts()

    _run_reset()

    assert _counts() == before


def test_reset_is_stable_across_repeated_runs(app_ctx):
    _run_reset()
    after_first = _counts()

    _run_reset()
    _run_reset()

    assert _counts() == after_first


def test_reset_restores_modified_motor_settings(app_ctx):
    motor = Motor.query.filter_by(name="turn_head_motor").one()
    original_velocity = motor.velocity
    motor.velocity = 1
    motor.turned_on = False
    db.session.commit()

    _run_reset()

    motor = Motor.query.filter_by(name="turn_head_motor").one()
    assert motor.velocity == original_velocity
    assert motor.turned_on is True


def test_reset_restores_default_pose_positions(app_ctx):
    startup_pose = Pose.query.filter_by(name=STARTUP_POSE_NAME).one()
    positions = MotorPosition.query.filter_by(pose_id=startup_pose.id).all()
    assert positions, "startup pose should have motor positions after seeding"
    expected = {p.motor_name: p.position for p in positions}

    for position in positions:
        position.position = 7777
    db.session.commit()

    _run_reset()

    startup_pose = Pose.query.filter_by(name=STARTUP_POSE_NAME).one()
    restored = {
        p.motor_name: p.position
        for p in MotorPosition.query.filter_by(pose_id=startup_pose.id).all()
    }
    assert restored == expected


def test_reset_keeps_user_created_data(app_ctx):
    db.session.add(
        Program(
            name="my_own_program",
            code_visual="<xml/>",
            program_number="user-program-0001",
        )
    )
    db.session.commit()

    _run_reset()

    assert Program.query.filter_by(name="my_own_program").count() == 1


def test_reset_recreates_deleted_default_rows(app_ctx):
    Pose.query.filter_by(name=CALIBRATION_POSE_NAME).delete()
    AssistantModel.query.filter_by(api_name="hermes-agent").delete()
    db.session.commit()

    _run_reset()

    assert Pose.query.filter_by(name=CALIBRATION_POSE_NAME).count() == 1
    assert AssistantModel.query.filter_by(api_name="hermes-agent").count() == 1


def test_button_programs_reference_bricklets_by_primary_key(app_ctx):
    """button_program.bricklet_id is a FK onto bricklet.id, not bricklet_number."""
    button_programs = ButtonProgram.query.all()
    assert button_programs, "button programs should exist after seeding"

    button_bricklet_ids = {
        b.id
        for b in Bricklet.query.filter(Bricklet.type == "RGB LED Button Bricklet").all()
    }
    for button_program in button_programs:
        assert button_program.bricklet_id in button_bricklet_ids


def test_reset_fills_a_cleared_button_assignment(app_ctx):
    cerebra = Program.query.filter_by(name="toggle_cerebra_fullscreen").one()
    assigned = ButtonProgram.query.filter_by(program_id=cerebra.id).one()
    assigned.program_id = None
    db.session.commit()

    _run_reset()

    assert ButtonProgram.query.filter_by(program_id=cerebra.id).count() == 1


def test_reset_does_not_clobber_a_custom_button_assignment(app_ctx):
    custom = Program(
        name="my_button_program",
        code_visual="<xml/>",
        program_number="user-button-0001",
    )
    db.session.add(custom)
    db.session.flush()

    button_program = ButtonProgram.query.first()
    button_program.program_id = custom.id
    db.session.commit()

    _run_reset()

    reloaded = db.session.get(ButtonProgram, button_program.id)
    assert reloaded.program_id == custom.id
