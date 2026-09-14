from typing import Any, Tuple

from sqlalchemy import inspect

from app.app import db, app
from model.assistant_model import AssistantModel
from model.bricklet_model import Bricklet
from model.bricklet_pin_model import BrickletPin
from model.camera_settings_model import CameraSettings
from model.chat_message_model import ChatMessage
from model.chat_model import Chat
from model.motor_model import Motor
from model.personality_model import Personality
from model.program_model import Program
from model.pose_model import Pose
from model.motor_position_model import MotorPosition
from default_pose_constants import (
    STARTUP_POSITIONS,
    CALIBRATION_POSITIONS,
    STARTUP_POSE_NAME,
    CALIBRATION_POSE_NAME,
)
from model.button_program_model import ButtonProgram


def _populate_db() -> None:
    _upsert_bricklet_data()
    _create_camera_data()
    _create_program_data()
    _create_chat_data_and_assistant()
    _create_default_poses()
    _create_button_program_data()
    db.session.commit()


@app.cli.command("seed_db")
def seed_db() -> None:
    if not _is_empty_db():
        print("Seeding database failed - database already contains data.")
        return
    _populate_db()
    print("Seeded the database with default data.")


@app.cli.command("reset_db")
def reset_db() -> None:
    """Re-apply the default data over an existing database.

    Unlike seed_db this does not bail out when rows are already present: every
    step below either inserts what is missing or updates what is there, so a
    robot whose motors have been mis-calibrated can be put back to the shipped
    defaults without dropping the database (and losing chats, programs and
    poses) first.
    """
    if not _is_empty_db():
        print("Warning: Database already contains data.")
    _populate_db()
    print("Reset the database with default data.")


def _is_empty_db() -> bool:
    inspector = inspect(db.engine)

    for table in inspector.get_table_names():
        if table == "alembic_version":
            continue
        table_class = db.Model.metadata.tables.get(table)
        if table_class is not None:
            count = db.session.query(table_class).count()
            if count > 0:
                return False
    return True


# bricklet_number -> type, as shipped.
DEFAULT_BRICKLETS = {
    1: "Servo Bricklet",
    2: "Servo Bricklet",
    3: "Servo Bricklet",
    4: "Solid State Relay Bricklet",
    5: "RGB LED Button Bricklet",
    6: "RGB LED Button Bricklet",
    7: "RGB LED Button Bricklet",
}


def _upsert_bricklet_data() -> None:
    data = _get_motor_list()
    motor_settings = {
        "pulse_width_min": 700,
        "pulse_width_max": 2500,
        "rotation_range_min": -9000,
        "rotation_range_max": 9000,
        "velocity": 16000,
        "acceleration": 10000,
        "deceleration": 5000,
        "period": 19500,
        "turned_on": True,
        "visible": True,
        "invert": False,
    }

    for item in data:
        motor = Motor.query.filter_by(name=item["name"]).first()
        is_new = motor is None
        if is_new:
            motor = Motor(name=item["name"], **motor_settings)
        else:
            for key, value in motor_settings.items():
                setattr(motor, key, value)

        if motor.name == "tilt_forward_motor":
            motor.rotation_range_min = -4500
            motor.rotation_range_max = 4500
        # modify all fingers
        elif motor.name.endswith("stretch") or "thumb" in motor.name:
            motor.pulse_width_min = 750
            motor.velocity = 100000
            motor.acceleration = 50000
            motor.deceleration = 50000
        # reduce upper arm rotation speed
        elif motor.name in ["upper_arm_left_rotation", "upper_arm_right_rotation"]:
            motor.velocity = 10000

        if is_new:
            db.session.add(motor)
        db.session.flush()

        # Re-create the pin assignment so a reset also repairs wrong pins.
        for stale_pin in BrickletPin.query.filter_by(motor_id=motor.id).all():
            db.session.delete(stale_pin)
        db.session.flush()

        bricklet_pins: [Tuple[int, int]] = item["bricklet_pins"]
        for bricklet_pin in bricklet_pins:
            bricklet_id, pin = bricklet_pin

            invert = False
            db.session.add(
                BrickletPin(
                    motor_id=motor.id, bricklet_id=bricklet_id, pin=pin, invert=invert
                )
            )
        db.session.flush()

    for bricklet_number, bricklet_type in DEFAULT_BRICKLETS.items():
        bricklet = Bricklet.query.filter_by(bricklet_number=bricklet_number).first()
        if bricklet is None:
            db.session.add(
                Bricklet(bricklet_number=bricklet_number, type=bricklet_type)
            )
        else:
            bricklet.type = bricklet_type
    db.session.flush()


def _create_button_program_data():
    cerebra_prog = Program.query.filter_by(name="toggle_cerebra_fullscreen").first()
    cerebra_prog_id = cerebra_prog.id if cerebra_prog else None

    # bricklet_number -> program assigned to that button by default.
    default_button_programs = {
        5: None,
        6: None,
        7: cerebra_prog_id,
    }

    for bricklet_number, program_id in default_button_programs.items():
        # button_program.bricklet_id is a foreign key onto bricklet.id, which
        # only coincides with bricklet_number on a freshly seeded database.
        bricklet = Bricklet.query.filter_by(bricklet_number=bricklet_number).first()
        if bricklet is None:
            continue

        button_program = ButtonProgram.query.filter_by(bricklet_id=bricklet.id).first()
        if button_program is None:
            db.session.add(
                ButtonProgram(bricklet_id=bricklet.id, program_id=program_id)
            )
        elif program_id is not None and button_program.program_id is None:
            # Only fill in a missing assignment; never clobber a user's own.
            button_program.program_id = program_id
    db.session.flush()


def _create_camera_data() -> None:
    if CameraSettings.query.first():
        return
    camera_settings = CameraSettings(
        resolution="SD", refresh_rate=0.1, quality_factor=80, res_x=640, res_y=480
    )
    db.session.add(camera_settings)
    db.session.flush()


def _create_program_data() -> None:
    default_programs = [
        (
            "hello_world",
            _get_example_program(),
            "e1d46e2a-935e-4e2b-b2f9-0856af4257c5",
        ),
        (
            "toggle_cerebra_fullscreen",
            '<xml xmlns="https://developers.google.com/blockly/xml"><block type="toggle_cerebra_fullscreen" id="cerebra_toggle" x="10" y="10"></block></xml>',
            "c3r3br4-f-u-l-l-s-c-r-e-e-n-001",
        ),
    ]

    for name, code_visual, program_number in default_programs:
        if Program.query.filter_by(name=name).first():
            continue
        db.session.add(
            Program(
                name=name,
                code_visual=code_visual,
                program_number=program_number,
            )
        )
    db.session.flush()


def _create_chat_data_and_assistant() -> None:
    # (visual_name, api_name, has_image_support)
    default_assistants = [
        ("GPT-4o [Text]", "gpt-4o", False),
        ("GPT-4o [Vision]", "gpt-4o", True),
        ("GPT-3.5 [Text]", "gpt-3.5-turbo", False),
        (
            "Claude 3 Sonnet [Vision]",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            True,
        ),
        ("Gemini 3.5 Flash", "gemini-3.5-flash", False),
        ("Hermes Agent (selbstlernend)", "hermes-agent", True),
    ]

    for visual_name, api_name, has_image_support in default_assistants:
        assistant = AssistantModel.query.filter_by(visual_name=visual_name).first()
        if assistant is None:
            db.session.add(
                AssistantModel(
                    visual_name=visual_name,
                    api_name=api_name,
                    has_image_support=has_image_support,
                )
            )
        else:
            assistant.api_name = api_name
            assistant.has_image_support = has_image_support
    db.session.flush()

    claude = AssistantModel.query.filter_by(
        visual_name="Claude 3 Sonnet [Vision]"
    ).first()
    gpt4o1 = AssistantModel.query.filter_by(visual_name="GPT-4o [Vision]").first()

    # (name, personality_id, gender, pause_threshold, message_history, model)
    default_personalities = [
        ("Eva", "8f73b580-927e-41c2-98ac-e5df070e7288", "Female", 0.8, 5, claude),
        ("Thomas", "8b310f95-92cd-4512-b42a-d3fe29c4bb8a", "Male", 1.0, 15, gpt4o1),
    ]

    for (
        name,
        personality_id,
        gender,
        pause_threshold,
        message_history,
        assistant_model,
    ) in default_personalities:
        if Personality.query.filter_by(personality_id=personality_id).first():
            continue
        db.session.add(
            Personality(
                name=name,
                personality_id=personality_id,
                gender=gender,
                pause_threshold=pause_threshold,
                message_history=message_history,
                assistant_model_id=(
                    assistant_model.id if assistant_model is not None else None
                ),
                stt_engine="local_whisper",
            )
        )
    db.session.flush()

    default_chats = [
        (
            "b4f01552-0c09-401c-8fde-fda753fb0261",
            "Nuernberg",
            "8f73b580-927e-41c2-98ac-e5df070e7288",
        ),
        (
            "ee3e80f9-c8f7-48c2-9f15-449ba9bbe4ab",
            "Home-Office",
            "8b310f95-92cd-4512-b42a-d3fe29c4bb8a",
        ),
    ]

    for chat_id, topic, personality_id in default_chats:
        if Chat.query.filter_by(chat_id=chat_id).first():
            continue
        db.session.add(
            Chat(chat_id=chat_id, topic=topic, personality_id=personality_id)
        )
    db.session.flush()

    default_messages = [
        (
            "539ed3e6-9e3d-11ee-8c90-0242ac120002",
            True,
            "hello pib!",
            "b4f01552-0c09-401c-8fde-fda753fb0261",
        ),
        (
            "0a080706-9e3e-11ee-8c90-0242ac120002",
            False,
            "hello user!",
            "b4f01552-0c09-401c-8fde-fda753fb0261",
        ),
    ]

    for message_id, is_user, content, chat_id in default_messages:
        if ChatMessage.query.filter_by(message_id=message_id).first():
            continue
        db.session.add(
            ChatMessage(
                message_id=message_id,
                is_user=is_user,
                content=content,
                chat_id=chat_id,
            )
        )
    db.session.flush()


def _create_default_poses() -> None:
    startup_pose = Pose.query.filter_by(name=STARTUP_POSE_NAME).first()
    if not startup_pose:
        startup_pose = Pose(name=STARTUP_POSE_NAME, deletable=False)
        db.session.add(startup_pose)
        db.session.flush()

    calibration_pose = Pose.query.filter_by(name=CALIBRATION_POSE_NAME).first()
    if not calibration_pose:
        calibration_pose = Pose(name=CALIBRATION_POSE_NAME, deletable=False)
        db.session.add(calibration_pose)
        db.session.flush()

    # Clear existing positions for these poses to ensure fresh data on reset.
    # Deleted through the ORM rather than as a bulk query: SQLite hands the
    # freed rowids straight back to the replacements inserted below, and a bulk
    # delete would leave the old objects in the identity map under those same
    # primary keys.
    stale_positions = MotorPosition.query.filter(
        MotorPosition.pose_id.in_([startup_pose.id, calibration_pose.id])
    ).all()
    for stale_position in stale_positions:
        db.session.delete(stale_position)
    db.session.flush()

    motors = _get_motor_list()

    startup_positions = [
        MotorPosition(
            position=STARTUP_POSITIONS.get(motor["name"], 0),
            motor_name=motor["name"],
            pose_id=startup_pose.id,
        )
        for motor in motors
    ]

    calibration_positions = [
        MotorPosition(
            position=CALIBRATION_POSITIONS.get(motor["name"], 0),
            motor_name=motor["name"],
            pose_id=calibration_pose.id,
        )
        for motor in motors
    ]

    db.session.add_all(startup_positions + calibration_positions)
    db.session.flush()


def _get_motor_list() -> [dict[str, Any]]:
    name: str = "name"
    bricklet_pins: str = "bricklet_pins"

    return [
        {name: "turn_head_motor", bricklet_pins: [(2, 4)]},
        {name: "tilt_forward_motor", bricklet_pins: [(2, 5)]},
        {name: "upper_arm_left_rotation", bricklet_pins: [(3, 9)]},
        {name: "elbow_left", bricklet_pins: [(3, 8)]},
        {name: "lower_arm_left_rotation", bricklet_pins: [(3, 7)]},
        {name: "shoulder_vertical_left", bricklet_pins: [(2, 9)]},
        {name: "shoulder_horizontal_left", bricklet_pins: [(2, 8)]},
        {name: "upper_arm_right_rotation", bricklet_pins: [(1, 9)]},
        {name: "elbow_right", bricklet_pins: [(1, 8)]},
        {name: "lower_arm_right_rotation", bricklet_pins: [(1, 7)]},
        {name: "shoulder_vertical_right", bricklet_pins: [(2, 1)]},
        {name: "shoulder_horizontal_right", bricklet_pins: [(2, 0)]},
        {name: "thumb_right_opposition", bricklet_pins: [(1, 0)]},
        {name: "thumb_right_stretch", bricklet_pins: [(1, 1)]},
        {name: "index_right_stretch", bricklet_pins: [(1, 2)]},
        {name: "middle_right_stretch", bricklet_pins: [(1, 3)]},
        {name: "ring_right_stretch", bricklet_pins: [(1, 4)]},
        {name: "pinky_right_stretch", bricklet_pins: [(1, 5)]},
        {name: "thumb_left_opposition", bricklet_pins: [(3, 0)]},
        {name: "thumb_left_stretch", bricklet_pins: [(3, 1)]},
        {name: "index_left_stretch", bricklet_pins: [(3, 2)]},
        {name: "middle_left_stretch", bricklet_pins: [(3, 3)]},
        {name: "ring_left_stretch", bricklet_pins: [(3, 4)]},
        {name: "pinky_left_stretch", bricklet_pins: [(3, 5)]},
        {name: "wrist_left", bricklet_pins: [(3, 6)]},
        {name: "wrist_right", bricklet_pins: [(1, 6)]},
    ]


def _get_example_program() -> str:
    return """{"blocks":{"languageVersion":0,"blocks":[{"type":"text_print","id":"QWplsQn`*28S!rmDws$4","x":315,"y":279,"inputs":{"TEXT":{"shadow":{"type":"text","id":"`{AWS~jvKQo-ve^M@z-(","fields":{"TEXT":"hello world"}}}}}]}}"""
