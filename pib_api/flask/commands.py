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
from default_pose_constants import STARTUP_POSITIONS, CALIBRATION_POSITIONS


def _populate_db() -> None:
    _upsert_bricklet_data()
    _create_camera_data()
    _create_program_data()
    _create_chat_data_and_assistant()
    _create_default_poses()
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
    if not _is_empty_db():
        print("Warning: Database already contains data.")
    _populate_db()
    print("Reset the database with default data.")

def _is_empty_db() -> bool:
    inspector = inspect(db.engine)

    for table in inspector.get_table_names():
        if table == "alembic_version":
            continue
        table_class = db.Model.metadata.tables[table]
        count = db.session.query(table_class).count()
        if count > 0:
            return False
    return True


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
        if not motor:
            motor = Motor(name=item["name"], **motor_settings)
            db.session.add(motor)
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

        db.session.flush()

        # Re-create bricklet pins for this motor
        BrickletPin.query.filter_by(motor_id=motor.id).delete()

        bricklet_pins: [Tuple[int, int]] = item["bricklet_pins"]
        for bricklet_pin_info in bricklet_pins:
            bricklet_id, pin = bricklet_pin_info
            invert = False
            db.session.add(
                BrickletPin(
                    motor_id=motor.id, bricklet_id=bricklet_id, pin=pin, invert=invert
                )
            )
        db.session.flush()

    # Upsert Bricklets
    for i in range(1, 4):
        if not Bricklet.query.filter_by(bricklet_number=i).first():
            db.session.add(Bricklet(bricklet_number=i, type="Servo Bricklet"))
            
    if not Bricklet.query.filter_by(bricklet_number=4).first():
        db.session.add(Bricklet(bricklet_number=4, type="Solid State Relay Bricklet"))
        
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
    if Program.query.filter_by(name="hello_world").first():
        return
    program = Program(
        name="hello_world",
        code_visual=_get_example_program(),
        program_number="e1d46e2a-935e-4e2b-b2f9-0856af4257c5",
    )
    db.session.add(program)
    db.session.flush()


def _create_chat_data_and_assistant() -> None:
    # Assistants
    assistants = [
        ("gpt-4o", "GPT-4o [Vision]", True),
        ("gpt-4o", "GPT-4o [Text]", False),
        ("gpt-3.5-turbo", "GPT-3.5 [Text]", False),
        ("anthropic.claude-3-sonnet-20240229-v1:0", "Claude 3 Sonnet [Vision]", True),
        ("gemini-2.5-flash", "Gemini 2.5 Flash", False),
    ]
    
    for api_name, visual_name, has_image_support in assistants:
        if not AssistantModel.query.filter_by(visual_name=visual_name).first():
            db.session.add(AssistantModel(
                visual_name=visual_name, api_name=api_name, has_image_support=has_image_support
            ))
    db.session.flush()

    claude = AssistantModel.query.filter_by(visual_name="Claude 3 Sonnet [Vision]").first()
    gpt4o1 = AssistantModel.query.filter_by(visual_name="GPT-4o [Vision]").first()
    
    if not claude or not gpt4o1:
        return

    if not Personality.query.filter_by(name="Eva").first():
        p_eva = Personality(
            name="Eva",
            personality_id="8f73b580-927e-41c2-98ac-e5df070e7288",
            gender="Female",
            pause_threshold=0.8,
            message_history=5,
            assistant_model_id=claude.id,
        )
        db.session.add(p_eva)

    if not Personality.query.filter_by(name="Thomas").first():
        p_thomas = Personality(
            name="Thomas",
            personality_id="8b310f95-92cd-4512-b42a-d3fe29c4bb8a",
            gender="Male",
            pause_threshold=1.0,
            message_history=15,
            assistant_model_id=gpt4o1.id,
        )
        db.session.add(p_thomas)
    db.session.flush()

    if not Chat.query.filter_by(topic="Nuernberg").first():
        c1 = Chat(
            chat_id="b4f01552-0c09-401c-8fde-fda753fb0261",
            topic="Nuernberg",
            personality_id="8f73b580-927e-41c2-98ac-e5df070e7288",
        )
        db.session.add(c1)
        
    if not Chat.query.filter_by(topic="Home-Office").first():
        c2 = Chat(
            chat_id="ee3e80f9-c8f7-48c2-9f15-449ba9bbe4ab",
            topic="Home-Office",
            personality_id="8b310f95-92cd-4512-b42a-d3fe29c4bb8a",
        )
        db.session.add(c2)
    db.session.flush()

    if not ChatMessage.query.filter_by(message_id="539ed3e6-9e3d-11ee-8c90-0242ac120002").first():
        m1 = ChatMessage(
            message_id="539ed3e6-9e3d-11ee-8c90-0242ac120002",
            is_user=True,
            content="hello pib!",
            chat_id="b4f01552-0c09-401c-8fde-fda753fb0261",
        )
        db.session.add(m1)
        
    if not ChatMessage.query.filter_by(message_id="0a080706-9e3e-11ee-8c90-0242ac120002").first():
        m2 = ChatMessage(
            message_id="0a080706-9e3e-11ee-8c90-0242ac120002",
            is_user=False,
            content="hello user!",
            chat_id="b4f01552-0c09-401c-8fde-fda753fb0261",
        )
        db.session.add(m2)
    db.session.flush()


def _create_default_poses() -> None:
    startup_pose = Pose.query.filter_by(name="Startup/Resting").first()
    if not startup_pose:
        startup_pose = Pose(name="Startup/Resting", deletable=False)
        db.session.add(startup_pose)
            
    calibration_pose = Pose.query.filter_by(name="Calibration").first()
    if not calibration_pose:
        calibration_pose = Pose(name="Calibration", deletable=False)
        db.session.add(calibration_pose)

    db.session.flush()

    # Clear existing positions for these poses to ensure fresh data on reset
    MotorPosition.query.filter_by(pose_id=startup_pose.pose_id).delete()
    MotorPosition.query.filter_by(pose_id=calibration_pose.pose_id).delete()

    motors = _get_motor_list()

    startup_positions = [
        MotorPosition(
            position=STARTUP_POSITIONS.get(motor["name"], 0),
            motor_name=motor["name"],
            pose_id=startup_pose.pose_id,
        )
        for motor in motors
    ]

    calibration_positions = [
        MotorPosition(
            position=CALIBRATION_POSITIONS.get(motor["name"], 0),
            motor_name=motor["name"],
            pose_id=calibration_pose.pose_id,
        )
        for motor in motors
    ]

    db.session.add_all(startup_positions + calibration_positions)
    db.session.commit()


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
