# record.py
#
# 리더암 텔레옵으로 팔로워암을 움직이면서, 그 (관측, 행동) 쌍과 카메라 영상을
# 에피소드 단위로 기록한다. lerobot의 lerobot_record.py 핵심 루프를 그대로 따르되
# LeRobotDataset의 청크/비디오 인코딩 등 대규모 최적화는 걷어냈다.
#
# 프레임 규약(lerobot과 동일): 매 프레임마다
#   observation.state = 팔로워의 "현재" 위치 (행동을 걸기 전)
#   action             = 리더의 "지금" 위치 (팔로워에게 이번 프레임 목표로 전달됨)
# 이렇게 저장해야 (state_t, action_t) -> state_{t+1} 형태로 모방학습에 바로 쓸 수 있다.
#
# 실행 (repo 루트에서):
#   conda run -n lerobot python -m record.record \
#       --leader-port /dev/ttyACM1 --leader-name leader \
#       --follower-port /dev/ttyACM0 --follower-name follower \
#       --camera front=0 --camera wrist=2 \
#       --root ~/so101_data/pick_cube --task "pick up the red cube" \
#       --num-episodes 20 --episode-time 15 --reset-time 5 \
#       --push-to-hub --repo-id myuser/so101-pick-cube

import argparse
import time

import rerun as rr

from calibration.calibration_io import load_calibration
from motor_setup.feetech_bus import FeetechBus
from motor_setup.sts3215_table import DEFAULT_BAUDRATE, OPERATING_MODE, SO101_MOTOR_IDS, TORQUE_ENABLE
from teleoperate.normalization import normalize, unnormalize
from teleoperate.teleoperate import read_positions, write_goal_positions

from .camera import Camera
from .dataset_writer import DatasetWriter
from .hub_upload import push_dataset_to_hub


def log_frame_to_rerun(state: dict, action: dict, images: dict) -> None:
    for name, val in state.items():
        rr.log(f"state/{name}", rr.Scalars(val))
    for name, val in action.items():
        rr.log(f"action/{name}", rr.Scalars(val))
    for cam, img in images.items():
        rr.log(f"camera/{cam}", rr.Image(img))


def run_episode(
    leader: FeetechBus,
    follower: FeetechBus,
    leader_cal: dict,
    follower_cal: dict,
    cameras: dict[str, Camera],
    fps: int,
    duration_s: float,
    writer: DatasetWriter | None,
    recording: bool,
) -> None:
    period = 1.0 / fps
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        loop_start = time.perf_counter()

        follower_state_norm = normalize(read_positions(follower), follower_cal)
        leader_action_norm = normalize(read_positions(leader), leader_cal)

        follower_goal_raw = unnormalize(leader_action_norm, follower_cal)
        write_goal_positions(follower, follower_goal_raw)

        images = {name: cam.read() for name, cam in cameras.items()}

        log_frame_to_rerun(follower_state_norm, leader_action_norm, images)
        if recording and writer is not None:
            writer.add_frame(
                state=follower_state_norm,
                action=leader_action_norm,
                images=images,
                timestamp=loop_start - start,
            )

        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, period - elapsed))


def record(
    leader_port: str,
    follower_port: str,
    leader_name: str,
    follower_name: str,
    cameras_spec: dict[str, int],
    root: str,
    task: str,
    num_episodes: int,
    episode_time_s: float,
    reset_time_s: float,
    fps: int,
    push_to_hub: bool,
    repo_id: str | None,
    private: bool,
) -> None:
    rr.init("so101_record", spawn=True)

    leader_cal = load_calibration(leader_name)
    follower_cal = load_calibration(follower_name)
    joint_names = list(SO101_MOTOR_IDS)

    leader = FeetechBus(leader_port)
    follower = FeetechBus(follower_port)
    leader.connect()
    follower.connect()
    leader.set_baudrate(DEFAULT_BAUDRATE)
    follower.set_baudrate(DEFAULT_BAUDRATE)

    cameras = {name: Camera(index) for name, index in cameras_spec.items()}
    for cam in cameras.values():
        cam.connect()

    writer = DatasetWriter(root, fps, joint_names, list(cameras), task)

    torque_addr, torque_len = TORQUE_ENABLE
    mode_addr, mode_len = OPERATING_MODE
    try:
        for motor_id in SO101_MOTOR_IDS.values():
            leader.write(torque_addr, torque_len, motor_id, 0)
            follower.write(mode_addr, mode_len, motor_id, 0)
            follower.write(torque_addr, torque_len, motor_id, 1)

        episode = 0
        while episode < num_episodes:
            print(f"\n=== 에피소드 {episode + 1}/{num_episodes} 녹화 시작 ({episode_time_s}s) ===")
            writer.start_episode()
            run_episode(
                leader, follower, leader_cal, follower_cal, cameras, fps, episode_time_s, writer, True
            )
            saved_dir = writer.save_episode()
            print(f"저장됨: {saved_dir}")
            episode += 1

            if episode < num_episodes:
                print(f"환경을 리셋하세요 ({reset_time_s}s, 녹화되지 않음)...")
                run_episode(
                    leader, follower, leader_cal, follower_cal, cameras, fps, reset_time_s, None, False
                )
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        for motor_id in SO101_MOTOR_IDS.values():
            follower.write(torque_addr, torque_len, motor_id, 0)
        leader.disconnect()
        follower.disconnect()
        for cam in cameras.values():
            cam.disconnect()

    print(f"\n총 {writer.num_episodes}개 에피소드 저장 완료: {root}")

    if push_to_hub:
        if not repo_id:
            raise ValueError("--push-to-hub 사용 시 --repo-id가 필요합니다.")
        url = push_dataset_to_hub(root, repo_id, private=private)
        print(f"HuggingFace Hub 업로드 완료: {url}")


def _parse_camera_args(camera_args: list[str]) -> dict[str, int]:
    cameras = {}
    for spec in camera_args:
        name, index = spec.split("=")
        cameras[name] = int(index)
    return cameras


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--leader-name", required=True)
    parser.add_argument("--follower-name", required=True)
    parser.add_argument(
        "--camera", action="append", default=[], metavar="NAME=INDEX", help="예: --camera front=0"
    )
    parser.add_argument("--root", required=True, help="로컬 저장 경로")
    parser.add_argument("--task", required=True, help="이 데이터셋이 수행하는 태스크 설명")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-time", type=float, default=15.0)
    parser.add_argument("--reset-time", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--push-to-hub", action="store_true", help="녹화 후 HuggingFace Hub에도 업로드")
    parser.add_argument("--repo-id", default=None, help="예: myuser/so101-pick-cube")
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()

    record(
        leader_port=args.leader_port,
        follower_port=args.follower_port,
        leader_name=args.leader_name,
        follower_name=args.follower_name,
        cameras_spec=_parse_camera_args(args.camera),
        root=args.root,
        task=args.task,
        num_episodes=args.num_episodes,
        episode_time_s=args.episode_time,
        reset_time_s=args.reset_time,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private,
    )
