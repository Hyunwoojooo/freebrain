# FreeBrain

OpenManipulator-X 자율 학습 브레인 — Piaget 발달 단계 기반 로봇 학습 시스템

## 개요

로봇 팔이 영아의 발달 과정처럼 스스로 학습하는 시스템입니다.

```
Stage 0 (반사)     → 관절 스윕, 자기 모델 학습
Stage 1 (운동 옹알이) → 호기심 기반 랜덤 탐색 (RND + PPO)
Stage 2 (물체 발견)  → IMGEP, 밀기/닿기 행동 발견
Stage 3 (LLM 가이드) → 언어 지시 기반 파지 학습
```

## 아키텍처

```
계층 3: Cognitive (LLM 집행부)
    ↓ CognitiveGoal
계층 2: Skills (스킬 라이브러리 + 탐색)
    ↓ JointCommand
계층 1: Motor + Safety (실행 + 안전)
    ↓ ros2_control
    HW / MJX Sim
```

## 기술 스택

| 구성 | 기술 |
|------|------|
| 로봇 | OpenManipulator-X (4-DOF + Gripper) |
| 미들웨어 | ROS 2 Humble |
| 시뮬레이션 | MuJoCo 3.5.0 + MJX (JAX 병렬) |
| RL 학습 | JAX + Brax PPO, RND, IMGEP |
| 인식 | Grounding DINO + SAM2 + Depth Anything V2 |
| 인지 | GPT-4o / Claude / Qwen2.5-VL |
| 메모리 | ChromaDB (에피소드/의미/절차) |

## 패키지 구조

```
src/
├── freebrain_msgs/          # 커스텀 메시지/서비스/액션
├── freebrain_description/   # MJCF 로봇 + 장면 모델
├── freebrain_safety/        # 안전 래퍼 (Python + JAX + ROS 2)
├── freebrain_motor/         # FK/IK + ros2_control 클라이언트
├── freebrain_sim/           # MJX 병렬 시뮬레이션 환경
├── freebrain_perception/    # 비전 파이프라인
├── freebrain_memory/        # 메모리 시스템
├── freebrain_skills/        # 스킬 라이브러리
├── freebrain_exploration/   # 호기심 기반 RL
├── freebrain_cognitive/     # LLM 인지 집행부
├── freebrain_metacog/       # 발달 단계 전환
└── freebrain_bringup/       # Launch 파일
```

## 빌드

```bash
# 의존성
sudo apt install ros-humble-desktop
pip install mujoco mujoco-mjx "jax[cuda12]"

# 빌드
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## 실행

```bash
# 안전 노드
ros2 run freebrain_safety safety_node

# 모터 노드
ros2 run freebrain_motor motor_node

# 모터 노드 (dry_run, HW 없이 테스트)
ros2 run freebrain_motor motor_node --ros-args -p dry_run:=true
```

## 테스트

```bash
# 전체 테스트
python3 -m pytest src/freebrain_safety/test/ -v   # 42 tests
python3 -m pytest src/freebrain_motor/test/ -v     # 20 tests
```

## 진행 상황

자세한 내용은 [PROGRESS.md](PROGRESS.md) 참고.

| Week | 작업 | 상태 |
|------|------|------|
| 1-1 | 워크스페이스 부트스트랩 | Done |
| 1-2 | MJCF 로봇 모델 | Done |
| 1-3 | Tabletop 장면 | Done |
| 2-1~2 | 안전 래퍼 (freebrain_safety) | Done |
| 2-3 | 모터 제어 (freebrain_motor) | Done |
| 2-4~5 | MJX 병렬 환경 (freebrain_sim) | Next |

## 라이선스

Apache-2.0
