# FreeBrain 진행 상황

## 프로젝트 개요
OpenManipulator-X 자율 학습 브레인 (Piaget 발달 단계 기반)
- ROS 2 Humble + MuJoCo 3.5.0 + JAX + Brax
- 15주 계획, Stage 0~3 핵심 범위
- 전체 계획: `plan.md`

---

## 완료된 작업

### Week 1 Day 1: 워크스페이스 부트스트랩 (commit `2ea8f7e`)
- 12개 colcon 패키지 생성 및 빌드 성공
- `freebrain_msgs`: SafetyStatus, DevelopmentalState, DetectedObject 등 메시지/서비스/액션 정의
- 빌드 순서: Tier 0 (msgs) → Tier 1 (description, safety) → ... → Tier 5 (bringup)

### Week 1 Day 2: MJCF 로봇 모델 (commit `9e1762d`)
- URDF → MJCF 변환: `src/freebrain_description/mjcf/open_manipulator_x.xml`
- FK 검증: 홈포즈 EE = (0.286, 0, 0.2045)
- 5 actuators: position control, kp=20, forcerange ±4.1Nm (arm) / ±1.5Nm (gripper)
- 16 sensors: 5 pos + 4 vel + 4 torque + 2 touch + 1 ee_pos(3D)
- Collision: primitive shapes, Visual: mesh STL
- Gripper mimic via equality constraint

### Week 1 Day 3: Tabletop Scene (commit `c0a72a1`)
- `src/freebrain_description/mjcf/tabletop_scene.xml`
- Table 0.5x0.6x0.3m + 3 objects (cube/cylinder/sphere) with freejoints
- Overhead camera (128x128, fov=60) + front camera
- 3 object position sensors (framepos)

### Week 2 Day 1-2: `freebrain_safety` 안전 래퍼 (42 tests pass)
3계층 구조: Pure Python core → JAX layer → ROS 2 node

**파일 구조:**
```
src/freebrain_safety/freebrain_safety/
├── config.py              # JointLimits, SafetyConfig, StagePreset (stage 0-3)
├── limits.py              # check_joint_limits/velocity/torque/workspace → SafetyCheckResult
├── collision_detector.py  # 토크 스파이크 이동평균 충돌 감지
├── safety_filter.py       # SafetyFilter: 클램프 + 속도제한 + 충돌 홀드
├── jax_safety.py          # JaxSafetyParams (NamedTuple pytree), jax_clip/check/cost (@jit)
├── ros_node.py            # SafetyNode: /joint_states → /safety_status @100Hz
└── __init__.py            # 공개 API
config/safety_params.yaml  # ROS 2 파라미터
test/
├── test_limits.py         # 22 tests
├── test_safety_filter.py  # 8 tests
├── test_jax_safety.py     # 10 tests (JAX JIT 검증)
└── test_fuzz.py           # 1000회 랜덤 퍼즈, false-negative 0건
```

**핵심 설계:**
- 관절 소프트 한계: HW 한계에서 5° (0.0873 rad) 버퍼
- Stage 프리셋: velocity_scale (0.10/0.30/0.50/0.70), workspace_scale, torque_scale
- 충돌 감지: 토크 이동평균 (윈도우 10 = 100ms @100Hz), 스파이크 임계값 [1.5, 1.5, 1.2, 1.0] Nm
- JAX: NamedTuple pytree로 JIT 호환, 배치 입력 지원

### Week 2 Day 3: `freebrain_motor` 모터 제어 (20 tests pass)

**파일 구조:**
```
src/freebrain_motor/freebrain_motor/
├── kinematics.py          # FK, numeric Jacobian, damped pseudo-inverse IK
├── ros2_control_client.py # Ros2ControlClient: trajectory, gripper, torque, dry_run
├── motor_node.py          # MotorNode: /joint_states → FK → /ee_position @50Hz
└── __init__.py            # 공개 API
test/
├── test_kinematics.py     # 14 tests (FK 홈포즈, IK roundtrip 등)
└── test_ros2_control_client.py  # 6 tests (dry_run 모드)
```

**핵심 설계:**
- FK: 4x4 homogeneous transform chain, Rodrigues rotation
- IK: damped pseudo-inverse (damping=0.01), max 100 iterations, tol=1mm
- OM_X_CONFIG 상수: joint_origins, joint_axes, ee_offset (MJCF 모델에서 도출)
- MotorNode: safety_status 구독 → 위반 시 명령 거부, EE 위치 PointStamped 발행

---

## 다음 작업: Week 2 Day 4-5

### `freebrain_sim/mjx_env.py` — MJX 병렬 시뮬레이션 환경

**사전 요구사항 (laptop에서 설치):**
```bash
pip install mujoco-mjx
pip install -U "jax[cuda12]"
```

**구현할 내용:**
1. `mjx_env.py`: JAX 기반 MJX 병렬 환경
   - 관찰: joint_pos[4] + joint_vel[4] + ee_pos[3] = 11차원
   - 행동: 4 arm joint position commands
   - MJX `mjx.put_model()` / `mjx.put_data()` / `mjx.step()` 사용
2. 도메인 랜덤화: 질량 ±20%, 마찰 ±30%, PD 게인 변동
3. 병렬 목표: 512 안정화 → 1024 기본 → 2048+ 스트레스 테스트
4. GPU 벤치마크: RTX 4070 (12GB VRAM)

**참조 파일:**
- 기존 RL env: `~/Projects/ros2_project/.../rl_env.py`
- MJCF scene: `src/freebrain_description/mjcf/tabletop_scene.xml`
- MJCF robot: `src/freebrain_description/mjcf/open_manipulator_x.xml`

### 이후 계획 (Week 3~)
- Week 3 Day 1-2: Stage 0 — MJX에서 관절 스윕, 자기 모델 JSON
- Week 3 Day 3-5: Stage 1 — RND + PPO 운동 옹알이 (MJX 1024 병렬)
- Week 4: Sim-to-Real 전이 + 실제 HW 배포

---

## 로봇 파라미터 요약

| 항목 | 값 |
|------|-----|
| 로봇 | OpenManipulator-X |
| Arm 관절 | 4 revolute (joint1-4) |
| Gripper | 1 prismatic (mimic 양쪽) |
| Arm 토크 | ±4.1 Nm (XM430-W350) |
| Gripper 토크 | ±1.5 Nm (XL430-W250) |
| 최대 속도 | 4.8 rad/s (~46 rpm) |
| EE 도달 거리 | ~0.286m |
| 제어 주기 | 100Hz |

## 환경 정보
- Ubuntu 22.04, ROS 2 Humble
- Python 3.10, PyTorch 2.5.1+cu121
- JAX 0.6.2 (현재 CPU only → CUDA 설치 필요)
- MuJoCo 3.5.0 (MJX 미설치 → `mujoco-mjx` 설치 필요)
- GPU: RTX 4070 (12GB VRAM)
