# FreeBrain: OpenManipulator-X 자율 학습 브레인 개발 계획

## Context

OpenManipulator-X를 위한 자율 학습 브레인을 구축하는 프로젝트. 피아제의 발달 단계를 모방하여 운동 옹알이 -> 물체 발견 -> 파지 학습 -> 소통적 학습 -> 자율 실험으로 성장하는 로봇 팔 시스템. ROS 2 Humble + 하드웨어 연동은 이미 완료된 상태에서, 3계층 인지 아키텍처(운동 피질 / 스킬 라이브러리 / 인지 집행부)와 4종 메모리 시스템을 개발한다. **15주 핵심 범위는 Stage 0~3 완성**이며, **Stage 4~5는 옵션 확장**으로 분리한다.

---

## 1. ROS 2 워크스페이스 구조 (12개 패키지)

```
/home/joo/Projects/freebrain/
├── src/
│   ├── freebrain_msgs/              # 커스텀 메시지/서비스/액션 정의
│   ├── freebrain_description/       # URDF/MJCF 모델, 메시
│   ├── freebrain_safety/            # [계층1] 안전 래퍼 (Soft safety + Hard safety 정책)
│   ├── freebrain_motor/             # [계층1] 모터 실행 (ros2_control, FK/IK, 정책/폴백 실행)
│   ├── freebrain_sim/               # MuJoCo MJX 시뮬레이션 환경
│   ├── freebrain_perception/        # 비전 파이프라인 (Grounding DINO, SAM2, Depth Anything V2)
│   ├── freebrain_memory/            # 메모리 시스템 (에피소드/의미/절차/공간/작업 메모리)
│   ├── freebrain_skills/            # [계층2] 스킬 라이브러리 + 선택기
│   ├── freebrain_exploration/       # 호기심 기반 RL (RND, IMGEP, PPO)
│   ├── freebrain_cognitive/         # [계층3] LLM 인지 집행부
│   ├── freebrain_metacog/           # MetaCog-RL 모니터링 + 단계 전환
│   └── freebrain_bringup/           # 단계별 launch 파일
├── .gitignore
├── setup_env.sh
└── requirements.txt
```

**빌드 순서:**
```
Tier 0: freebrain_msgs (의존성 없음, 최우선 빌드)
Tier 1: freebrain_description, freebrain_safety
Tier 2: freebrain_motor, freebrain_sim, freebrain_perception, freebrain_memory
Tier 3: freebrain_skills, freebrain_exploration
Tier 4: freebrain_cognitive, freebrain_metacog
Tier 5: freebrain_bringup
```

---

## 2. 핵심 인터페이스

### 커스텀 메시지
- `SafetyStatus.msg` — 안전 상태 (100Hz): all_ok, joint_limits_ok, velocity_ok, current_ok, workspace_ok
- `DetectedObject.msg` — 감지 물체: label, position_3d, bbox_2d, mask, depth_estimate
- `SceneGraph.msg` — 장면 그래프: node_ids, node_labels, node_positions, edges
- `DevelopmentalState.msg` — 발달 상태: current_stage(0-5), stage_progress, 메타인지 신호들
- `CognitiveGoal.msg` — 인지 목표: goal_description, goal_type, reward_function_code

### 커스텀 서비스
- `ExecuteSkill.srv` — 스킬 실행 요청/결과
- `QueryMemory.srv` / `StoreMemory.srv` — 메모리 조회/저장
- `DecomposeTask.srv` — LLM 태스크 분해
- `GenerateReward.srv` — LLM 보상 함수 생성

### 커스텀 액션
- `TrainSkill.action` — MJX에서 스킬 훈련 (목표 → 결과 → 피드백)

### 안전 아키텍처 (이중 경로)
- **Hard safety** (`ros2_control`/드라이버): joint/velocity limit enforcement, watchdog timeout stop, e-stop pass-through. 컨트롤러 루프에서 fail-close로 차단.
- **Soft safety** (`freebrain_safety`): workspace 경계, 전류 스파이크 기반 접촉 감지, 단계별 제한치 튜닝. 실행 전 필터 + 실행 중 모니터링.
- **검증 기준**: 시뮬레이터/실기 각각 안전 경계 fuzz test 1000회, hard safety false-negative 0건.

### LLM 코드 실행 샌드박스 정책
- 실행 격리: 독립 프로세스(권장: 컨테이너)에서만 실행, 호스트 네트워크 기본 차단.
- 파일 권한: 읽기 전용 코드 볼륨 + `/tmp/freebrain_sandbox` 쓰기 허용, 그 외 쓰기 금지.
- 리소스 제한: timeout 3초(훈련 코드 생성은 10초), 메모리 512MB, CPU 1코어.
- 코드 제한: AST allowlist(`math`, `numpy` 등 최소 허용), `import os/subprocess/socket` 및 `exec/eval` 금지.
- 감사 로그: 원본 프롬프트, 생성 코드 hash, 실행 결과, 차단 사유를 전부 저장.

### 토픽 아키텍처
```
계층3 (Cognitive) ──CognitiveGoal──> 계층2 (Skills) ──JointCommand──> 계층1 (Motor)
     ^                                    ^                                |
     └──DevelopmentalState──(MetaCog)─────┘                     /joint_states (피드백)
     └──DetectedObjectArray──(Perception)─┘                     /camera/image (피드백)
```

---

## 3. Phase 1 — 기반 구축 (1~4주차)

### Week 1: 워크스페이스 부트스트랩 + MJCF 모델
| 일 | 작업 | 산출물 |
|----|------|--------|
| 1 | colcon 워크스페이스 초기화, `freebrain_msgs` 전체 메시지 정의, `freebrain_description` 생성 | `colcon build` 성공 |
| 2 | 기존 URDF(`/home/joo/robot_ws/.../open_manipulator_x.urdf.xacro`)를 MJCF로 변환. 액추에이터(위치+속도), 센서(위치/속도/토크), 접촉 쌍 추가 | MJCF 모델 MuJoCo 뷰어에서 검증 |
| 3 | 테이블 + 물체(큐브/원통/구) MJCF 장면 생성. 기존 `rl_table.sdf` 치수(0.5x0.6x0.3m) 재활용 | 물리 시뮬레이션 정상 동작 |

### Week 2: 안전 래퍼 + MJX 환경
| 일 | 작업 | 산출물 |
|----|------|--------|
| 1-2 | `freebrain_safety` 구현: **Hard safety**(ros2_control limit/stop controller + watchdog) + **Soft safety**(`safety_wrapper.py`, ROS/MJX 공용). 관절 한계(±5° 버퍼), 속도 상한(46rpm), 전류 충돌 감지. 기존 `safety.py` 패턴 확장 | 하드/소프트 안전 테스트 통과 |
| 3 | `freebrain_motor` 구현: 기존 `ros2_control_api.py` 포팅, `kinematics.py` FK/IK 포팅 | 실제 HW/시뮬 양쪽 동작 |
| 4-5 | `freebrain_sim/mjx_env.py`: JAX 기반 MJX 병렬 환경. 병렬 수 목표를 단계형으로 운영(512 안정화 → 1024 기본 운용 → 2048+ 스트레스 테스트). 도메인 랜덤화(질량±20%, 마찰±30%, PD게인 변동). **중요: `jax[cuda12_local]` CUDA 지원 재설치 필요** | RTX 4070에서 1024 기본 운용 + 2048+ 스트레스 통과 |

### Week 3: Stage 0 (반사) + Stage 1 (운동 옹알이)
| 일 | 작업 | 산출물 |
|----|------|--------|
| 1-2 | Stage 0: MJX에서 각 관절 스윕, 토크 피드백 기록, FK 검증 | 검증된 자기 모델 JSON |
| 3-5 | `freebrain_exploration`: RND 모듈(JAX, 3레이어 256유닛 MLP 2개) + Brax PPO. 상태=관절위치[4]+관절속도[4]+EE위치[3]=11차원. MJX 1024 기본 병렬에서 훈련(필요 시 2048+ 확장) | 에이전트가 다양한 관절 구성 탐색 |

### Week 4: Sim-to-Real 전이 + 통합 테스트
| 일 | 작업 | 산출물 |
|----|------|--------|
| 1-2 | `sim_to_real.py`: 보수적 행동 스케일링(30%), 관측 정규화, 행동 스무딩 | 전이 유틸리티 테스트 |
| 3-4 | 실제 HW에 Stage 0+1 배포. 안전 래퍼 활성 상태에서 옹알이 정책 실행 | 실제 팔 자율 탐색 |
| 5 | `stage0/1_launch.py` 생성, W&B 추적 설정 | End-to-end 동작 |

---

## 4. Phase 2 — 물체 상호작용 (5~7주차)

### Week 5: 인식 파이프라인
| 일 | 작업 | 산출물 |
|----|------|--------|
| 1 | 웹캠 드라이버 + 카메라 캘리브레이션 (`cv2.calibrateCamera`) | 1080p@30fps 퍼블리시 |
| 2 | ArUco 기반 hand-eye 캘리브레이션 (스케일 기준점) | 카메라-로봇 좌표 변환 |
| 3-4 | Depth Anything V2 (ViT-S) + Grounding DINO + SAM2 파이프라인. RGB→물체감지→분할→3D위치 | **단일 KPI:** 정적 물체 평균 오차 <3cm, 부분가림/이동 물체 <5cm |
| 5 | NetworkX 장면 그래프 (5Hz 업데이트) | 3~5개 물체 정확 표현 |

### Week 6: 메모리 시스템 + IMGEP
| 일 | 작업 | 산출물 |
|----|------|--------|
| 1-2 | `freebrain_memory`: ChromaDB 3개 컬렉션(에피소드/의미/절차), sentence-transformers 임베딩, 작업 메모리 조립기 | 메모리 서비스 정상 응답 |
| 3-5 | IMGEP 모듈: 목표 공간=물체 위치, 학습 진행 기반 목표 선택, MJX 500K 스텝 훈련 | 밀기 행동 자율 발견 |

### Week 7: Stage 2 (물체 발견) + 실제 배포
- 인식↔IMGEP 통합 (시뮬 + 실제)
- 실제 HW에서 3~5개 물체 자율 탐색
- 모든 에피소드 에피소드 메모리에 기록

---

## 5. Phase 3 — LLM 통합 (8~11주차)

### Week 8: 인지 집행부 핵심
| 일 | 작업 | 산출물 |
|----|------|--------|
| 1-2 | `llm_client.py`: GPT-4o / Claude / Qwen2.5-VL-7B(4bit) 통합 클라이언트 | 3개 백엔드 동작 |
| 3-4 | `cognitive_executive_node.py`: 작업 메모리 조립 → LLM 추론 → 목표 생성 → 스킬 디스패치 → 피드백 주입 (Inner Monologue) | 유효한 목표 제안 |
| 5 | 피드백 인젝터: 스킬 실행 후 장면 변화/성공여부 → LLM 컨텍스트 | 폐루프 추론 |

### Week 9: 스킬 라이브러리 + LLM 가이드 훈련
| 일 | 작업 | 산출물 |
|----|------|--------|
| 1-2 | 스킬 매니저: ChromaDB 인덱싱, **py_trees 고정** 시퀀싱, 어포던스 스코어링 | 의미 검색 기반 스킬 선택 |
| 3-4 | 보상 생성기: LLM이 Python 보상 코드 생성 → 샌드박스 검증(네트워크 차단, AST allowlist, timeout/메모리 제한, 감사 로그). 태스크 분해기 구현 | 유효한 보상 함수 생성 |
| 5 | LLM 보상 → MJX 훈련 → 검증 → 스킬 저장 전체 파이프라인 연결 | End-to-end 자동 스킬 학습 |

### Week 10: Stage 3 (LLM 가이드 파지)
- LLM이 파지 보상 생성 (접근→정렬→파지→리프트)
- MJX 1024 기본 병렬에서 PPO 훈련, 2048+는 가속 옵션 → >70% 성공률
- 실제 HW 배포, 스킬 라이브러리 등록

### Week 11: 정책 실행기 + Stage 3 통합
| 일 | 작업 | 산출물 |
|----|------|--------|
| 1-3 | `policy_runner_node.py`: PPO/MoveIt 정책 전환. 기본 스킬 등록: home_pose, reach, open/close gripper, move_joints | 다중 정책 실행 + 5개 프리미티브 스킬 |
| 4-5 | `stage3_grasping.launch.py`, 실제 HW 파지 테스트 | End-to-end 파지 동작 |

---

## 6. Phase 4 — 안정화/평가 (12~15주차, 핵심 범위)

### Week 12: 통합 안정화 + 폴백 경로 고정
- 정책 실행 arbitration 규칙 확정(RL 정책 실패 시 MoveIt 폴백)
- 실기 안정화 테스트(연속 4시간), 장애 복구 시나리오 점검
- Stage 0~3 운영 대시보드(성공률/지연/안전 이벤트) 고정

### Week 13: MetaCog-RL + 단계 게이트(Stage 0~3)
- 메타인지 신호 모니터링: 스킬 학습률, 순방향 모델 신뢰도, 태스크 성공률
- 유한 상태 기계로 Stage 0→3 자동 전환(핵심 범위)
- 시뮬레이션 가속 실행으로 단계 게이트 검증

### Week 14: 회귀/장기 운용 테스트
- `full_system_sim.launch.py` 통합 테스트 + 실기 회귀 테스트
- Stage 0~3 시나리오 자동 평가(리칭/밀기/파지)
- 12GB VRAM 시간 다중화 스케줄 고정(인식↔훈련↔추론)

### Week 15: 평가 + 마무리
- 신규 물체 테스트, 스킬 성장률, 비발달적 기준선 비교(Stage 0~3 기준)
- 실제 HW 24시간 연속 운용
- 문서화, 실험 대시보드

---

## 7. Phase 5 (옵션) — 자율 성장 확장 (16주차 이후)

### Option A: Stage 4 (소통적 학습)
- 인간 Q&A 인터페이스 + 웹 검색 도구(SerpAPI/Tavily)
- 미지 물체 만남 시 질문/검색 → 의미 메모리 저장
- Human-in-the-loop 수정 메커니즘(TRANSIC 패턴)

### Option B: Stage 5 (자율 실험)
- Voyager 스타일 자동 커리큘럼: LLM이 스킬 격차 식별 → 새 도전 제안
- 비대칭 셀프 플레이: Alice(구성 생성) vs Bob(재현) → 점진적 난이도 증가
- 메타인지 기반 장기 성장 루프 검증

---

## 8. 재사용할 기존 코드

| 파일 | 용도 |
|------|------|
| `/home/joo/Projects/ros2_project/.../rl_env.py` | MJX 환경 관측공간(18차원), 보상 구조 참고 |
| `/home/joo/robot_ws/.../open_manipulator_x.urdf.xacro` | MJCF 변환 소스 (관성, 관절한계, 메시) |
| `/home/joo/Projects/ros2_project/.../ros2_control_api.py` | ros2_control 인터페이스 패턴 재사용 |
| `/home/joo/Projects/ros2_project/.../safety.py` | 안전 상태 머신 패턴 확장 |
| `/home/joo/Projects/ros2_project/.../kinematics.py` | FK/IK 포팅 |
| `/home/joo/Projects/ros2_project/.../rl_table.sdf` | 테이블 치수, 물체 모델 참고 |

---

## 9. 리스크 및 완화 전략

| 리스크 | 가능성 | 완화 방법 |
|--------|--------|-----------|
| Sim-to-Real 전이 실패 | 높음 | 도메인 랜덤화, 30% 행동 스케일링, Gazebo 중간 검증, MoveIt 폴백 |
| LLM 레이턴시 | 중간 | 쿼리 캐싱, 로컬 Qwen(시간 민감), API(복잡 추론), 계층별 시간 분리 |
| 12GB VRAM 부족 | 높음 | 시간 다중화(인식→훈련→추론 교대), 경량 모델 선택, 명시적 메모리 관리, 병렬 수 단계형 스케일링(512→1024→2048+) |
| 단안 깊이 정확도 | 중간 | ArUco 앵커 포인트, 접촉 피드백 보정, 작업대 높이 제약 |
| 15주 타임라인 | 중간 | 15주 핵심 범위를 Stage 0~3로 고정, Stage 4~5는 Phase 5 옵션으로 분리 |
| LLM 보상 함수 오류/보안 | 중-높 | 샌드박스 검증(네트워크 차단, AST allowlist, timeout/메모리 제한, 감사 로그), 기존 보상 템플릿, 자동 행동 검증, 수동 검토 폴백 |

---

## 10. Python 의존성

```
# GPU/ML 핵심
jax[cuda12_local]>=0.6.0    # CUDA 지원 재설치 필수!
mujoco>=3.5.0, mujoco-mjx>=3.5.0, brax>=0.12.0

# 비전
groundingdino-py, segment-anything-2, depth-anything-v2
opencv-contrib-python>=4.8.0

# LLM/NLP
openai>=1.0, anthropic>=0.35.0, transformers>=4.45.0
bitsandbytes>=0.44.0, sentence-transformers>=3.0.0

# 메모리: chromadb>=0.5.0
# RL: gymnasium>=1.0.0, stable-baselines3>=2.0
# 추적: wandb>=0.15.0
# 유틸: networkx>=3.0
```

---

## 11. 검증 방법

- **Phase 1**: 안전 래퍼 경계조건 단위테스트, MJX 환경 결정성 테스트, 실제 팔 1시간 연속 옹알이 안전 동작
- **Phase 2**: 3D 위치 추정 KPI(정적 <3cm, 부분가림/이동 <5cm), IMGEP 역량 곡선 상승, 메모리 검색 정확도
- **Phase 3**: LLM 추론 루프 폐루프 동작, MJX 파지 훈련 수렴, sim-to-real 파지 성공률 >50%
- **Phase 4**: Stage 0~3 자동 단계 전환 시뮬레이션 검증, 24시간 실제 연속 운용, 비발달적 기준선 대비 우위 입증
