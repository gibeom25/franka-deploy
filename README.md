# franka-deploy

Web bridge between an arbitrary policy server (VLA/RL, any IP) and a Franka
FR3. You define exactly what gets sent to the model (any field names,
shapes, encodings); the response format (joint/EE, absolute/delta) is
auto-detected on connect and shown to you for one-click confirmation before
any motion is commanded.

## Why this exists

Two sibling projects each solved half of this problem on this robot:

- [`~/franka-policy-runner`](../franka-policy-runner): safety pipeline
  (jerk/accel/velocity-clamped reference filter, joint limits, oscillation
  watchdog) for **local** Python policies. No network client.
- [`~/teleop-franka/manipulation-stack-dev`](../teleop-franka/manipulation-stack-dev):
  a real network policy client (`apps/fr3_policy_client.py`) with a
  chunk-boundary-stall fix and analytic FK/IK for EE-delta actions, but its
  request/response JSON shape is hardcoded to one lab's checkpoint.

franka-deploy vendors the validated pieces from both and adds the
generalization layer: a user-authored request schema, and an auto-detector
for the response format.

## Setup

pylibfranka is not pip-installable -- it's a source-built package that
already lives in `~/pylibfranka-venv` (shared with the other Franka
projects on this machine). Install franka-deploy into that same venv:

```bash
source ~/pylibfranka-venv/bin/activate
cd ~/franka-deploy
pip install -e .
```

## Run (two processes)

**1. Robot node** -- owns the 1 kHz control loop, nothing else. Must be
running before you hit "연결" in the dashboard.

```bash
~/pylibfranka-venv/bin/python -m franka_deploy.robot.robot_node --robot-ip 172.16.0.2 --port 5560
```

**2. App** -- web dashboard, policy HTTP client, IK, cameras.

```bash
~/pylibfranka-venv/bin/uvicorn franka_deploy.api.main:app --host 0.0.0.0 --port 8000
```

Then open `http://<this machine>:8000/` for the dashboard. The "로봇 노드
주소" field is the robot node's ZMQ address (`tcp://127.0.0.1:5560` by
default), not the robot's own FCI IP -- that only goes on the robot node's
`--robot-ip` command line.

## Recommended workflow (staged rollout)

1. **설정** -- 로봇 IP, 정책 서버 IP/포트, 카메라 시리얼(비우면 mock 이미지 사용),
   송신 스키마(JSON, 자유 지정), 안전 한계를 입력하고 저장.
2. **연결** -- `read_only`로 연결 (기본값). 모션 명령이 전혀 나가지 않는 상태에서
   실제 로봇 상태 스트리밍만 확인.
3. **감지** -- `/reset`+`/predict` 1회 호출로 응답 포맷(joint/EE, absolute/delta,
   그리퍼 컨벤션)을 자동 감지. 결과와 샘플 값을 보고 필요하면 직접 수정.
4. **확정** -- 감지 결과를 확정하면 그제서야 실제 모션이 허용됨.
5. **Start** -- 실행 시작. 텔레메트리(관절 상태, 정책 지연시간, 재계획 횟수)를
   실시간으로 확인.
6. 언제든 **Stop**(감속 정지) 또는 **E-STOP**(즉시 컨트롤러 정지).

## Architecture

**Two processes, not one.** The first version ran everything (web server,
policy HTTP client, analytic IK, cameras) in the same process as the 1 kHz
`FR3Runtime` control thread, following franka-policy-runner's validated
single-process pattern -- but that pattern was only ever validated for
*cheap* per-tick work (a local, near-free policy call). Live testing here
hit two real reflex aborts that single-process design doesn't protect
against:
- `communication_constraints_violation` right after `connect()`, when
  connect/reset/first-predict all landed on the control thread's startup.
- `joint_motion_generator_velocity/acceleration_discontinuity` during
  EE-delta actions, where analytic IK (~4ms/call) held the GIL long
  enough to starve the 1 kHz thread of CPU time it needed for its own
  1ms budget.

A GIL-tuning workaround (`sys.setswitchinterval`, reordering `start()`)
fixed the first but not the second -- IK cost doesn't go away just by
scheduling around it. The actual fix is `robot/robot_node.py`: a
**separate OS process** that owns `FR3Runtime` and nothing else, talked to
over a ZMQ REQ/REP socket (`robot/zmq_client.py`) by everything else. This
mirrors `mstack.comm.zmq_core.robot_node.ZMQServerRobot`, which
manipulation-stack already validated in production for the same reason.
Within the robot node process the ZMQ handler thread and the control
thread still share a GIL, but that handler only ever does small
get_state/set_joint_target work -- nothing CPU-heavy runs there anymore.

A live `FR3Runtime` thread tracks a setpoint through a jerk/accel/
velocity-clamped reference filter, while a background policy thread (in
the APP process) talks to the remote server and updates that setpoint via
ZMQ. See `franka_deploy/control_loop.py` for the full state machine and
the async chunk-overlap scheduling (ported from manipulation-stack's
`fr3_policy_client.py`, ~40ms replan tail no longer stalls the arm).

| module | role |
|---|---|
| `robot/robot_node.py` | **separate process** -- owns FR3Runtime, ZMQ REP server, nothing else |
| `robot/zmq_client.py` | app-side stub matching FR3Runtime's public API, talks to robot_node.py |
| `robot/reference_filter.py`, `robot/fr3_runtime.py` | vendored from franka-policy-runner, unchanged API |
| `safety/*` | vendored: joint limits, EMA smoother, oscillation watchdog; `workspace_bounds.py` (new) is a teleop-recorded Cartesian fence, checked every tick |
| `cameras/manager.py` | one background thread per camera, shared by the policy loop and the dashboard's live preview |
| `kinematics/fr3_kinematics.py` | vendored analytic FK/IK for EE action spaces |
| `kinematics/rotations.py` | quat/rot6d <-> matrix, generalizing beyond the vendored axis-angle-only convention |
| `schema/spec.py` | the request/response schema dataclasses -- start here |
| `schema/encode.py` | resize/dtype/layout/normalize/encoding pipeline for outbound fields |
| `schema/sources.py`, `schema/plugins.py` | where a field's value comes from, including user-written custom transforms |
| `schema/response_detect.py` | auto-detects joint/EE, absolute/delta, gripper convention |
| `schema/apply.py` | converts a confirmed-format response row into a joint target |
| `control_loop.py` | the two-rate orchestrator + staged-rollout state machine |
| `api/` | FastAPI routes; `static/` is the vanilla-JS dashboard |

`configs/examples/libero_style_example.yaml` demonstrates the schema DSL
using the protocol shape this project was scoped against -- it is a
reference, not a default.

## Tests

No hardware needed (unit tests + a stub HTTP server for the request/response
round trip and the full control-loop state machine with a fake robot
runtime):

```bash
PYTHONPATH= ~/pylibfranka-venv/bin/python -m pytest tests/ -q
```

(`PYTHONPATH=` clears a ROS Humble `pytest11` plugin that this venv
otherwise picks up and that fails to import.)

Hardware-in-the-loop verification (read_only dry run, then a low-risk real
motion test at reduced safety limits) is a separate, supervised step -- see
the workflow above.

## Operational note

franka-deploy and manipulation-stack's robot node cannot hold the FCI at
the same time -- run only one against the physical robot at once.
