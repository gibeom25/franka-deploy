#!/bin/bash
# franka-deploy 런처 -- 바탕화면 아이콘이 이것을 부른다.
#
# 로봇 노드(robot_node.py, 1kHz 제어 전용 프로세스)와 웹 앱 서버를 둘 다
# 띄우고 대시보드를 브라우저로 연다. 둘 중 이미 떠 있는 게 있으면 절대
# 재시작하지 않는다 -- 재시작하면 로봇 연결/세션 설정이 끊긴다
# (manipulation-stack의 "이미 떠 있는 GUI는 pull 영향 안 받음"과 같은 이유).
#
# 실행 전 자동으로 git pull을 시도한다 (오프라인/실패 시 경고만 찍고 현재
# 코드로 계속 -- 실행을 막는 쪽이 더 나쁘다).
#
# venv/포트/로봇 IP는 환경변수로 바꿀 수 있다 (기본값은 이 기계 기준).

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
git pull --ff-only 2>/dev/null || echo "[franka-deploy] git pull 실패 (오프라인?) -- 현재 코드로 실행합니다"

VENV="${FRANKA_DEPLOY_VENV:-$HOME/pylibfranka-venv}"
ROBOT_IP="${FRANKA_DEPLOY_ROBOT_IP:-172.16.0.2}"
ROBOT_NODE_PORT="${FRANKA_DEPLOY_ROBOT_NODE_PORT:-5560}"
APP_PORT="${FRANKA_DEPLOY_APP_PORT:-8000}"
LOG_DIR="$HOME/.franka-deploy/logs"
mkdir -p "$LOG_DIR"

if ss -tln 2>/dev/null | grep -q ":${ROBOT_NODE_PORT} "; then
  echo "[franka-deploy] robot_node.py already running on :${ROBOT_NODE_PORT} -- leaving it alone"
else
  echo "[franka-deploy] starting robot_node.py on :${ROBOT_NODE_PORT} (robot_ip=${ROBOT_IP})"
  nohup "$VENV/bin/python" -m franka_deploy.robot.robot_node \
    --robot-ip "$ROBOT_IP" --port "$ROBOT_NODE_PORT" \
    > "$LOG_DIR/robot_node.log" 2>&1 &
  disown
fi

if ss -tln 2>/dev/null | grep -q ":${APP_PORT} "; then
  echo "[franka-deploy] app server already running on :${APP_PORT} -- leaving it alone"
else
  echo "[franka-deploy] starting app server on :${APP_PORT}"
  PYTHONPATH= nohup "$VENV/bin/uvicorn" franka_deploy.api.main:app \
    --host 0.0.0.0 --port "$APP_PORT" \
    > "$LOG_DIR/app.log" 2>&1 &
  disown
  sleep 1.5   # give uvicorn a moment to bind before opening the browser
fi

xdg-open "http://localhost:${APP_PORT}/" >/dev/null 2>&1 &
