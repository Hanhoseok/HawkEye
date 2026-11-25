#!/usr/bin/env bash
# 렌탈 서버에서 HawkEye 시연 구성 요소를 tmux 세션 'hawkeye' 하나에 띄운다.
#
#   deploy/start.sh                # 띄우기 (이미 떠 있으면 아무것도 안 함)
#   tmux attach -t hawkeye         # 보기 (빠져나오기: Ctrl+b 다음 d)
#   tmux kill-session -t hawkeye   # 모두 끄기
#
# 창: relay (MediaMTX) · alert (경보 서버) · pipeline (tracking 을 직접 실행할 자리)
# 각 창의 스크립트가 env 파일을 직접 읽는다 — tmux 명령줄에 비밀번호를 넘기지 않는다.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
app_pos="$(cd "$here/.." && pwd)"
session=hawkeye
die() { echo "[start] $*" >&2; exit 2; }

command -v tmux >/dev/null || die "tmux 가 없습니다."
if tmux has-session -t "$session" 2>/dev/null; then
  echo "[start] 이미 실행 중입니다: tmux attach -t $session"
  exit 0
fi

env_file="$(cd "$(dirname "${HAWKEYE_ENV:-$here/.env}")" 2>/dev/null && pwd)/$(basename "${HAWKEYE_ENV:-$here/.env}")"
[ -f "$env_file" ] || die "env 파일이 없습니다: $env_file (.env.example 을 복사해 채우세요)"
perm="$(stat -c %a "$env_file" 2>/dev/null || echo '?')"
[ "$perm" = 600 ] || echo "[start] 주의: $env_file 권한이 $perm 입니다. 비밀번호가 있으니 chmod 600 을 권장합니다."
# shellcheck disable=SC1090
set -a; . "$env_file"; set +a

# 1) 설정 확인. 중계 설정이 틀리면 아무것도 띄우지 않는다.
HAWKEYE_ENV="$env_file" "$app_pos/relay/run-server.sh" --check || die "중계 설정을 고친 뒤 다시 실행하세요."
alert_ok=1
HAWKEYE_ENV="$env_file" "$here/run-alert.sh" --check || { alert_ok=0; echo "[start] 경보 서버는 띄우지 않습니다 (위 이유)."; }

rec="${HAWKEYE_RECORDINGS:-$HOME/hawkeye/recordings}"
logs="${HAWKEYE_LOGS:-$HOME/hawkeye/logs}"
mkdir -p "$rec" "$logs"

# 2) 디스크 여유 (녹화는 자동으로 지우지 않는다)
free_gb=$(df -BG --output=avail "$rec" | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -lt 50 ]; then
  echo "[start] 경고: 디스크 여유 ${free_gb}GB — 50GB 미만. 녹화 폴더($rec) $(du -sh "$rec" | cut -f1)"
elif [ "${free_gb:-0}" -lt 100 ]; then
  echo "[start] 주의: 디스크 여유 ${free_gb}GB — 100GB 미만"
else
  echo "[start] 디스크 여유 ${free_gb}GB"
fi

# 3) tmux 세션
q() { printf '%q' "$1"; }
tmux new-session -d -s "$session" -n relay -c "$app_pos/relay" \
  "HAWKEYE_ENV=$(q "$env_file") ./run-server.sh 2>&1 | tee -a $(q "$logs/mediamtx.log")"
if [ "$alert_ok" = 1 ]; then
  tmux new-window -t "$session" -n alert -c "$app_pos/server" \
    "HAWKEYE_ENV=$(q "$env_file") $(q "$here/run-alert.sh") 2>&1 | tee -a $(q "$logs/alert-server.log")"
else
  tmux new-window -t "$session" -n alert -c "$app_pos/server"
fi
tmux new-window -t "$session" -n pipeline -c "$app_pos/.."

bind_ip="${HAWKEYE_BIND_IP:-$(tailscale ip -4 2>/dev/null | head -1)}"
echo "[start] 띄웠습니다: tmux attach -t $session   (로그: $logs)"
echo "[start] 파이프라인 입력: rtsp://$bind_ip:8554/cam1 (서버 안이라 계정 없이), 경보: --alert-server http://$bind_ip:8000"
echo "[start] AI 화면 시험 (파이프라인 없이): cd $app_pos/client && python3 -m hawkeye_client.stream_demo"
