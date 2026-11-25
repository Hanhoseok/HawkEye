#!/usr/bin/env bash
# 렌탈 서버에서 경보 서버를 실행한다 (deploy/start.sh 가 tmux 안에서 부른다).
# 설정은 env 파일에서 읽고, 비어 있는 값은 기본값을 만든다. 비밀번호는 명령줄에 쓰지 않는다.
#
#   deploy/run-alert.sh            # 실행
#   deploy/run-alert.sh --check    # 설정만 확인 (비밀번호·키는 출력하지 않음)
set -u

here="$(cd "$(dirname "$0")" && pwd)"
app_pos="$(cd "$here/.." && pwd)"
check_only=0
[ "${1:-}" = "--check" ] && check_only=1
die() { echo "[run-alert] $*" >&2; exit 2; }

env_file="${HAWKEYE_ENV:-$here/.env}"
[ -f "$env_file" ] || die "env 파일이 없습니다: $env_file"
# shellcheck disable=SC1090
set -a; . "$env_file"; set +a

[ -n "${HAWKEYE_API_KEY:-}" ] || die "HAWKEYE_API_KEY 가 비어 있습니다 — Tailscale 에 여는 서버라 키 없이는 실행하지 않습니다."
for name in HAWKEYE_READ_USER HAWKEYE_READ_PASS; do
  [ -n "${!name:-}" ] || die "$name 이 비어 있습니다 (앱 라이브 주소에 필요)."
done

bind_ip="${HAWKEYE_BIND_IP:-$(tailscale ip -4 2>/dev/null | head -1)}"
[ -n "$bind_ip" ] || die "Tailscale IP 를 얻지 못했습니다. HAWKEYE_BIND_IP 를 주세요."
case "$bind_ip" in 0.0.0.0|"::"|"[::]") die "HAWKEYE_BIND_IP=$bind_ip 은 거부합니다 (공인 IP 노출)." ;; esac

data="${HAWKEYE_DATA:-$HOME/hawkeye/server-data}"
cred="$HAWKEYE_READ_USER:$HAWKEYE_READ_PASS@$bind_ip:8554"
export HAWKEYE_STREAMS="${HAWKEYE_STREAMS:-cam1=rtsp://$cred/cam1,cam2=rtsp://$cred/cam2}"
export HAWKEYE_AI_STREAMS="${HAWKEYE_AI_STREAMS:-cam1=rtsp://$cred/cam1_ai,cam2=rtsp://$cred/cam2_ai}"
export HAWKEYE_DB="${HAWKEYE_DB:-$data/hawkeye.db}"
export HAWKEYE_SNAPSHOTS="${HAWKEYE_SNAPSHOTS:-$data/snapshots}"
export HAWKEYE_CLIPS="${HAWKEYE_CLIPS:-$data/clips}"
export HAWKEYE_PLAYBACK_URL="${HAWKEYE_PLAYBACK_URL:-http://127.0.0.1:9996}"
export HAWKEYE_RECORDINGS="${HAWKEYE_RECORDINGS:-$HOME/hawkeye/recordings}"
export HAWKEYE_LIBRARY="${HAWKEYE_LIBRARY:-$data/library}"

py="${HAWKEYE_PYTHON:-$app_pos/server/.venv/bin/python}"
[ -x "$py" ] || die "경보 서버 가상환경이 없습니다: $py (relay/README.md '렌탈 서버' 참고)"

echo "[run-alert] 주소      : http://$bind_ip:8000  (관리자 앱 설정에 넣는 서버 주소)"
echo "[run-alert] 데이터    : $data"
echo "[run-alert] 라이브    : $(echo "$HAWKEYE_STREAMS" | sed -E 's#//[^@/]+@#//<보기 계정>@#g')"
echo "[run-alert] AI 화면   : $(echo "$HAWKEYE_AI_STREAMS" | sed -E 's#//[^@/]+@#//<보기 계정>@#g')"
echo "[run-alert] 재생 API  : $HAWKEYE_PLAYBACK_URL"
echo "[run-alert] 보관함    : $HAWKEYE_LIBRARY  (넣기: python -m hawkeye_server.library import --help)"
[ "$check_only" = 1 ] && exit 0

mkdir -p "$data"
cd "$app_pos/server" || exit 1
exec "$py" -m hawkeye_server --host "$bind_ip" --port 8000
