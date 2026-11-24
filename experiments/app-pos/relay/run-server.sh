#!/usr/bin/env bash
# 렌탈 서버(Linux)에서 MediaMTX 를 실행한다. 죽으면 2초 뒤 다시 켠다.
#
#   relay/run-server.sh            # 실행 (보통 deploy/start.sh 가 tmux 안에서 부른다)
#   relay/run-server.sh --check    # 설정만 확인하고 끝냄 (비밀번호는 출력하지 않음)
#
# 설정은 env 파일에서 읽는다 (기본: 이 폴더 기준 ../deploy/.env, 바꾸려면 HAWKEYE_ENV=<파일>).
#   HAWKEYE_PUBLISH_USER / HAWKEYE_PUBLISH_PASS   촬영 폰 송출 계정 (필수)
#   HAWKEYE_READ_USER / HAWKEYE_READ_PASS         관리자 앱 라이브 보기 계정 (필수)
#   HAWKEYE_BIND_IP      RTSP·SRT 를 열 주소 (기본: tailscale ip -4). 0.0.0.0 은 거부한다 — 공인 IP 노출
#   HAWKEYE_MEDIAMTX     MediaMTX 실행 파일 (기본: 이 폴더의 bin/mediamtx)
#   HAWKEYE_RECORDINGS   녹화 폴더 (기본: ~/hawkeye/recordings)
set -u

here="$(cd "$(dirname "$0")" && pwd)"
check_only=0
[ "${1:-}" = "--check" ] && check_only=1

die() { echo "[run-server] $*" >&2; exit 2; }

env_file="${HAWKEYE_ENV:-$here/../deploy/.env}"
[ -f "$env_file" ] || die "env 파일이 없습니다: $env_file (deploy/.env.example 을 복사해 채우세요)"
# shellcheck disable=SC1090
set -a; . "$env_file"; set +a

for name in HAWKEYE_PUBLISH_USER HAWKEYE_PUBLISH_PASS HAWKEYE_READ_USER HAWKEYE_READ_PASS; do
  [ -n "${!name:-}" ] || die "$name 이 비어 있습니다 — 계정 없이 열면 아무나 송출·시청할 수 있어 실행하지 않습니다."
done
[ "$HAWKEYE_PUBLISH_USER" != "$HAWKEYE_READ_USER" ] || die "송출 계정과 보기 계정은 달라야 합니다."

bind_ip="${HAWKEYE_BIND_IP:-}"
if [ -z "$bind_ip" ]; then
  command -v tailscale >/dev/null || die "tailscale 이 없습니다. 설치하거나 HAWKEYE_BIND_IP 를 주세요."
  bind_ip="$(tailscale ip -4 2>/dev/null | head -1)"
  [ -n "$bind_ip" ] || die "Tailscale IP 를 얻지 못했습니다 (tailscale up 했는지 확인)."
fi
case "$bind_ip" in
  0.0.0.0|"::"|"[::]") die "HAWKEYE_BIND_IP=$bind_ip 은 모든 네트워크에 엽니다. 이 서버는 공인 IP 라 거부합니다." ;;
esac

mtx="${HAWKEYE_MEDIAMTX:-$here/bin/mediamtx}"
[ -x "$mtx" ] || die "MediaMTX 실행 파일이 없습니다: $mtx (relay/README.md 의 '렌탈 서버' 참고)"
conf="$here/mediamtx.server.yml"
rec="${HAWKEYE_RECORDINGS:-$HOME/hawkeye/recordings}"

# 설정 파일의 계정 자리표시자·주소를 환경변수로 덮어쓴다 (번호는 mediamtx.server.yml 순서).
export MTX_AUTHINTERNALUSERS_0_USER="$HAWKEYE_PUBLISH_USER" MTX_AUTHINTERNALUSERS_0_PASS="$HAWKEYE_PUBLISH_PASS"
export MTX_AUTHINTERNALUSERS_1_USER="$HAWKEYE_READ_USER"    MTX_AUTHINTERNALUSERS_1_PASS="$HAWKEYE_READ_PASS"
# 서버 자신이 Tailscale IP 로 붙을 때도 '서버 안'으로 본다 (tracking 이 rtsp://<Tailscale IP>:8554/cam1 을 계정 없이 읽음).
export MTX_AUTHINTERNALUSERS_2_IPS="127.0.0.1,::1,$bind_ip"
export MTX_AUTHINTERNALUSERS_3_IPS="127.0.0.1,::1,$bind_ip"
export MTX_RTSPADDRESS="$bind_ip:8554" MTX_SRTADDRESS="$bind_ip:8890"
export MTX_PATHDEFAULTS_RECORDPATH="$rec/%path/%Y-%m-%d_%H-%M-%S-%f"

echo "[run-server] MediaMTX : $mtx"
echo "[run-server] 설정     : $conf"
echo "[run-server] RTSP     : rtsp://$bind_ip:8554/<cam1|cam2|cam1_ai|cam2_ai>"
echo "[run-server] SRT      : srt://$bind_ip:8890?streamid=publish:<cam1|cam2>:<송출 계정>:<비밀번호>&pkt_size=1316"
echo "[run-server] 재생 API : http://127.0.0.1:9996 (서버 안만)"
echo "[run-server] 녹화     : $rec"
[ "$check_only" = 1 ] && exit 0

mkdir -p "$rec"
cd "$here" || exit 1
while true; do
  "$mtx" "$conf"
  echo "[run-server] MediaMTX 종료됨 (exit $?) — 2초 뒤 다시 시작"
  sleep 2
done
