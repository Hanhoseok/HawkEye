#!/usr/bin/env bash
# 렌탈 서버 스크립트 시험 (Linux, bash). MediaMTX·ffmpeg·Tailscale 없이 돈다.
#   bash relay/tests/server-scripts.tests.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
relay="$(cd "$here/.." && pwd)"
deploy="$(cd "$relay/../deploy" && pwd)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

pass=0; fail=0
ok()   { pass=$((pass + 1)); echo "  ok   $1"; }
bad()  { fail=$((fail + 1)); echo "  FAIL $1"; [ -n "${2:-}" ] && echo "       $2"; }
expect_exit() {  # 설명, 기대 종료 코드, 명령...
  local what=$1 want=$2; shift 2
  out="$("$@" 2>&1)"; code=$?
  [ "$code" = "$want" ] && ok "$what" || bad "$what" "종료 코드 $code (기대 $want): $out"
}
contains()     { case "$out" in *"$2"*) ok "$1" ;; *) bad "$1" "'$2' 가 없음: $out" ;; esac; }
not_contains() { case "$out" in *"$2"*) bad "$1" "'$2' 가 출력에 있음" ;; *) ok "$1" ;; esac; }

printf '#!/bin/sh\nexit 0\n' > "$tmp/mediamtx"; chmod +x "$tmp/mediamtx"
printf '#!/bin/sh\nexit 0\n' > "$tmp/python"; chmod +x "$tmp/python"
: > "$tmp/a.mp4"; : > "$tmp/b.mp4"
good() {
  cat > "$tmp/.env" <<EOF
HAWKEYE_PUBLISH_USER=pubuser
HAWKEYE_PUBLISH_PASS=PubSecret123
HAWKEYE_READ_USER=readuser
HAWKEYE_READ_PASS=ReadSecret456
HAWKEYE_API_KEY=KeySecret789
HAWKEYE_BIND_IP=100.64.0.1
HAWKEYE_MEDIAMTX=$tmp/mediamtx
HAWKEYE_PYTHON=$tmp/python
HAWKEYE_RECORDINGS=$tmp/rec
HAWKEYE_DATA=$tmp/data
EOF
}
export HAWKEYE_ENV="$tmp/.env"

echo "== run-server.sh --check"
rm -f "$tmp/.env"
expect_exit "env 파일이 없으면 거부" 2 "$relay/run-server.sh" --check
good; sed -i 's/^HAWKEYE_PUBLISH_PASS=.*/HAWKEYE_PUBLISH_PASS=/' "$tmp/.env"
expect_exit "송출 비밀번호가 비면 거부" 2 "$relay/run-server.sh" --check
contains "  거부 이유에 변수 이름" "HAWKEYE_PUBLISH_PASS"
good; sed -i 's/^HAWKEYE_READ_USER=.*/HAWKEYE_READ_USER=pubuser/' "$tmp/.env"
expect_exit "송출·보기 계정이 같으면 거부" 2 "$relay/run-server.sh" --check
good; sed -i 's/^HAWKEYE_BIND_IP=.*/HAWKEYE_BIND_IP=0.0.0.0/' "$tmp/.env"
expect_exit "0.0.0.0 에 열기 거부" 2 "$relay/run-server.sh" --check
good; sed -i "s#^HAWKEYE_MEDIAMTX=.*#HAWKEYE_MEDIAMTX=$tmp/none#" "$tmp/.env"
expect_exit "실행 파일이 없으면 거부" 2 "$relay/run-server.sh" --check
good
expect_exit "올바른 설정은 통과" 0 "$relay/run-server.sh" --check
contains "  RTSP 주소는 Tailscale IP" "rtsp://100.64.0.1:8554"
contains "  SRT 주소는 Tailscale IP" "srt://100.64.0.1:8890"
not_contains "  송출 비밀번호를 출력하지 않음" "PubSecret123"
not_contains "  보기 비밀번호를 출력하지 않음" "ReadSecret456"

echo "== deploy/run-alert.sh --check"
good; sed -i 's/^HAWKEYE_API_KEY=.*/HAWKEYE_API_KEY=/' "$tmp/.env"
expect_exit "API 키가 비면 거부" 2 "$deploy/run-alert.sh" --check
good; sed -i 's/^HAWKEYE_BIND_IP=.*/HAWKEYE_BIND_IP=0.0.0.0/' "$tmp/.env"
expect_exit "0.0.0.0 에 열기 거부" 2 "$deploy/run-alert.sh" --check
good
expect_exit "올바른 설정은 통과" 0 "$deploy/run-alert.sh" --check
contains "  앱 서버 주소" "http://100.64.0.1:8000"
contains "  라이브 주소는 계정을 가림" "rtsp://<보기 계정>@100.64.0.1:8554/cam1"
contains "  AI 화면 주소" "cam1_ai"
not_contains "  보기 비밀번호를 출력하지 않음" "ReadSecret456"
not_contains "  API 키를 출력하지 않음" "KeySecret789"

echo "== relay/replay-server.sh --dry-run"
good
export HAWKEYE_REPLAY_ENCODER=libx264
expect_exit "파일이 없으면 거부" 2 "$relay/replay-server.sh" "$tmp/none.mp4" --dry-run
expect_exit "시차가 숫자가 아니면 거부" 2 "$relay/replay-server.sh" "$tmp/a.mp4" "$tmp/b.mp4" --offset-ms abc --dry-run
expect_exit "모르는 인코더 거부" 2 env HAWKEYE_REPLAY_ENCODER=vp9 "$relay/replay-server.sh" "$tmp/a.mp4" --dry-run
expect_exit "영상 1개" 0 "$relay/replay-server.sh" "$tmp/a.mp4" --dry-run
contains "  cam1 로 보냄" "8554/cam1"
not_contains "  cam2 는 없음" "8554/cam2"
expect_exit "영상 2개, cam2 를 1.5초 건너뜀" 0 "$relay/replay-server.sh" "$tmp/a.mp4" "$tmp/b.mp4" --offset-ms 1500 --dry-run
contains "  cam2 앞 -ss 1.500" "-ss 1.500 -i $tmp/b.mp4"
contains "  cam1 은 0" "-ss 0.000 -i $tmp/a.mp4"
expect_exit "음수 시차는 cam1 을 건너뜀" 0 "$relay/replay-server.sh" "$tmp/a.mp4" "$tmp/b.mp4" --offset-ms -800 --dry-run
contains "  cam1 앞 -ss 0.800" "-ss 0.800 -i $tmp/a.mp4"
contains "  안드로이드 호환 형식" "format=yuv420p"
contains "  Main 프로파일" "-profile:v main"
contains "  B-프레임 없음" "-bf 0"
not_contains "  송출 비밀번호를 출력하지 않음" "PubSecret123"

echo "== deploy/cleanup.sh"
good
export HAWKEYE_TMUX_SESSION="hawkeye-test-$$"   # 실제 세션과 겹치지 않게
mkdir -p "$tmp/rec/cam1" "$tmp/data/snapshots" "$tmp/outside"
: > "$tmp/rec/cam1/a.mp4"; : > "$tmp/data/hawkeye.db"; : > "$tmp/outside/other.db"; : > "$tmp/outside/other.db-wal"
echo "HAWKEYE_DB=$tmp/outside/other.db" >> "$tmp/.env"
out="$(echo 아니 | "$deploy/cleanup.sh" 2>&1)"; code=$?
[ "$code" = 1 ] && [ -e "$tmp/rec/cam1/a.mp4" ] && ok "'삭제' 가 아니면 아무것도 안 지움" || bad "'삭제' 가 아니면 아무것도 안 지움" "$code $out"
contains "  data 밖의 DB 도 목록에 있음" "$tmp/outside/other.db"
out="$(echo 삭제 | "$deploy/cleanup.sh" 2>&1)"; code=$?
[ "$code" = 0 ] && ok "'삭제' 면 지움" || bad "'삭제' 면 지움" "$code $out"
for f in "$tmp/rec" "$tmp/data" "$tmp/outside/other.db" "$tmp/outside/other.db-wal"; do
  [ -e "$f" ] && bad "  지워짐: $f" || ok "  지워짐: ${f#$tmp/}"
done

echo
echo "통과 $pass, 실패 $fail"
[ "$fail" = 0 ]
