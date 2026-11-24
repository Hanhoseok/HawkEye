#!/usr/bin/env bash
# 렌탈 종료 전 정리: 녹화 영상과 경보 서버 데이터(DB·장면 사진·클립)를 지운다.
# 사람 얼굴이 찍힌 영상·사진이라 서버에 남기지 않는다. 남길 것은 먼저 팀 드라이브로 옮긴다.
#
#   deploy/cleanup.sh          # 지울 것과 크기를 보여 주고, '삭제' 를 입력해야 지운다
#
# 지우는 경로는 run-alert.sh 와 같은 규칙으로 정한다 (HAWKEYE_DB·SNAPSHOTS·CLIPS 를 따로 지정했으면 그 경로도).
# 지우지 않는 것: 코드, deploy/.env (계정), ~/.ssh/authorized_keys — 필요하면 직접 지운다.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
env_file="${HAWKEYE_ENV:-$here/.env}"
# shellcheck disable=SC1090
[ -f "$env_file" ] && { set -a; . "$env_file"; set +a; }
rec="${HAWKEYE_RECORDINGS:-$HOME/hawkeye/recordings}"
data="${HAWKEYE_DATA:-$HOME/hawkeye/server-data}"
db="${HAWKEYE_DB:-$data/hawkeye.db}"
snaps="${HAWKEYE_SNAPSHOTS:-$data/snapshots}"
clips="${HAWKEYE_CLIPS:-$data/clips}"
session="${HAWKEYE_TMUX_SESSION:-hawkeye}"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "[cleanup] tmux 세션 '$session' 이 실행 중입니다. 먼저 끄세요: tmux kill-session -t $session" >&2
  exit 2
fi

# data 폴더 밖에 따로 둔 것만 더한다 (안에 있으면 data 를 지울 때 함께 지워진다)
inside() { case "$1" in "$data"/*) return 0 ;; *) return 1 ;; esac; }
targets=("$rec" "$data")
inside "$db" || targets+=("$db" "$db-journal" "$db-wal" "$db-shm")
inside "$snaps" || targets+=("$snaps")
inside "$clips" || targets+=("$clips")

echo "[cleanup] 지울 것:"
for t in "${targets[@]}"; do
  if [ -e "$t" ]; then echo "  $t  ($(du -sh "$t" | cut -f1))"; else echo "  $t  (없음)"; fi
done
printf "[cleanup] 되돌릴 수 없습니다. 지우려면 '삭제' 를 입력하세요: "
read -r answer
[ "$answer" = "삭제" ] || { echo "[cleanup] 취소했습니다."; exit 1; }
rm -rf -- "${targets[@]}"
left=0
for t in "${targets[@]}"; do [ -e "$t" ] && { echo "[cleanup] 남음: $t" >&2; left=1; }; done
[ "$left" = 0 ] && echo "[cleanup] 지웠습니다." || exit 1
