#!/usr/bin/env bash
# 대체 경로 (시연 B): 촬영 폰으로 미리 녹화한 영상 2개를 cam1·cam2 에 '실시간처럼' 동시에 흘려보낸다.
# 파이프라인·경보 서버·관리자 앱은 폰이 실시간으로 송출할 때와 똑같이 동작한다.
#
#   relay/replay-server.sh <cam1 영상> [<cam2 영상>] [--offset-ms N] [--dry-run]
#
#   --offset-ms N   두 영상의 시차 맞추기. 양수면 cam2 앞부분 N ms 를 건너뛰고, 음수면 cam1 앞부분 |N| ms 를 건너뛴다.
#                   (박수 소리 등으로 잰 'cam2 가 cam1 보다 늦게 시작한 만큼' 이 아니라
#                    'cam2 에서 같은 장면이 N ms 더 뒤에 나온다' 면 +N)
#   --dry-run       ffmpeg 명령만 보여 준다 (비밀번호는 가림)
#
# 폰 녹화는 HEVC(삼성 '고효율 동영상')이거나 High 프로파일일 수 있어 그대로 복사하지 않고,
# 안드로이드가 확실히 트는 형식(H.264 Main, yuv420p, B-프레임 없음, 1초 키프레임, 5Mbps)으로 다시 인코딩한다.
# GPU 인코더(h264_nvenc)를 먼저 쓰고, 안 되면 libx264 로 바꾼다 (HAWKEYE_REPLAY_ENCODER=libx264 로 강제 가능).
set -u

here="$(cd "$(dirname "$0")" && pwd)"
die() { echo "[replay] $*" >&2; exit 2; }

files=(); offset_ms=0; dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --offset-ms) [ $# -ge 2 ] || die "--offset-ms 뒤에 숫자가 필요합니다"; offset_ms="$2"; shift 2 ;;
    --dry-run) dry=1; shift ;;
    -*) die "모르는 옵션: $1" ;;
    *) files+=("$1"); shift ;;
  esac
done
[ ${#files[@]} -ge 1 ] && [ ${#files[@]} -le 2 ] || die "영상 파일을 1~2개 주세요 (cam1 [cam2])"
[[ "$offset_ms" =~ ^-?[0-9]+$ ]] || die "--offset-ms 는 정수(ms)여야 합니다: $offset_ms"
for f in "${files[@]}"; do [ -f "$f" ] || die "파일이 없습니다: $f"; done

env_file="${HAWKEYE_ENV:-$here/../deploy/.env}"
[ -f "$env_file" ] || die "env 파일이 없습니다: $env_file"
# shellcheck disable=SC1090
set -a; . "$env_file"; set +a
[ -n "${HAWKEYE_PUBLISH_USER:-}" ] && [ -n "${HAWKEYE_PUBLISH_PASS:-}" ] || die "HAWKEYE_PUBLISH_USER/PASS 가 비어 있습니다"

bind_ip="${HAWKEYE_BIND_IP:-}"
if [ -z "$bind_ip" ]; then
  bind_ip="$(tailscale ip -4 2>/dev/null | head -1)"
  [ -n "$bind_ip" ] || die "Tailscale IP 를 얻지 못했습니다. HAWKEYE_BIND_IP 를 주세요."
fi

encoder="${HAWKEYE_REPLAY_ENCODER:-}"
if [ -z "$encoder" ]; then
  # 1초짜리 시험 인코딩으로 GPU 인코더가 실제로 되는지 확인
  if ffmpeg -nostdin -hide_banner -loglevel error -f lavfi -i testsrc2=size=320x240:rate=10 -frames:v 10 \
       -c:v h264_nvenc -f null - >/dev/null 2>&1; then encoder=h264_nvenc; else encoder=libx264; fi
fi
case "$encoder" in
  h264_nvenc) enc=(-c:v h264_nvenc -preset p4 -tune ll -profile:v main) ;;
  libx264)    enc=(-c:v libx264 -preset veryfast -tune zerolatency -profile:v main) ;;
  *) die "HAWKEYE_REPLAY_ENCODER 는 h264_nvenc 또는 libx264: $encoder" ;;
esac
enc+=(-vf "scale=-2:'min(1080,ih)',format=yuv420p" -r 30 -g 30 -bf 0 -b:v 5M -maxrate 5M -bufsize 5M -an)

# 시차: 건너뛸 쪽 입력 앞에 -ss 를 둔다
skip1=0; skip2=0
if [ "$offset_ms" -gt 0 ]; then skip2="$offset_ms"; elif [ "$offset_ms" -lt 0 ]; then skip1=$(( -offset_ms )); fi
sec() { awk -v ms="$1" 'BEGIN { printf "%.3f", ms / 1000 }'; }

url() { echo "rtsp://$HAWKEYE_PUBLISH_USER:$HAWKEYE_PUBLISH_PASS@$bind_ip:8554/$1"; }
masked() { echo "rtsp://<송출 계정>@$bind_ip:8554/$1"; }

cmd=(ffmpeg -nostdin -hide_banner -loglevel warning -re -ss "$(sec "$skip1")" -i "${files[0]}")
[ ${#files[@]} -eq 2 ] && cmd+=(-re -ss "$(sec "$skip2")" -i "${files[1]}")
cmd+=(-map 0:v:0 "${enc[@]}" -f rtsp -rtsp_transport tcp "$(url cam1)")
[ ${#files[@]} -eq 2 ] && cmd+=(-map 1:v:0 "${enc[@]}" -f rtsp -rtsp_transport tcp "$(url cam2)")

echo "[replay] 인코더: $encoder, 시차: ${offset_ms}ms (cam1 건너뜀 ${skip1}ms, cam2 건너뜀 ${skip2}ms)"
echo "[replay] cam1 ← ${files[0]}"
[ ${#files[@]} -eq 2 ] && echo "[replay] cam2 ← ${files[1]}"
if [ "$dry" = 1 ]; then
  shown=("${cmd[@]}")
  for i in "${!shown[@]}"; do
    case "${shown[$i]}" in rtsp://*@*) shown[$i]="$(masked "${shown[$i]##*/}")" ;; esac
  done
  printf '%q ' "${shown[@]}"; echo
  exit 0
fi
exec "${cmd[@]}"
