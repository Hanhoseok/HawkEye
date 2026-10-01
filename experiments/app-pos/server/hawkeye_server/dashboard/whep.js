// WHEP(WebRTC-HTTP Egress Protocol)로 중계기(MediaMTX)의 영상을 받아 <video> 에 띄운다.
// 브라우저는 RTSP 를 못 틀기 때문에, 중계기가 같은 영상을 WebRTC 로도 내보낸다.

function waitIceGathering(pc, timeoutMs = 3000) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => { pc.removeEventListener("icegatheringstatechange", check); resolve(); };
    const check = () => { if (pc.iceGatheringState === "complete") done(); };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(done, timeoutMs);
  });
}

/** whepUrl 에 연결해 video 에 재생한다. 끊기면 onLost 를 부른다. */
export async function playWhep(video, whepUrl, onLost) {
  const pc = new RTCPeerConnection();
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });
  pc.ontrack = (e) => { if (video.srcObject !== e.streams[0]) video.srcObject = e.streams[0]; };
  pc.onconnectionstatechange = () => {
    if (["failed", "disconnected", "closed"].includes(pc.connectionState)) onLost?.(pc.connectionState);
  };

  await pc.setLocalDescription(await pc.createOffer());
  await waitIceGathering(pc);
  const r = await fetch(whepUrl, {
    method: "POST", headers: { "Content-Type": "application/sdp" }, body: pc.localDescription.sdp,
  });
  if (!r.ok) {
    pc.close();
    throw new Error(r.status === 404 ? "카메라 영상이 아직 없습니다 (중계기에 cam1 송출 확인)" : `중계기 응답 ${r.status}`);
  }
  await pc.setRemoteDescription({ type: "answer", sdp: await r.text() });
  return pc;
}
