"""AI-Hub 2-097 파일명 규약 파서 (238-1 구매행동 + 238-2 이상행동 공용).

설명서의 정의:

    파일분류(단계)_행동분류_클래스분류_시나리오번호_지역_촬영장소_날짜_시간_카메라위치_
    데이터종류_촬영팀_행위자정보

그런데 **실제 파일명은 설명서와 다르게 여러 변형이 있다.** 실제 파일에서 확인한 것들:

    C_3_7_10_BU_DYA_08-23_13-47-40_CC_RGB_DF2_M2          이상행동, 12토큰
    C_1_1_11_BU_DYB_10-09_13-16-40_CE_DF1_F1_F1           매장이동, 12토큰, RGB 없음, 배우 2명
    C_1_1_20_BU_DYB_10-09_15-17-39_CF_RGB_DF1_F2_F2       매장이동, 13토큰
    C_1_1_16_BU_DYA_08_19_15_46_22_CF_RGB_DF1_M3_M3       매장이동, 16토큰, 날짜/시간이 '_' 로 쪼개짐

따라서 토큰 개수를 고정해서 파싱하면 안 된다. 앞쪽 6개를 고정으로 읽고,
**카메라 토큰(C + 영문 1자)을 찾아 기준점으로 삼은 뒤** 뒤쪽을 해석한다.

행동분류(2번째 토큰)
    1 = 매장이동, 2 = 구매행동   (238-1)
    3 = 이상행동                (238-2)

클래스분류(3번째 토큰) — 행동분류와 함께 봐야 클래스가 정해진다.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# (행동분류, 클래스분류) -> 내부 클래스명
FILENAME_CLASS_CODE: dict[tuple[str, str], str] = {
    # 238-2 이상행동 (실제 내려받은 3종)
    ("3", "7"): "fall",
    ("3", "8"): "broken",
    ("3", "12"): "theft",
    # 238-1 구매행동
    ("1", "1"): "moving",
    ("2", "2"): "select",
    ("2", "3"): "test",
    ("2", "4"): "buying",
    ("2", "5"): "return",
    ("2", "6"): "compare",
}

CATEGORY_KR = {"1": "매장이동", "2": "구매행동", "3": "이상행동"}

# XML track label(접두사) -> 내부 클래스명
XML_EVENT_LABEL = {
    "fall": "fall",
    "broken": "broken",
    "theft": "theft",
    "select": "select",
    "test": "test",
    "buying": "buying",
    "return": "return",
    "compare": "compare",
}

# 매장이동은 start/end 쌍이 아니라 단일 라벨이다(영상 전체가 해당 행동).
WHOLE_VIDEO_LABELS = {"moving"}

STORE_CODE = {
    "DYA": "대형유인매장A",
    "DYB": "대형유인매장B",
    "SYA": "소형유인매장A",
    "SYB": "소형유인매장B",
    "SMA": "소형무인매장A",
    "SMB": "소형무인매장B",
    "SMC": "소형무인카페C",
}

_ACTOR_RE = re.compile(r"^([MF])(\d+)$")
_CAMERA_RE = re.compile(r"^C[A-Z]$")
# 일부 파일은 카메라를 'CA' 가 아니라 소문자 한 글자 'a' 로 적어 놓았다(실제 178개 확인).
_CAMERA_ALT_RE = re.compile(r"^[a-z]$")
_CREW_RE = re.compile(r"^DF\d+$")


def _normalize_camera(tok: str) -> str:
    """'a' → 'CA' 로 정규화. 이미 'CA' 형태면 그대로."""
    if _CAMERA_ALT_RE.match(tok):
        return "C" + tok.upper()
    return tok


@dataclass(frozen=True)
class ClipName:
    stem: str
    stage: str          # C
    category: str       # 1 매장이동 / 2 구매행동 / 3 이상행동
    class_code: str     # 클래스분류 토큰
    class_name: str     # fall / broken / theft / moving / select / ...
    scenario: str
    region: str         # BU
    store: str          # DYA ...
    date: str           # MM-DD 로 정규화
    time: str           # HH-MM-SS 로 정규화
    camera: str         # CA ...
    modality: str       # RGB (없으면 빈 문자열)
    crew: str           # DF1 / DF2
    actor: str          # 대표 배우 코드 (첫 번째)
    actors: tuple[str, ...]
    actor_gender: str
    actor_index: str

    @property
    def store_kr(self) -> str:
        return STORE_CODE.get(self.store, self.store)

    @property
    def category_kr(self) -> str:
        return CATEGORY_KR.get(self.category, self.category)

    @property
    def dataset(self) -> str:
        """238-1(구매행동) / 238-2(이상행동) 구분."""
        return "238-2" if self.category == "3" else "238-1"

    @property
    def n_actors(self) -> int:
        return len(self.actors)

    @property
    def seconds(self) -> int:
        try:
            h, m, s = (int(x) for x in self.time.split("-"))
            return h * 3600 + m * 60 + s
        except Exception:
            return -1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["actors"] = list(self.actors)
        d["store_kr"] = self.store_kr
        d["category_kr"] = self.category_kr
        d["dataset"] = self.dataset
        d["n_actors"] = self.n_actors
        d["seconds"] = self.seconds
        return d


class NameParseError(ValueError):
    pass


def _normalize_datetime(tokens: list[str]) -> tuple[str, str]:
    """카메라 앞의 날짜/시간 토큰들을 (MM-DD, HH-MM-SS) 로 정규화한다.

    관측된 두 형태:
        ['10-09', '13-16-40']           -> ('10-09', '13-16-40')
        ['08', '19', '15', '46', '22']  -> ('08-19', '15-46-22')
    """
    if len(tokens) == 2 and "-" in tokens[0]:
        return tokens[0], tokens[1]
    if len(tokens) == 5 and all(t.isdigit() for t in tokens):
        return f"{tokens[0]}-{tokens[1]}", f"{tokens[2]}-{tokens[3]}-{tokens[4]}"
    if len(tokens) == 1:
        return tokens[0], ""
    # 알 수 없는 형태는 이어 붙여 보존한다(정보를 버리지 않는다)
    return "-".join(tokens[:2]), "-".join(tokens[2:])


def parse_stem(stem: str) -> ClipName:
    """확장자를 뺀 파일명을 파싱한다. 규약과 다르면 NameParseError."""
    parts = stem.split("_")
    if len(parts) < 10:
        raise NameParseError(f"토큰이 너무 적음({len(parts)}): {stem}")

    stage, category, class_code, scenario, region, store = parts[:6]

    # 기준점은 촬영팀 토큰(DF1/DF2)이다. 그 앞이 (RGB) 카메라, 뒤가 행위자다.
    # 카메라 토큰을 직접 찾는 방식은 소문자 한 글자('a') 표기 때문에 실패한다.
    crew_idx = None
    for i in range(7, len(parts)):
        if _CREW_RE.match(parts[i]):
            crew_idx = i
            break

    if crew_idx is not None:
        modality = ""
        cam_idx = crew_idx - 1
        if cam_idx > 6 and parts[cam_idx].upper() == "RGB":
            modality = parts[cam_idx]
            cam_idx -= 1
        crew = parts[crew_idx]
        actors = tuple(t for t in parts[crew_idx + 1:] if _ACTOR_RE.match(t))
    else:
        # 촬영팀 토큰이 없으면 카메라 토큰을 직접 찾는다(예전 경로).
        cam_idx = None
        for i in range(6, len(parts)):
            if _CAMERA_RE.match(parts[i]):
                cam_idx = i
                break
        if cam_idx is None:
            raise NameParseError(f"촬영팀(DF*)도 카메라 토큰도 찾지 못함: {stem}")
        rest = list(parts[cam_idx + 1:])
        modality = rest.pop(0) if rest and rest[0].upper() == "RGB" else ""
        crew = ""
        actors = tuple(t for t in rest if _ACTOR_RE.match(t))

    if cam_idx <= 5:
        raise NameParseError(f"카메라 위치를 특정하지 못함: {stem}")
    camera = _normalize_camera(parts[cam_idx])
    if not (_CAMERA_RE.match(camera) or len(camera) <= 3):
        raise NameParseError(f"카메라 토큰이 이상함({parts[cam_idx]}): {stem}")
    date, time = _normalize_datetime(parts[6:cam_idx])
    if not actors:
        raise NameParseError(f"행위자 정보를 찾지 못함: {stem}")

    class_name = FILENAME_CLASS_CODE.get((category, class_code))
    if class_name is None:
        raise NameParseError(
            f"알 수 없는 (행동분류={category}, 클래스분류={class_code}): {stem}")

    m = _ACTOR_RE.match(actors[0])
    gender = {"M": "male", "F": "female"}.get(m.group(1), "unknown")

    return ClipName(
        stem=stem, stage=stage, category=category, class_code=class_code,
        class_name=class_name, scenario=scenario, region=region, store=store,
        date=date, time=time, camera=camera, modality=modality, crew=crew,
        actor=actors[0], actors=actors, actor_gender=gender,
        actor_index=m.group(2),
    )


def take_key(n: ClipName) -> str:
    """촬영 사건(take) 후보 키.

    같은 연기를 여러 카메라로 찍으면 카메라마다 시작 시각이 몇 초 차이난다.
    따라서 시각은 키에서 빼고, build_index 에서 시간 근접도로 최종 병합한다.
    """
    return f"{n.category}|{n.class_code}|{n.store}|{n.date}|{n.scenario}|{n.actor}"
