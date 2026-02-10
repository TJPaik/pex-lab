# polygon_parser.py 상세 설명

`polygon_parser.py`는 `polygons.txt` 텍스트 포맷을
Python 자료구조로 변환하는 모듈입니다.

---

## 1. 파일 역할

입력 포맷:
```text
Net: <net_name>
  rect x1 y1 x2 y2 <layer_name>
  rect x1 y1 x2 y2 <layer_name>
```

출력 자료구조:
- `Dict[str, NetGeometry]`

이 구조는 이후
- `mesh.py`(메쉬 생성),
- `bem_solver.py`(ground 해석 모델),
- `fastercap_export.py`(FasterCap 입력 생성)
에서 공통으로 사용됩니다.

---

## 2. 데이터 클래스

## 2.1 `Rect`

```python
@dataclass
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int
    layer: str
```

좌표 규약:
- 정수 좌표(보통 centnm 단위)
- `x2 > x1`, `y2 > y1` 가정

유틸 메서드:
- `area() -> int`
- `width() -> int`
- `height() -> int`

---

## 2.2 `NetGeometry`

```python
@dataclass
class NetGeometry:
    name: str
    rects: List[Rect] = field(default_factory=list)
```

유틸 메서드:
- `rects_by_layer() -> Dict[str, List[Rect]]`
- `layers_present() -> set`

의미:
- 하나의 전기적 net에 속한 모든 rect 도형 묶음

---

## 3. 파서 함수

## 3.1 `parse_polygons(filepath)`

시그니처:
```python
def parse_polygons(filepath: str) -> Dict[str, NetGeometry]
```

동작 순서:
1. 파일을 줄 단위 순회
2. `Net:` 라인을 만나면 새 `NetGeometry` 생성
3. `rect` 라인을 만나면 `Rect` 파싱 후 현재 net에 append
4. 최종적으로 net dict 반환

파싱 규칙:
- 빈 줄은 무시
- `current_net is None` 상태에서는 `rect`를 무시
- 좌표는 `int()` 강제

주의점:
- 포맷이 깨진 줄에 대한 강한 예외 처리 로직은 없음
- 입력 파일 품질 보장이 중요

---

## 4. 타입 관점 요약

- 파일 전체 반환 타입:
  - `Dict[str, NetGeometry]`
- 각 net 내부:
  - `List[Rect]`
- 각 rect 좌표:
  - `int`

이 단순하고 일관된 타입 계약 덕분에,
후속 모듈은 geometry 처리에 집중할 수 있습니다.

---

## 5. 다른 모듈과의 연계

- `cap_extract.py`
  - 메인 파이프라인 시작점 입력 파싱
- `fastercap_export.py`
  - FasterCap `.qui` 생성용 geometry 입력
- `mesh.py`
  - panel mesh의 기본 도형 소스

