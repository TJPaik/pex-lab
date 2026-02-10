# gds_to_polygons.py 상세 설명

이 문서는 `gds_to_polygons.py`를 상세히 설명합니다.

---

## 1. 파일 역할

`gds_to_polygons.py`는 GDS 레이아웃을 읽어서
레포 표준 입력 포맷인 `polygons.txt`로 변환합니다.

핵심 기능:
1. GDS 계층 flatten
2. 도형(layer/datatype) -> 내부 layer 이름 매핑
3. rect 추출
4. 텍스트 label 추출
5. overlap + via 규칙으로 connectivity 구성
6. MOS diffusion/poly 교차 분할(소스/드레인 분리)
7. net 이름 할당 후 파일 출력

---

## 2. 외부 라이브러리

- `gdstk`: GDS 읽기/flatten
- `numpy`: polygon 좌표 처리

Import 실패 시 즉시 종료하도록 되어 있습니다.

---

## 3. 상수/매핑 테이블

## 3.1 `SKY130A_LAYER_MAP`

- key: `(gds_layer: int, datatype: int)`
- value: 내부 layer 이름(`metal1`, `via1`, `poly`, `nsubdiff` 등)

## 3.2 `SKY130A_LABEL_LAYER_TO_CONDUCTOR`

- label(text) layer를 어느 도체층 net label로 해석할지 정의

## 3.3 `VIA_CONNECTIVITY`

- via layer가 연결하는 하부/상부 도체층 규칙
- 예: `via1 -> (metal1, metal2)`

## 3.4 `LICON_DIFF_CONNECTIVITY`

- sky130 특성상 `polycont`(licon1)가 poly 접촉/확산 접촉을 겸함
- poly와 겹치지 않는 licon은 diffusion contact로 처리

---

## 4. 데이터 구조

## 4.1 `GDSRect`

```python
@dataclass
class GDSRect:
    x1: int
    y1: int
    x2: int
    y2: int
    layer: str
    gds_layer: int
    gds_datatype: int
```

- 내부 처리 좌표는 정수 좌표(기본 centnm 스케일)

## 4.2 `UnionFind`

- connectivity grouping을 위한 disjoint-set
- 메서드:
  - `find(x) -> int`
  - `union(x, y)`

---

## 5. 유틸 함수

## 5.1 `polygon_to_rect(vertices, scale)`

입력:
- `vertices: np.ndarray (N,2)`
- `scale: float`

출력:
- `Optional[(x1,y1,x2,y2)]`

동작:
- polygon의 min/max bounding box를 int로 환산
- 축 정렬 rectangle로 간주
- 폭/높이 <= 0이면 `None`

주의:
- 비직사각형 polygon도 bbox 직사각형으로 환원됩니다.

## 5.2 `rects_overlap(r1, r2)`

- 두 rect가 겹치거나 경계를 접하면 `True`
- connectivity union 조건에 사용

## 5.3 `point_in_rect(px, py, r)`

- label 점이 도체 rect 내부/경계에 있는지 판단

---

## 6. `GDSToPolygons` 클래스

## 6.1 생성자 `__init__(gds_path, layer_map=None, coord_scale=None)`

입력:
- `gds_path: str`
- `layer_map: Optional[dict]`
- `coord_scale: Optional[float]`

동작:
1. `gdstk.read_gds()` 로드
2. layer map 설정
3. 좌표 스케일 결정
   - 명시값 있으면 사용
   - 없으면 `lib.unit / 1e-8` 자동 계산
4. top-level cell 선택

좌표 스케일 의미:
- GDS 단위를 내부 정수 단위(centnm)로 맞추기 위한 배율

---

## 6.2 `extract_all()`

반환:
- `rects: List[GDSRect]`
- `labels: Dict[str, List[(x, y, conductor_layer)]]`

절차:
1. top cell copy 후 `flatten()`
2. polygon 순회
   - layer map으로 내부 layer 이름 찾기
   - `polygon_to_rect` 변환
   - `GDSRect` append
3. label 순회
   - label layer를 conductor layer로 해석
   - 좌표를 정수 스케일로 환산
   - `labels[label_text].append((x,y,layer))`

---

## 6.3 `split_diff_at_poly(rects)`

입력:
- `List[GDSRect]`

출력:
- 분할 반영된 `List[GDSRect]`

배경:
- poly가 diffusion을 가로지르는 MOS 구조에서,
  source/drain이 전기적으로 분리되어야 함

동작:
1. split 대상 diffusion layer 선택 (`ndiff`, `nsubdiff`)
2. 각 diffusion rect에 대해 poly 교차 구간(`gate_x1~gate_x2`) 수집
3. gate 구간을 제외한 segment로 diffusion rect 분할

효과:
- net connectivity가 실제 MOS 구조에 더 가깝게 됨

---

## 6.4 `build_connectivity(rects, labels)`

입력:
- `rects: List[GDSRect]`
- `labels: Dict[str, List[(x,y,layer)]]`

출력:
- `Dict[str, List[GDSRect]]` (net 이름 -> rect 목록)

내부 단계:

### Step 1: 동일 layer overlap union
- 같은 layer 내 rect들을 x 정렬 후 sweep
- 겹치거나 접하면 union

### Step 2: via connectivity union
- 각 via rect가 lower/upper 도체와 겹치면 union

### Step 2b: licon 특수 처리
- licon(`polycont`)이 poly와 겹치면 poly contact
- poly와 겹치지 않으면 diffusion contact 후보
- diffusion layer와 겹치면 union

### Step 3: label로 net seed
- label point가 해당 layer rect 안에 있으면 그 UF root에 net 이름 부여

### Step 4: 그룹화 + 이름 할당
- UF root별로 member 모음
- label 없는 그룹은 `net_1`, `net_2`, ... 자동 이름

---

## 6.5 `run(output_path)`

전체 파이프라인 실행:
1. `extract_all()`
2. `split_diff_at_poly()`
3. `build_connectivity()`
4. `write_polygons_txt()`

중간 로그:
- polygon/label 수
- split 후 polygon 수
- net 수

---

## 6.6 `write_polygons_txt(net_rects, output_path)`

정적 메서드.

출력 포맷:
```text
Net: <name>
  rect x1 y1 x2 y2 <layer>
```

net 이름 정렬 후 기록합니다.

---

## 7. CLI `main()`

옵션:
- `gds_file`
- `-o/--output`
- `--scale`

유효성:
- `--scale <= 0`이면 argparse error

---

## 8. 정확도/신뢰성 관점 주의

1. polygon -> rect 환원
- 비정형 polygon에서 정보 손실 가능

2. overlap 기반 connectivity
- 실제 공정/DRC 규칙의 모든 예외를 반영하지는 않음

3. licon 특수 규칙
- sky130의 핵심 함정을 처리하지만,
  PDK 버전 변화 시 추가 규칙이 필요할 수 있음

---

## 9. cap_extract.py와의 연결

`cap_extract.py --from-gds`를 사용하면 이 모듈이 자동 호출되어
`polygons.txt`가 생성됩니다.

즉 이 파일은 **layout 입력 전처리의 관문**입니다.

