# mesh.py 상세 설명

이 문서는 `mesh.py`의 3D 표면 메쉬 생성 로직을 상세히 설명합니다.

---

## 1. 파일 역할

`mesh.py`는 net별 직사각형 도형(`Rect`)을 공정 스택 높이로 extrude하여,
BEM에서 사용하는 panel(`Panel`) 리스트로 변환합니다.

핵심 기능:
- 축 분할(균일/edge refine)
- top/bottom/sidewall panel 생성
- adaptive mesh (근접도 기반)
- internal face 제거
- net별 panel size 오버라이드

---

## 2. 데이터 구조

## 2.1 `Panel` dataclass

```python
@dataclass
class Panel:
    cx: float
    cy: float
    cz: float
    nx: float
    ny: float
    nz: float
    area: float
    net_idx: int
    u_axis: int = 0
    v_axis: int = 1
    du: float = 0.0
    dv: float = 0.0
```

필드 의미:
- `(cx,cy,cz)`: 패널 중심 좌표 (um)
- `(nx,ny,nz)`: 법선 벡터
- `area`: 패널 면적 (um^2)
- `net_idx`: 소속 net 인덱스
- `(u_axis, v_axis, du, dv)`: near-field subcell 적분용 로컬 좌표 정보

---

## 3. 핵심 상수

```python
SKIP_LAYERS = {"nwell", "nmos", "pmos"}
```

- 물리 도체로 모델링하지 않는 레이어 필터
- layer thickness가 0인 경우와 함께 panel 생성 제외

---

## 4. 함수별 상세

## 4.1 `_axis_segments(start, end, max_size, edge_refine_factor, edge_refine_fraction)`

출력:
- `List[(seg_start, seg_end, seg_center, seg_len)]`

기능:
- 1차원 구간 `[start,end]`를 panel 폭 제약에 맞춰 분할
- edge refine 옵션이 켜지면 양 끝에 더 작은 segment 배치

기본 분할:
$$
N = \lceil \frac{span}{max\_size} \rceil,
\quad d = \frac{span}{N}
$$

edge refine 분할:
- 전체 길이 중 `edge_refine_fraction` 비율을 양끝 edge 영역으로 사용
- edge는 `max_size * edge_refine_factor`로 분할
- center는 `max_size`로 분할

---

## 4.2 `subdivide_horizontal(...)`

입력:
- 직사각형 `x1,y1,x2,y2`
- z 고정면(`top` 또는 `bottom`)
- 법선 부호 `nz_sign`

출력:
- `List[Panel]`

동작:
- x/y 축을 각각 `_axis_segments`로 분할
- Cartesian product로 작은 panel 생성

---

## 4.3 `subdivide_vertical(...)`

입력:
- 벽면 span (`u1..u2`), 높이 (`z1..z2`), 고정 좌표(`fixed`)
- 방향 문자열 `orient in {x+,x-,y+,y-}`

출력:
- `List[Panel]`

동작:
- wall span 축 + z 축 분할
- orientation에 따라 법선 벡터 지정
- x-wall/y-wall에 맞게 center 좌표 축 배치

---

## 4.4 `_panel_key(corners)` / `_rect_face_keys(rect, layer, scale)`

목적:
- face를 corner set으로 canonical key화하여 동일 면 검출

`_rect_face_keys` 반환:
- `{"top": key, "bot": key, "front": key, "back": key, "right": key, "left": key}`
- via 레이어는 top/bottom을 만들지 않음

용도:
- internal face 제거 시 공유면 판별

---

## 4.5 `mesh_rect_3d(...)`

입력:
- `Rect`, `LayerInfo`, `net_idx`, `scale`, `max_size`
- 선택적으로 `external_face_keys`

출력:
- 해당 rect의 `List[Panel]`

동작:
1. rect를 공정 높이로 3D box화
2. top/bottom 생성(비-via)
3. 4개 sidewall 생성
4. `external_face_keys`가 주어지면 그 안의 face만 panel화

---

## 4.6 `_box_distance(a, b)`

입력:
- 3D AABB box 두 개: `(x1,y1,z1,x2,y2,z2)`

출력:
- 최소 거리(float)

수식:
$$
d_x=\max(a_{x1}-b_{x2},\; b_{x1}-a_{x2},\;0)
$$
$$
d=\sqrt{d_x^2+d_y^2+d_z^2}
$$

---

## 4.7 `_compute_adaptive_rect_panel_sizes(...)`

입력:
- net별 rect geometry
- `max_panel_size`, `min_panel_size`
- `proximity_distance`, `proximity_factor`

출력:
- `Dict[net_name, Dict[rect_index, local_panel_size]]`

절차:
1. 각 rect의 3D box 생성
2. 다른 net rect들과의 최소거리 `min_dist` 계산
3. 아래 식으로 local size 결정:
$$
ratio = \min(\frac{min\_dist}{prox\_dist},1)
$$
$$
s_{local}=s_{max}(p+(1-p)ratio)
$$
4. `min/max panel size` 경계로 clamp

의도:
- net 간 가까운 구역일수록 더 작은 panel 사용

---

## 4.8 `mesh_net(...)`

입력:
- `net: NetGeometry`, `net_idx`, `stack`
- 선택적 `rect_panel_sizes`
- `remove_internal_faces`

출력:
- `List[Panel]`

동작:
1. `remove_internal_faces=True`이면
   - 모든 face key 카운트
   - 등장 횟수 1인 face만 external face로 채택
2. rect마다 local panel size 결정
3. `mesh_rect_3d` 호출

---

## 4.9 `mesh_all_nets(...)`

시그니처(핵심):
```python
def mesh_all_nets(
    nets: Dict[str, NetGeometry],
    stack: ProcessStack,
    max_panel_size: float = 1.0,
    min_panel_size: float = 0.2,
    adaptive_mesh: bool = False,
    proximity_distance: float = 2.0,
    proximity_factor: float = 0.6,
    remove_internal_faces: bool = False,
    net_max_panel_size: Optional[Dict[str, float]] = None,
    edge_refine_factor: float = 1.0,
    edge_refine_fraction: float = 0.0,
) -> Tuple[List[Panel], Dict[str, int]]
```

출력:
- `all_panels: List[Panel]`
- `net_indices: Dict[str, int]`

중요 로직:
- net 이름 정렬 후 인덱싱
- `adaptive_mesh`면 rect-level local size 사전 계산
- `net_max_panel_size` 지정된 net(예: explicit GND)은 adaptive를 끄고 고정 panel size 사용
- panel이 하나도 없는 net은 인덱스에 추가하지 않음

---

## 5. 정확도/성능 관점 포인트

1. `panel-size`를 줄이면
- 장점: 형상 근사/근접장 표현 개선
- 단점: panel 수 증가 -> 메모리/시간 증가

2. `remove_internal_faces`
- 도체 내부 공유면 제거로 비물리 면 제거
- FasterCap 정합에서 매우 중요

3. `edge_refine_*`
- edge field 집중 구역 해상도 향상
- low-cap/근접 coupling 개선에 유리

4. adaptive mesh
- 전체를 무조건 미세화하지 않고 필요한 곳만 정밀화

---

## 6. cap_extract.py와의 연결

`cap_extract.py`는 `mesh_all_nets()`로 panel을 생성하고,
생성된 `(panels, net_indices)`를 `BEMSolver`에 바로 넘깁니다.

즉 mesh 품질은 곧 solver 정확도/시간을 지배합니다.

