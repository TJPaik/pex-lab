# cap_extract.py 상세 설명

이 문서는 `cap_extract.py`를 중심으로, 실제 실행 흐름/자료형/옵션/내부 로직을 상세하게 설명합니다.

---

## 1. 파일 역할

`cap_extract.py`는 레포의 메인 CLI 오케스트레이터입니다.

입력:
- `polygons.txt` 또는 `GDS` (`--from-gds`)
- 공정 스택 JSON (`--stack`, 생략 시 기본 stack)

출력:
- `CSV` (`net1, net2, coupling_cap_fF`)

내부적으로 수행하는 단계:
1. (선택) GDS -> polygons 변환
2. polygons 파싱
3. stack 로드
4. (선택) FasterCap 정합 preset 적용
5. (선택) 명시적 GND 평면 삽입
6. 메쉬 생성
7. BEM 해석
8. CSV 기록

---

## 2. import와 의존성

```python
from polygon_parser import NetGeometry, Rect, parse_polygons
from process_stack import LayerInfo, ProcessStack, default_sky130a_stack
from mesh import SKIP_LAYERS, mesh_all_nets
from bem_solver import BEMSolver
```

의미:
- `polygon_parser`: 입력 geometry 파싱 자료구조 (`Rect`, `NetGeometry`)
- `process_stack`: 레이어 높이/두께/유전체 정의
- `mesh`: 3D panel discretization
- `bem_solver`: 수치 해석 핵심 (행렬 구성/선형계 풀이/coupling 추출)

런타임 조건부 import:
```python
from gds_to_polygons import GDSToPolygons
```
- `--from-gds`일 때만 사용

---

## 3. 주요 자료형

### 3.1 `argparse.Namespace` (함수 `parse_args()` 반환)

핵심 필드 예:
- `input: str`
- `output: str`
- `from_gds: bool`
- `panel_size: float`
- `adaptive_mesh: bool`
- `near_field_samples: int`
- `uniform_epsilon: Optional[float]`
- `ground_model: Literal["analytic","matrix","both"]`
- `match_fastercap: bool`
- `explicit_ground_plane: bool`

### 3.2 geometry container

- `nets: Dict[str, NetGeometry]`
  - key: net 이름 (예: `VDD`, `net_1`)
  - value: 해당 net의 `Rect` 리스트

### 3.3 mesh 결과

- `panels: List[Panel]`
- `net_indices: Dict[str, int]`

### 3.4 결과

- `results: List[Tuple[str, str, float]]`
  - `(net1, net2, cap_fF)`

---

## 4. 함수별 상세

## 4.1 `parse_args()`

시그니처:
```python
def parse_args():
```

역할:
- 전체 CLI 옵션 정의
- 타입 강제(`type=float`, `type=int`)
- choice 제한(`--ground-model`)

핵심 옵션 묶음:

1) 입출력/입력형식
- `input`, `-o/--output`, `--from-gds`, `--polygons-out`

2) 메쉬 관련
- `--panel-size`, `--min-panel-size`
- `--adaptive-mesh`, `--proximity-distance`, `--proximity-factor`
- `--edge-refine-factor`, `--edge-refine-fraction`
- `--remove-internal-faces`

3) BEM 커널/근접 적분
- `--near-field-factor`, `--near-field-samples`
- `--uniform-epsilon`

4) ground 처리
- `--no-ground-plane`
- `--ground-net`, `--ground-model`
- `--signal-scale`, `--ground-scale`
- `--explicit-ground-plane` 계열 옵션

5) FasterCap 정합 프리셋
- `--match-fastercap`

`--match-fastercap`는 사용자가 옵션을 여러 개 직접 맞추는 대신,
내부에서 일괄 세팅을 강제해 비교 정합 실수를 줄여줍니다.

---

## 4.2 `write_csv(results, output_path)`

시그니처:
```python
def write_csv(results: List[Tuple[str, str, float]], output_path: str):
```

입력 타입:
- `results`: `(str, str, float)` 튜플 리스트
- `output_path`: 파일 경로

동작:
- CSV 헤더를 고정(`net1, net2, coupling_cap_fF`)
- 커패시턴스를 `%.5f` 문자열로 직렬화

주의:
- downstream 비교 도구(`compare_fc_vs_bem.py`)는 이 헤더를 전제로 동작

---

## 4.3 `add_explicit_ground_plane(...)`

시그니처:
```python
def add_explicit_ground_plane(
    nets: Dict[str, NetGeometry],
    stack: ProcessStack,
    ground_net: str,
    ground_layer: str,
    margin_um: float,
    ground_z_bottom: float = None,
    ground_thickness: float = 0.2,
) -> bool:
```

역할:
- 입력 net들의 bbox를 계산하고, 그 바깥 margin을 준 하나의 큰 사각형을 GND net으로 삽입

세부 절차:
1. `ground_z_bottom`이 있으면 `stack.layers[ground_layer]`를 동적으로 생성/재정의
2. 입력 net 전체를 순회해 유효 도체(`SKIP_LAYERS` 제외, thickness>0) bbox 산출
3. margin(um)을 내부 좌표 단위(`centnm`)로 환산
4. `Rect` 생성 후 `nets[ground_net]`에 append 또는 신규 생성

반환:
- 성공 시 `True`
- 유효 bbox가 없으면 `False`

왜 필요한가:
- image-ground 대신 유한 크기 도체를 실제 패널로 넣어 경계조건을 물리적으로 실험할 때 사용

실무 주의:
- 유한 GND 패치는 신호-신호 coupling을 왜곡할 수 있음
- FasterCap reference가 동일 경계조건이 아닐 경우 비교 지표가 악화될 수 있음

---

## 4.4 `main()`

시그니처:
```python
def main():
```

`main()`은 파이프라인 제어를 모두 담당합니다.

### 4.4.1 Step 1: 입력 로드/변환

- `--from-gds`이면 `GDSToPolygons.run()` 실행
- 아니면 입력 파일을 바로 polygons 경로로 사용

산출물:
- `polygons_path: str`

### 4.4.2 Step 2: polygons 파싱

```python
nets = parse_polygons(polygons_path)
```

- `Dict[str, NetGeometry]` 생성
- net 개수와 총 rect 수를 로그로 출력

### 4.4.3 Step 3: stack 로드

- `--stack` 지정 시 `ProcessStack.from_json()`
- 생략 시 `default_sky130a_stack()`

### 4.4.4 FasterCap 정합 preset 적용 (`--match-fastercap`)

아래를 강제:
- `no_ground_plane=True`
- `ground_model="matrix"`
- `remove_internal_faces=True`
- `uniform_epsilon=stack.get_effective_epsilon(1.5)` (사용자가 직접 지정 안 했을 때)

의도:
- `fastercap_export.py` 쪽 가정(대표 유전율 + 내부면 제거 + matrix row GND)에 맞춰 조건 정합

### 4.4.5 (선택) 명시적 GND 평면 삽입

- `add_explicit_ground_plane(...)` 호출
- 성공 시 `net_max_panel_size = {ground_net: value}` 설정 가능

### 4.4.6 Step 4: 메쉬 생성

```python
panels, net_indices = mesh_all_nets(...)
```

입력 옵션이 거의 그대로 전달됩니다:
- panel size, adaptive mesh, edge refine, internal face 제거 등

### 4.4.7 Step 5: BEM 풀이

내부 nested function:
```python
def solve_for_mode(mode: str):
```

핵심:
- `BEMSolver(...)` 인스턴스 생성
- `mode == "analytic"`면 `nets_data`를 넘겨 해석적 ground model 사용
- `mode == "matrix"`면 `nets_data=None`으로 matrix row 기반 ground 사용

중복 방지 로직:
- 명시적 GND 평면을 이미 넣었는데 image-ground가 켜져 있으면 자동으로 꺼서 이중 ground 경계 반영을 막음

### 4.4.8 Step 6: 출력

- `ground_model == both`면 `_analytic.csv`, `_matrix.csv` 두 개 출력
- 아니면 지정 output 하나만 출력
- 마지막에 `Total time` 출력

---

## 5. `solve_for_mode`의 입력/출력 타입

입력:
- `mode: str` (`"analytic"` 또는 `"matrix"`)

출력:
- `results_local: List[Tuple[str, str, float]]`

간접 입력:
- 외부 스코프의 `panels`, `net_indices`, `stack`, `args`

이 구조의 장점:
- 동일 mesh/동일 solver 설정에서 ground 모델만 바꿔 비교 가능

---

## 6. 수학 모델을 어디서 호출하는가

`cap_extract.py` 자체는 수학식을 직접 계산하지 않고,
아래 컴포넌트 호출로 조합합니다.

- 메쉬 수학/기하 분할: `mesh.py`
- P 행렬 구성, 선형계 풀이, C 추출: `bem_solver.py`

즉 `cap_extract.py`는 **Controller layer**입니다.

---

## 7. 실행 모드별 의미

### 7.1 연구/튜닝 모드

- `--adaptive-mesh`
- `--near-field-factor`, `--near-field-samples`
- `--edge-refine-*`

목적: 수치 정확도 향상

### 7.2 FasterCap 정합 모드

- `--match-fastercap`

목적: 비교 기준과 경계조건/유전율 가정을 맞춰 오차 감소

### 7.3 물리 실험 모드(명시적 GND)

- `--explicit-ground-plane` 계열

목적: 유한 GND 패치 경계조건 실험

---

## 8. 예외/디버깅 포인트

1. `panels=[]` 발생
- 원인: 모든 도형이 skip layer이거나 thickness<=0
- 대응: layer map/stack 파일 확인

2. 명시적 ground 삽입 실패
- `add_explicit_ground_plane`이 `False`
- 원인: 유효한 bbox를 만들 도체가 없음

3. ground 중복 모델링
- 명시적 GND + image-ground 동시 적용은 물리 중복
- 현재 코드가 자동으로 image-ground를 끔

4. 결과 CSV pair 누락
- `--min-cap`이 크면 작은 coupling row가 필터됨

---

## 9. 실무 팁

- FasterCap 비교 목적이면 먼저 `--match-fastercap`로 baseline 확보
- 이후 `panel-size`, `near-field-samples`를 천천히 올려 수렴 확인
- `--ground-model both`로 analytic/matrix ground를 동시에 뽑아 민감도 점검

