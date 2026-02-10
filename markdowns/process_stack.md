# process_stack.py 상세 설명

이 문서는 공정 스택 정의 모듈 `process_stack.py`를 설명합니다.

---

## 1. 파일 역할

`process_stack.py`는 커패시턴스 계산에 필요한 물리 정보를 제공합니다.

핵심 정보:
- 도체 레이어별 높이/두께/비아 여부
- 유전체 층별 `epsilon_r`
- 입력 좌표 스케일(`scale_to_um`)

이 파일의 출력 객체(`ProcessStack`)는
- `mesh.py`에서 3D extrude 기준으로,
- `bem_solver.py`에서 유전율 조회 및 해석적 ground 모델에 사용됩니다.

---

## 2. 데이터 클래스

## 2.1 `LayerInfo`

```python
@dataclass
class LayerInfo:
    name: str
    z_bottom: float
    thickness: float
    is_via: bool = False
```

파생 프로퍼티:
```python
@property
def z_top(self) -> float:
    return self.z_bottom + self.thickness
```

의미:
- 레이어를 `z_bottom ~ z_top` 구간의 도체로 해석

---

## 2.2 `DielectricLayer`

```python
@dataclass
class DielectricLayer:
    z_bottom: float
    z_top: float
    epsilon_r: float
```

의미:
- z 구간별 유전율 분포 정의
- `ProcessStack.get_effective_epsilon(z)`에서 참조

---

## 2.3 `ProcessStack`

```python
@dataclass
class ProcessStack:
    name: str
    units: str
    scale_to_um: float
    layers: Dict[str, LayerInfo]
    dielectrics: List[DielectricLayer]
    substrate_epsilon_r: float = 11.7
```

핵심 필드:
- `scale_to_um`: polygons 좌표 단위를 um로 환산하는 계수
  - 예: `centnm` 기준이면 `0.01`

---

## 3. 메서드 상세

## 3.1 `ProcessStack.from_json(path)`

입력:
- `path: str`

출력:
- `ProcessStack`

동작:
1. JSON 로드
2. `layers` dict를 `LayerInfo` 객체 dict로 변환
3. `dielectrics` list를 `DielectricLayer` list로 변환
4. `ProcessStack` 인스턴스 반환

요구 JSON 스키마(핵심):
- `name`, `units`, `scale_to_um`
- `layers: {layer_name: {name, z_bottom, thickness, is_via?}}`
- `dielectrics: [{z_bottom, z_top, epsilon_r}, ...]`

---

## 3.2 `ProcessStack.get_effective_epsilon(z)`

입력:
- `z: float` (um)

출력:
- 해당 높이에서의 `epsilon_r: float`

동작:
- `dielectrics`를 순회하여 `z_bottom <= z < z_top` 조건 만족 구간 반환
- 없으면 `1.0` 반환(상부 공기 가정)

주의:
- 이 함수는 z 위치의 로컬 유전율을 주는 단순 모델
- 복잡한 경계조건 해석의 full Green 함수 대체는 아님

---

## 3.3 `ProcessStack.get_layer(name)`

입력:
- `name: str`

출력:
- `Optional[LayerInfo]`

용도:
- parser/mesher 단계에서 레이어 유효성 검사

---

## 4. `default_sky130a_stack()`

역할:
- 기본 stack를 자동 로딩하거나 fallback 하드코드 제공

우선순위:
1. `sky130a_stack_from_pdk.json` 존재 시 로드
2. 없으면 `sky130a_stack.json` 로드
3. 둘 다 없으면 코드 내 하드코드 테이블 사용

하드코드 fallback 내용:
- sky130A 주요 레이어(`poly`, `locali`, `metal1~4`, `via*` 등)
- 유전체 구간(`FOX`, `LINT`, `NILD*`, `TOPOX+air`) 근사값

---

## 5. 다른 파일과의 데이터 계약

1. `mesh.py`
- `scale_to_um`, `get_layer()` 사용
- rect를 실제 3D 좌표로 변환

2. `bem_solver.py`
- `get_effective_epsilon(z)` 사용
- 해석적 ground cap 계산 시 `dielectrics` 전체 순회

3. `cap_extract.py`
- CLI에서 stack JSON 선택/로드

---

## 6. 실무 팁

- 가능한 `generate_stack_from_pdk.py`로 생성한 최신 JSON 사용 권장
- `scale_to_um`가 입력 geometry 단위와 맞지 않으면 모든 수치가 망가짐
- layer 이름 불일치(예: GDS layer map vs stack layer key) 확인 필수

