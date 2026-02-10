# Python BEM 기반 기생 커패시턴스 추출기 (FasterCap 정합 중심)

이 저장소는 GDS/폴리곤 입력에서 3D 패널 메쉬를 만들고, BEM(Boundary Element Method)으로 커패시턴스 행렬을 계산해 `CSV(net1, net2, coupling_cap_fF)`를 출력합니다.

이 README는 **현재 레포에서 얻은 최종 best 정합 설정**(캘리브레이션 없이)을 기준으로 작성했습니다.

---

## 1. 최종 best 결과 (캘리브레이션 없음)

기준 파일:
- FasterCap reference: `OTA_FF_992_0_fastercap_ref_a0p01.csv`
- Python BEM best: `OTA_FF_992_0_bem_match3_p0p38_nf1p6_s7_e03f025.csv`
- 비교 리포트: `OTA_FF_992_0_fc_vs_bem_match3_p0p38_nf1p6_s7_e03f025.txt`

주요 지표:
- All-pair RMSE: **0.112002 fF**
- All-pair MAE: **0.062651 fF**
- Signal-only RMSE: **0.123801 fF**
- GND 포함 공통 pair 수: **55**
- FasterCap-only/BEM-only pair: **0 / 0**

---

## 2. 빠른 시작

### 2.1 의존성

`requirements.txt`:
- `numpy>=1.24`
- `scipy>=1.10`
- `gdstk>=0.9`

설치 예:
```bash
pip install -r requirements.txt
```

### 2.2 최종 best 설정으로 실행

```bash
PYENV_VERSION=torch pyenv exec python cap_extract.py OTA_FF_992_0.gds \
  --from-gds \
  --polygons-out /tmp/OTA_FF_992_0_from_gds_polygons.txt \
  --stack sky130a_stack_from_pdk.json \
  -o OTA_FF_992_0_bem_match3_p0p38_nf1p6_s7_e03f025.csv \
  --match-fastercap \
  --panel-size 0.38 \
  --min-panel-size 0.038 \
  --adaptive-mesh \
  --proximity-distance 2.0 \
  --proximity-factor 0.8 \
  --edge-refine-factor 0.3 \
  --edge-refine-fraction 0.25 \
  --near-field-factor 1.6 \
  --near-field-samples 7 \
  --min-cap 1e-6
```

### 2.3 FasterCap reference 생성 (ground truth)

먼저 PDK 기반 stack JSON을 생성합니다.

```bash
PYENV_VERSION=torch pyenv exec python generate_stack_from_pdk.py \
  --pdk-root /home/paiktj/skywater-pdk \
  -o sky130a_stack_from_pdk.json
```

그 다음 FasterCap을 실행해서 reference CSV를 만듭니다.

```bash
PYENV_VERSION=torch pyenv exec python fastercap_export.py OTA_FF_992_0.gds \
  --from-gds \
  --polygons-out /tmp/OTA_FF_992_0_fc_from_gds_polygons.txt \
  --stack sky130a_stack_from_pdk.json \
  -o /tmp/OTA_FF_992_0_fastercap_a0p01 \
  --run \
  --accuracy 0.01 \
  --timeout 21600 \
  --galerkin \
  --csv-out OTA_FF_992_0_fastercap_ref_a0p01.csv \
  > OTA_FF_992_0_fastercap_ref_a0p01.log 2>&1
```

생성 파일:
- `OTA_FF_992_0_fastercap_ref_a0p01.csv` (reference 커플링 결과)
- `OTA_FF_992_0_fastercap_ref_a0p01.log` (FasterCap 실행 로그)

참고: 한 번에 reference + BEM + 비교까지 실행하려면 아래 스크립트를 써도 됩니다.

```bash
./run_high_accuracy_compare.sh OTA_FF_992_0.gds /home/paiktj/skywater-pdk 0.01 21600 1.0
```

### 2.4 FasterCap와 비교 + scatter 생성

```bash
PYENV_VERSION=torch pyenv exec python compare_fc_vs_bem.py \
  --fastercap OTA_FF_992_0_fastercap_ref_a0p01.csv \
  --bem OTA_FF_992_0_bem_match3_p0p38_nf1p6_s7_e03f025.csv \
  --out-csv OTA_FF_992_0_fc_vs_bem_match3_p0p38_nf1p6_s7_e03f025.csv \
  --summary OTA_FF_992_0_fc_vs_bem_match3_p0p38_nf1p6_s7_e03f025.txt \
  --scatter OTA_FF_992_0_fc_vs_bem_match3_p0p38_nf1p6_s7_e03f025.png \
  --lowcap-threshold 0.5
```

---

## 3. 전체 계산 파이프라인

1. `gds_to_polygons.py`
- GDS를 flatten하고 layer map을 적용해 직사각형 집합으로 변환
- label + via overlap + 동일층 overlap으로 net connectivity 구성
- `polygons.txt` 형식 출력

2. `polygon_parser.py`
- `polygons.txt`를 `Dict[str, NetGeometry]`로 파싱

3. `process_stack.py` / `generate_stack_from_pdk.py`
- 도체/유전체 물성(`z_bottom`, `thickness`, `epsilon_r`) 로드

4. `mesh.py`
- 각 직사각형을 3D 박스로 extrude
- 표면을 panel로 분할(균일/적응/edge refine)
- 선택적으로 internal face 제거

5. `bem_solver.py`
- 패널 간 potential coefficient matrix `P` 구성
- 선형계 `P * sigma = V`를 각 excitation마다 풀이
- net-level Maxwell C matrix 계산
- pair coupling 및 GND row 추출

6. `cap_extract.py`
- 위 단계를 orchestration하고 CSV 출력

---

## 4. 핵심 수학 모델과 코드 매핑

## 4.1 패널 기반 BEM 기본식

패널 `j`의 균일 전하밀도에 의해 패널 `i` 중심 전위 기여를

a. 오프대각 근사로

$$
P_{ij} = \frac{A_j}{4\pi\epsilon_0\,\bar\epsilon_{ij}\,r_{ij}} \quad (i\neq j)
$$

b. 대각(Self) 항은 직사각형 패널 자기 적분식으로 둡니다.

구현 위치:
- 오프대각: `bem_solver.py`의 `build_coefficient_matrix()`
- 대각(Self): `bem_solver.py`의 `_self_term_rect()`

## 4.2 이미지 전하(ground plane) 모델

이미지 모델을 켜면
$$
P_{ij} \leftarrow P_{ij} - \frac{A_j}{4\pi\epsilon_0\,\bar\epsilon_{ij}\,r'_{ij}}
$$
를 적용합니다.

구현 위치:
- `bem_solver.py`의 `build_coefficient_matrix()` (`use_ground_plane`)

> FasterCap 정합 모드(`--match-fastercap`)에서는 이 항을 끕니다.

## 4.3 Near-field 정밀화

가까운 패널 쌍에 대해서는 소스 패널을 `n x n` subcell로 쪼개 직접 합산:
$$
P_{ij}^{\text{near}} \approx \sum_{k=1}^{n^2}
\frac{\Delta A_k}{4\pi\epsilon_0\,\bar\epsilon_{ij}\,|\mathbf r_i-\mathbf r_{j,k}|}
$$

구현 위치:
- `bem_solver.py`의 `_apply_near_field_correction()`
- 옵션: `--near-field-factor`, `--near-field-samples`

## 4.4 선형계 풀이와 Maxwell 행렬

net `k`를 1V, 나머지를 0V로 둔 excitation 벡터 `V^{(k)}`에 대해:
$$
P\sigma^{(k)} = V^{(k)}
$$

그 다음 net `m`의 총 전하:
$$
Q_m^{(k)} = \sum_{i \in m} \sigma_i^{(k)} A_i
$$

Maxwell 행렬 원소:
$$
C_{mk} = Q_m^{(k)}
$$

구현 위치:
- `bem_solver.py`의 `solve_capacitance_matrix()`

## 4.5 커플링/그라운드 추출

- Net-to-net coupling:
$$
C_{ij}^{\text{coupling}} = -C_{ij}^{\text{Maxwell}}\quad(i\neq j)
$$

- Matrix 기반 GND row:
$$
C_{i,GND} = C_{ii} + \sum_{j\neq i} C_{ij}
$$

구현 위치:
- `bem_solver.py`의 `extract_coupling_caps()`

## 4.6 Adaptive mesh 식

사각형-사각형 최소거리 `d`를 기준으로 local panel size:
$$
s_{\text{local}} = s_{\max}\left(p + (1-p)\min\left(\frac{d}{d_0},1\right)\right)
$$

여기서 `p=proximity_factor`, `d0=proximity_distance`.

구현 위치:
- `mesh.py`의 `_compute_adaptive_rect_panel_sizes()`

---

## 5. FasterCap 정합 전략 (캘리브레이션 제외)

이번 best 성능을 얻은 핵심은 아래 4가지입니다.

1. `--remove-internal-faces`
- 도체 내부 공유면 제거
- FasterCap export 측면의 geometry 처리와 더 유사

2. `--uniform-epsilon 4.5` (또는 `--match-fastercap` 자동값)
- 현재 FasterCap export는 `.lst`에서 대표 유전율 단일값을 사용
- BEM도 동일 가정으로 맞추면 조건 정합이 좋아짐

3. `--no-ground-plane` + `--ground-model matrix`
- 명시적 image ground와의 불일치 항을 줄이고,
- row-sum 기반 GND 산출을 FasterCap 비교 형태에 맞춤

4. 고해상도 mesh + near-field 샘플 증가
- panel size 0.38, near-field 7x7에서 오차 감소

중요:
- 위는 **데이터 기반 보정(캘리브레이션)** 이 아니라,
  **물리/수치 모델의 조건 정합 및 해상도 강화**입니다.

---

## 6. `--match-fastercap` 플래그 동작

`cap_extract.py`의 `--match-fastercap`는 다음을 자동 적용합니다.

- `no_ground_plane=True`
- `ground_model="matrix"`
- `remove_internal_faces=True`
- `uniform_epsilon=stack.get_effective_epsilon(1.5)`

즉, FasterCap export가 사용하는 대표 유전율과 내부면 제거 가정에 맞춰
빠르게 정합 모드로 실행할 수 있습니다.

---

## 7. 문서 구조

아래 파일들이 각 Python 모듈 상세 문서입니다.

- `cap_extract.md`
- `bem_solver.md`
- `mesh.md`
- `process_stack.md`
- `polygon_parser.md`
- `gds_to_polygons.md`
- `fastercap_export.md`
- `compare_fc_vs_bem.md`
- `generate_stack_from_pdk.md`
- `merge_smallcap_refine.md`

각 문서에는 함수/클래스 역할, 입출력 타입, 내부 알고리즘, 예외/주의점이 포함됩니다.
