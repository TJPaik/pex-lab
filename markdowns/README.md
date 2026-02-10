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

주요 인자 의미 (cap_extract.py):
- `input`: 입력 파일 경로. 기본은 `polygons.txt`, `--from-gds`면 GDS로 해석.
- `--from-gds`: GDS를 읽어 `polygons.txt`로 변환 후 진행.
- `--polygons-out`: GDS 변환 결과 `polygons.txt` 저장 경로.
- `--stack`: 공정 스택 JSON. 없으면 내장 sky130A 기본값 사용.
- `--panel-size`: 패널 최대 변 길이(um). 값이 작을수록 정확도↑, 시간↑.
- `--min-panel-size`: adaptive mesh 최저 패널 크기(um).
- `--adaptive-mesh`: 근접도 기반으로 패널 크기 적응.
- `--proximity-distance`: adaptive 기준 거리(um). 이보다 가까우면 더 잘게.
- `--proximity-factor`: 근접 영역 축소 비율(0~1). 작을수록 더 촘촘.
- `--edge-refine-factor`: 엣지 패널 크기 축소 비율(0~1).
- `--edge-refine-fraction`: 엣지 주변에서 정밀화할 영역 비율(0~0.49).
- `--remove-internal-faces`: 같은 net 내부 공유면 제거.
- `--near-field-factor`: 가까운 패널을 서브샘플로 재적분(>0이면 활성).
- `--near-field-samples`: 서브샘플 분할 수(n x n).
- `--uniform-epsilon`: 모든 상호작용에 단일 εr 사용(층별 대신 대표값).
- `--no-ground-plane`: 이미지 ground plane 모델 끄기.
- `--ground-model`: `analytic`(해석 GND), `matrix`(행렬 기반 GND), `both`.
- `--ground-net`: GND로 출력할 net 이름.
- `--signal-scale`: net‑net 커플링 스케일(후처리 배율).
- `--ground-scale`: GND 커플링에만 적용되는 배율.
- `--explicit-ground-plane`: bbox 기반 GND 도체면을 물리적으로 추가.
- `--ground-plane-layer`: explicit GND에 쓸 레이어 이름.
- `--ground-plane-z-bottom`: explicit GND 레이어의 z_bottom(um) 강제 지정.
- `--ground-plane-thickness`: explicit GND 두께(um).
- `--ground-plane-margin`: bbox 여유 마진(um).
- `--ground-plane-panel-size`: explicit GND에만 고정 패널 크기(um).
- `--min-cap`: 출력에서 제거할 최소 커플링(fF).
- `--match-fastercap`: FasterCap export 가정과 맞추는 프리셋.

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

주요 인자 의미 (fastercap_export.py):
- `input`: GDS 또는 `polygons.txt` 경로(`--from-gds`로 구분).
- `-o/--output-dir`: FasterCap 입력(.lst/.qui)과 결과 CSV 저장 폴더.
- `--stack`: 공정 스택 JSON 경로.
- `--panel-size`: 출력 패널 크기 힌트(현재는 FasterCap 자체 refine 사용).
- `--run`: export 후 FasterCap을 바로 실행.
- `--fastercap-bin`: FasterCap 실행 파일 경로.
- `--accuracy`: FasterCap `-a` 정확도 옵션(작을수록 정확, 느림).
- `--timeout`: FasterCap 실행 제한 시간(초).
- `--galerkin`: FasterCap `-g`(Galerkin) 모드 사용.
- `--csv-out`: parsed coupling CSV 출력 경로.
- `--min-cap`: CSV에 포함할 최소 커플링(fF).

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

주요 인자 의미 (compare_fc_vs_bem.py):
- `--fastercap`: FasterCap reference CSV.
- `--bem`: Python BEM 결과 CSV.
- `--out-csv`: pairwise 비교 CSV 출력.
- `--summary`: 통계 요약 텍스트 출력.
- `--scatter`: scatter PNG 출력(없으면 `--out-csv` 기반 자동 생성).
- `--lowcap-threshold`: low‑cap 확대 범위 기준값(fF).

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

이 절은 **cap이 수학적으로 무엇인지**부터, **왜 BEM 식이 나오는지**를
순서대로 설명합니다. 기본 가정은 정전기(DC), 선형 유전체, 완전 도체입니다.

### 4.1 Capacitance의 정의 (1개/2개/다도체)

가장 기본적인 정의는 전하와 전위차의 비입니다.

$$
C = \frac{Q}{\Delta V}
$$

다도체(여러 net) 문제에서는 각 도체의 전하가 모든 전위에 의해 결정되므로
**Maxwell capacitance matrix**를 사용합니다.

$$
Q_i = \sum_{j=1}^{N} C_{ij} V_j
$$

여기서

$$
C_{ij} = \frac{\partial Q_i}{\partial V_j}
$$

이고, 선형 유전체일 때 `C`는 대칭 행렬입니다. 또한

$$
U = \frac{1}{2}\sum_{i,j} C_{ij} V_i V_j
  = \frac{1}{2}\mathbf V^{T}\mathbf C \mathbf V
$$

가 성립합니다. 커플링은 통상

$$
C^{\text{coupling}}_{ij} = -C_{ij}\quad(i\neq j)
$$

로 정의하며, GND에 대한 커플링은 행렬의 row‑sum으로 계산합니다.

$$
C_{i,\text{GND}} = C_{ii} + \sum_{j\neq i} C_{ij}
$$

코드 매핑:
- Maxwell 행렬 생성: `bem_solver.py`의 `solve_capacitance_matrix()`
- 커플링/그라운드 출력: `bem_solver.py`의 `extract_coupling_caps()`

### 4.2 정전기 방정식과 경계조건

도체 내부에는 전기장이 없고, 유전체 영역에서는 전하가 없다고 가정하면
전위는 다음을 만족합니다.

$$
\nabla \cdot (\epsilon \nabla \phi) = 0
$$

도체 표면에서는 전위가 일정하므로
$$
\phi|_{S_k} = V_k
$$
가 됩니다. 표면 전하밀도는
$$
\sigma = -\epsilon \frac{\partial \phi}{\partial n}
$$
로 정의됩니다.

이때 임의의 관측점 $\mathbf r$에서의 전위는
표면 전하에 대한 적분으로 표현됩니다.

$$
\phi(\mathbf r) =
\frac{1}{4\pi\epsilon_0}
\int_S \frac{\sigma(\mathbf r')}{\epsilon_r(\mathbf r')\,|\mathbf r-\mathbf r'|}
\,dS'
$$

이 적분식이 BEM의 출발점입니다.  
엄밀한 다층 유전체 해석은 경계조건을 포함해 풀어야 하지만,
이 구현은 패널 높이별 대표 $\epsilon_r$ 또는 단일 $\epsilon_r$로
근사하여 계산 비용을 줄입니다(`--uniform-epsilon`).

코드 매핑:
- 유전율 처리: `bem_solver.py`의 `eps_at_panel`, `stack.get_effective_epsilon()`

### 4.3 패널화와 BEM 행렬식

도체 표면을 작은 패널로 나누고, 각 패널 내 전하밀도 $\sigma_j$를
상수로 근사합니다. 각 패널 중심점에서 전위를 샘플링하면

$$
\phi_i \approx \sum_j P_{ij}\sigma_j
$$

이며,
$$
P_{ij} =
\frac{1}{4\pi\epsilon_0}
\int_{S_j}\frac{1}{\bar\epsilon_{ij}\,|\mathbf r_i-\mathbf r'|}\,dS'
$$
입니다. 패널 간 거리가 충분히 멀면 적분을 면적/거리로 근사하여

$$
P_{ij} \approx
\frac{A_j}{4\pi\epsilon_0\,\bar\epsilon_{ij}\,r_{ij}}
\quad (i\neq j)
$$

가 됩니다. 이 식은 **멀리 있는 패널의 전위를 점전하 근사**로 본 결과입니다.

코드 매핑:
- 오프대각 근사: `bem_solver.py`의 `build_coefficient_matrix()`

### 4.4 자기항(Self‑term)과 근거리 보정

자기항($i=j$)은 적분이 발산하므로 패널 형상에 대한 **정확 적분식**을 씁니다.
직사각형 패널 중심 기준 적분 결과는

$$
I_{\text{rect}} =
4\left(
a\ln\frac{b+r}{a}+b\ln\frac{a+r}{b}
\right),\quad r=\sqrt{a^2+b^2}
$$

이고,
$$
P_{ii} = \frac{I_{\text{rect}}}{4\pi\epsilon_0\,\epsilon_r}
$$
로 구현됩니다.

또한 패널들이 매우 가까우면 면적/거리 근사가 부정확하므로,
소스 패널을 `n x n`으로 쪼개 직접 합산합니다.

$$
P_{ij}^{\text{near}} \approx
\sum_{k=1}^{n^2}
\frac{\Delta A_k}{4\pi\epsilon_0\,\bar\epsilon_{ij}\,|\mathbf r_i-\mathbf r_{j,k}|}
$$

코드 매핑:
- 자기항: `bem_solver.py`의 `_self_term_rect()`
- 근거리 보정: `bem_solver.py`의 `_apply_near_field_correction()`

### 4.5 이미지 전하(ground plane) 모델

z=0 평면을 이상적인 GND로 두면 이미지 전하를 이용해 전위를 보정할 수 있습니다.
각 패널의 이미지 위치 $\mathbf r'_j$에 대해

$$
P_{ij} \leftarrow P_{ij} -
\frac{A_j}{4\pi\epsilon_0\,\bar\epsilon_{ij}\,r'_{ij}}
$$

를 적용합니다. FasterCap 정합 모드에서는 이 항을 끕니다.

코드 매핑:
- `bem_solver.py`의 `build_coefficient_matrix()` (`use_ground_plane`)

### 4.6 선형계 풀이와 Maxwell 행렬 구성

net `k`만 1V로 두고 나머지를 0V로 둔 excitation 벡터 $V^{(k)}$에 대해

$$
P\,\sigma^{(k)} = V^{(k)}
$$

를 풉니다. 그 다음 net `m`의 총 전하는

$$
Q_m^{(k)} = \sum_{i\in m} \sigma_i^{(k)} A_i
$$

이고, Maxwell 행렬 원소는

$$
C_{mk} = Q_m^{(k)}
$$

입니다. 이 과정을 모든 net에 대해 반복합니다.

코드 매핑:
- `bem_solver.py`의 `solve_capacitance_matrix()`

### 4.7 GND 커플링(해석 모델 vs 행렬 모델)

이 코드에서는 두 가지 GND 모델을 제공합니다.

1. **행렬 기반(row‑sum) GND**  
   이미 계산한 Maxwell 행렬로부터 GND 커플링을 계산:
$$
C_{i,\text{GND}} = C_{ii} + \sum_{j\neq i} C_{ij}
$$

2. **해석 기반 GND**  
   다층 유전체 병렬‑판 모델 + 프린지 + 접합 커패시턴스:
$$
\frac{1}{C/A} = \sum_k \frac{d_k}{\epsilon_0 \epsilon_{r,k}}
$$
프린지는 아래 근사식을 사용합니다.
$$
\frac{C_{\text{fringe}}}{L} =
\frac{\epsilon_0\epsilon_r}{\pi}\frac{t}{h}\ln\left(1+\frac{2h}{t}\right)
$$

코드 매핑:
- 해석 GND: `bem_solver.py`의 `compute_ground_caps()`
- 행렬 GND: `bem_solver.py`의 `extract_coupling_caps()`

### 4.8 Adaptive mesh 식

사각형 간 최소거리 `d`에 따라 local panel size를 줄이는 규칙은

$$
s_{\text{local}} =
s_{\max}\left(p + (1-p)\min\left(\frac{d}{d_0},1\right)\right)
$$

이며, 여기서 `p=proximity_factor`, `d0=proximity_distance`입니다.

코드 매핑:
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
