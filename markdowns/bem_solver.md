# bem_solver.py 상세 설명

이 문서는 `bem_solver.py`의 수학 모델, 자료형, 함수 흐름을 상세하게 설명합니다.

---

## 1. 파일 역할

`bem_solver.py`는 panelized geometry를 입력받아
- potential coefficient matrix `P`를 구성하고,
- 선형계 `P * sigma = V`를 여러 excitation에 대해 풀고,
- net-level Maxwell capacitance matrix를 만든 뒤,
- 최종 coupling row (`net1, net2, fF`)를 추출합니다.

즉 이 레포의 수치해석 코어입니다.

---

## 2. 상수와 단위

```python
EPSILON_0 = 8.854187817e-12  # F/m
UM_TO_M = 1e-6
FF_FACTOR = 1e15
```

입력 geometry는 `um` 단위이므로, 물리식 계산 전에 `m`로 변환합니다.
최종 출력은 `F`를 `fF`로 변환합니다.

---

## 3. 클래스 구조

## 3.1 `class BEMSolver`

생성자:
```python
def __init__(
    self,
    panels: List[Panel],
    net_indices: Dict[str, int],
    stack: ProcessStack,
    use_ground_plane: bool = True,
    near_field_factor: float = 0.0,
    near_field_samples: int = 3,
    uniform_epsilon_r: float = None,
)
```

핵심 멤버:
- `self.panels: List[Panel]`
- `self.net_indices: Dict[str, int]`
- `self.num_nets: int`
- `self.N: int` (패널 수)
- `self.centers: np.ndarray` shape `(N,3)`
- `self.areas: np.ndarray` shape `(N,)`
- `self.net_ids: np.ndarray` shape `(N,)`
- `self.eps_at_panel: np.ndarray` shape `(N,)`

`eps_at_panel` 구성 방식:
- `uniform_epsilon_r` 지정 시: 모든 패널 동일 값
- 아니면 `stack.get_effective_epsilon(p.cz)`로 층별 값

---

## 4. 함수별 상세

## 4.1 `_panel_sample_points(panel)`

입력:
- `panel: Panel`

출력:
- `np.ndarray` shape `(n*n, 3)` 또는 `(1,3)`

역할:
- near-field 정밀화를 위해 소스 패널을 `n x n` subcell 중심점으로 샘플링
- `panel.u_axis`, `panel.v_axis`, `panel.du`, `panel.dv`를 사용해 좌표 오프셋 생성

---

## 4.2 `_self_term_rect(du_um, dv_um, eps_r)`

직사각형 패널 자기항 계산:
$$
a = \frac{du}{2},\; b = \frac{dv}{2},\; r=\sqrt{a^2+b^2}
$$
$$
I = 4\left[a\ln\frac{b+r}{a} + b\ln\frac{a+r}{b}\right]
$$
$$
P_{ii}=\frac{I}{4\pi\epsilon_0\epsilon_r}
$$

출력 단위:
- potential coefficient (SI 기반)

---

## 4.3 `_apply_near_field_correction(P, dist, eps_avg, areas_m2)`

목적:
- 가까운 패널 상호작용을 단순 `1/r` 중심점 근사 대신, subcell 합산으로 교체

근접 판정:
- 각 source panel `j`에 대해
- `dist[i,j] < near_field_factor * max(char_len[i], char_len[j])`

교체식:
$$
P_{ij} \approx \sum_k \frac{\Delta A_k}{4\pi\epsilon_0\epsilon_{ij}|r_i-r_{j,k}|}
$$

`use_ground_plane=True`일 때:
- image source 점(`z -> -z`)을 동일하게 계산해 빼줌

부작용/비용:
- 계산량이 커지며 실행시간 증가

---

## 4.4 `build_coefficient_matrix()`

출력:
- `P: np.ndarray`, shape `(N,N)`

절차:
1. center 간 pairwise 거리 `dist` 계산
2. `eps_avg = 0.5*(eps_i + eps_j)` 계산
3. 오프대각:
$$
P_{ij}=\frac{A_j}{4\pi\epsilon_0\epsilon_{avg}r_{ij}}
$$
4. `use_ground_plane`이면 image 항을 빼기
5. 대각(Self) 항은 `_self_term_rect` 적용
6. near-field correction 적용
7. `nan/inf` 정리

주의:
- `use_ground_plane`은 무한 평면 이미지 근사이므로,
  명시적 GND 도체를 동시에 사용하면 중복 경계조건이 될 수 있습니다.

---

## 4.5 `solve_capacitance_matrix()`

출력:
- `cap_matrix: np.ndarray`, shape `(num_nets, num_nets)`, 단위 `F`

절차:
1. `P` 구성
2. `lu_factor(P)`로 LU 분해
3. 각 excitation net `k`에 대해:
   - `V` 벡터 구성 (`net==k`인 패널만 1V)
   - `sigma = lu_solve((lu,piv), V)`
   - 각 net `m`의 총전하 `Q_m = sum(sigma_i * A_i)`
   - `cap_matrix[m,k] = Q_m`

수학적으로:
$$
P\sigma^{(k)} = V^{(k)},\quad C_{mk}=Q_m^{(k)}
$$

---

## 4.6 `_multilayer_cap_per_area(z_conductor)`

해석적 ground 모델용 함수.

다층 유전체 series 식:
$$
\frac{1}{C''} = \sum_{\ell}\frac{d_\ell}{\epsilon_0\epsilon_{r,\ell}},\quad C''=\frac{1}{\sum d_\ell/(\epsilon_0\epsilon_{r,\ell})}
$$

출력:
- `F/m^2`

---

## 4.7 `compute_ground_caps(nets_data)`

입력:
- `nets_data`: `Dict[str, NetGeometry]`

출력:
- `Dict[str, float]` (net -> ground cap in fF)

모델 구성:
1. `nwell` depletion cap (면적 비례 상수)
2. diffusion/substrate 계층 junction cap (area + perimeter)
3. 일반 금속층:
   - parallel-plate (`C'' * A`)
   - fringe (`perimeter` 기반)

fringe 근사식:
$$
C_{fringe} = \epsilon_0\epsilon_{eff}\cdot
\frac{t}{\pi h}\ln\left(1+\frac{2h}{t}\right)\cdot L
$$

주의:
- 이 함수는 해석적 모델이라, full BEM ground 해와 다를 수 있습니다.

---

## 4.8 `extract_coupling_caps(...)`

시그니처:
```python
def extract_coupling_caps(
    self,
    ground_net: str = "GND",
    nets_data=None,
    min_cap_fF: float = 1e-6,
    signal_scale: float = 1.0,
    ground_scale: float = 1.0,
) -> List[Tuple[str, str, float]]
```

입출력:
- 입력: scaling, ground mode 선택 정보
- 출력: `(net1, net2, cap_fF)` 리스트

작동:
1. `cap_matrix(F)` -> `cap_fF`
2. 대칭화: `0.5*(C + C^T)`
3. `i<j` off-diagonal에서 coupling 추출:
$$
C_{ij}^{coupling} = -C_{ij}
$$
4. ground row 추출:
   - `nets_data` 있으면 해석적 ground 사용
   - 아니면 matrix row-sum 사용
$$
C_{i,GND}=C_{ii}+\sum_{j\neq i}C_{ij}
$$
5. explicit ground net이 이미 존재하면 추가 ground row를 만들지 않음

숫자 안정성 처리:
- row-sum이 음수면 `abs` 처리(미세 수치오차 보정)

---

## 5. 성능/정확도 trade-off 포인트

1. 패널 수 `N` 증가
- 정확도 상승 가능
- 메모리/시간 급증 (`O(N^2)` 행렬 저장, 선형해석 비용 큼)

2. near-field 샘플 수 (`near_field_samples`)
- 근접 상호작용 정확도 개선
- 계산시간 증가

3. `uniform_epsilon_r`
- FasterCap export가 단일 유전율 가정일 때 정합에 유리
- 다층 물리 엄밀성은 낮아질 수 있음

---

## 6. 요약

`bem_solver.py`는 다음 3가지를 담당합니다.
- 물리 커널 `P` 구성
- 선형계 풀이로 net-level `C` 도출
- 출력 포맷(coupling/GND row) 변환

정합 목적에서는 `uniform_epsilon`, `no_ground_plane`, `matrix ground` 조합이
의미 있게 동작하며, 이는 `cap_extract.py --match-fastercap`에서 자동으로 묶어 제공합니다.

