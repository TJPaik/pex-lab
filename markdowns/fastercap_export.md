# fastercap_export.py 상세 설명

이 문서는 `fastercap_export.py`를 설명합니다.

---

## 1. 파일 역할

`fastercap_export.py`는 Python geometry를 FasterCap 입력으로 내보내고,
원하면 FasterCap을 직접 실행한 뒤 결과 행렬을 CSV로 변환합니다.

핵심 산출물:
- net별 `.qui` 파일
- 전체 참조 `.lst` 파일
- (옵션) FasterCap 결과 파싱 CSV

---

## 2. 데이터 흐름

입력:
- `polygons.txt` 또는 `GDS(--from-gds)`
- `ProcessStack`

중간:
- net별 외부 표면 panel 문자열(`Q ...`)

출력:
- `input.lst`
- `*.qui`
- (옵션) `fastercap_output.csv`

---

## 3. 핵심 함수

## 3.1 `rect_to_quads(rect, layer, scale, net_name, max_size)`

역할:
- 하나의 rect + layer 높이 정보를 FasterCap `Q` 라인으로 변환

출력:
- `List[str]` (`Q name x1 y1 z1 ... x4 y4 z4`)

현재 구현 특징:
- face subdivision을 실제로 하지는 않고 full quad를 출력
- 실질적 내부면 제거/표면 추출은 `_collect_panels_for_net`가 담당

---

## 3.2 `_panel_key(corners)`

- corner 집합을 canonical key로 변환
- 같은 면(순서가 달라도 동일)을 식별하기 위해 사용

---

## 3.3 `_collect_panels_for_net(net, stack, net_name)`

입력:
- `net: NetGeometry`

출력:
- `(lines, panel_count, internal_count)`

동작:
1. 각 rect를 3D box 표면으로 확장(top/bottom/side)
2. face key 카운트
3. 등장 횟수 1인 면만 채택(내부 공유면 제거)
4. FasterCap `Q` 라인 문자열 생성

이 함수가 `remove_internal_faces`에 해당하는 FasterCap 측 핵심 로직입니다.

---

## 3.4 `export_fastercap(nets, stack, output_dir, max_size=1.0)`

출력:
- `lst_path: str`

절차:
1. net별 `_collect_panels_for_net` 실행
2. `net_name.qui` 파일 생성
3. 대표 유전율 계산: `stack.get_effective_epsilon(1.5)`
4. `input.lst` 작성 (`C <file> <eps> 0 0 0`)

중요:
- 현재 `.lst`는 단일 대표 `eps_r`를 사용합니다.
- 이 가정이 BEM 정합에서 `uniform_epsilon` 옵션으로 연결됩니다.

---

## 3.5 `run_fastercap(lst_path, fastercap_bin, accuracy, timeout_s, use_galerkin)`

역할:
- subprocess로 FasterCap 실행

옵션 매핑:
- `-b`: batch mode
- `-a<accuracy>`: 정확도
- `-g`: Galerkin 모드(선택)

출력:
- `stdout + stderr` 텍스트

---

## 3.6 `parse_fastercap_matrix(output_text)`

출력:
- `(names: List[str], matrix_fF: np.ndarray)`

절차:
1. 로그에서 "Capacitance matrix is:" 블록 탐색
2. `Dimension NxN` 파싱
3. 행렬 숫자 파싱
4. 단위 변환
   - FasterCap 수치(um geometry 기준) -> fF
   - 코드에서 `*1e9` 적용

추가 처리:
- conductor 이름의 `g1_` 같은 그룹 prefix 제거

---

## 3.7 `matrix_to_coupling_rows(names, matrix_fF, min_cap_fF)`

출력:
- `List[(net1, net2, cap)]`

규칙:
- `i<j` off-diagonal에서 coupling은 `-Cij`
- ground row는 row-sum:
$$
C_{i,GND} = C_{ii} + \sum_{j\neq i}C_{ij}
$$

---

## 3.8 `write_caps_csv(rows, csv_path)`

- 표준 헤더로 CSV 기록
- `cap_extract.py` 출력 포맷과 동일하게 맞춤

---

## 4. CLI 모드

입력 옵션:
- `input`, `--from-gds`, `--stack`, `--panel-size`

실행 옵션:
- `--run`, `--fastercap-bin`, `--accuracy`, `--timeout`, `--galerkin`

결과 옵션:
- `--csv-out`, `--min-cap`

---

## 5. 정합 관점에서의 중요 포인트

1. 내부면 제거
- `_collect_panels_for_net`에서 공유면을 제거
- BEM 측도 `--remove-internal-faces`를 써야 조건이 맞음

2. 단일 유전율 사용
- `.lst`에서 대표 epsilon 하나만 사용
- BEM 정합 시 `--uniform-epsilon`이 필요

3. ground 행 산출
- row-sum 기반 GND이므로 BEM 비교 시 `ground-model matrix`가 대응됨

