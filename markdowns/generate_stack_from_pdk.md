# generate_stack_from_pdk.py 상세 설명

이 문서는 공식 skywater-pdk checkout에서 stack JSON을 자동 생성하는
`generate_stack_from_pdk.py`를 설명합니다.

---

## 1. 파일 역할

입력(공식 PDK 문서 파일):
- `docs/_static/metal_stack.ps`
- `docs/rules/rcx/capacitance-parallel.tsv`
- `docs/rules/rcx/capacitance-fringe-upward.tsv`

출력:
- `sky130a_stack_from_pdk.json` 형태의 JSON

이 JSON은 `ProcessStack.from_json()`과 호환됩니다.

---

## 2. 유틸 함수

## 2.1 `_read_text(path)`

- 파일 존재 검증 후 텍스트 로드
- 누락 시 `FileNotFoundError`

## 2.2 `_normalize_layer_name(name)`

- TSV/문서 layer 문자열을 내부 키(`met1`, `li`, `poly`)로 정규화

## 2.3 `_parse_k_values_from_metal_stack(ps_text)`

- `metal_stack.ps`에서 `(NAME K=value)` 패턴 추출
- 반환: `Dict[str, float]`

## 2.4 `_require_ps_values(ps_text, values)`

- 필수 숫자 marker 존재 여부 검증
- 누락 시 `ValueError`

목적:
- PDK 문서 형식 변경/파싱 실패를 조기 감지

## 2.5 `_parse_tsv_matrix(path)`

입력:
- tab-separated matrix 파일

출력:
- `Dict[(row_layer, col_layer), float]`

동작:
- 헤더/행 이름 정규화
- 숫자 값만 파싱

## 2.6 `_git_commit_short(repo)`

- PDK repo의 short commit hash를 가져옴
- 실패 시 `unknown`

---

## 3. 핵심 함수 `build_stack_json(pdk_root)`

출력:
- `dict` (JSON 직렬화 가능한 stack 구조)

주요 단계:
1. source 파일 경로 구성
2. `metal_stack.ps`에서 유전율 `K` 맵 추출
3. 필수 물리 수치 marker 검증
4. 금속/비아 높이 및 두께 하드코딩 값 설정
   - 값 출처: `metal_stack.ps` 라벨
5. TSV에서 병렬/프린지 RCX 값 읽기
6. 최종 stack dict 구성
   - `layers`
   - `dielectrics`
   - 부가 메타(`_source`, `_source_commit`, `_generated_by`)
   - RCX 참조값(`parallel_plate_caps_aF_per_um2`, `fringe_caps_aF_per_um`)

---

## 4. 생성 JSON의 의미

핵심 필드:
- `scale_to_um: 0.01` (centnm -> um)
- `layers`: 도체/비아 3D 형상 정보
- `dielectrics`: z 구간별 epsilon

추가 필드:
- `_source`, `_source_commit`: 재현성(traceability)
- `parallel_plate_caps_*`, `fringe_caps_*`: RCX 레퍼런스 기록

---

## 5. CLI

옵션:
- `--pdk-root` (기본 `/home/paiktj/skywater-pdk`)
- `-o/--output` (기본 `sky130a_stack_from_pdk.json`)

동작:
1. `build_stack_json`
2. JSON pretty print 저장

---

## 6. 왜 중요한가

`ProcessStack` 품질은 BEM 정확도의 기본 조건입니다.
레이어 높이/두께/유전율이 틀리면,
메쉬가 정확해도 전체 커패시턴스는 체계적으로 어긋납니다.

이 스크립트는 PDK 문서에서 stack를 자동 추출해
수동 입력 실수를 줄여줍니다.

