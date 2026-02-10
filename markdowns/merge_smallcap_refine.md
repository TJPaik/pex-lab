# merge_smallcap_refine.py 상세 설명

이 문서는 작은 커패시턴스 구간만 선택적으로 교체하는 도구
`merge_smallcap_refine.py`를 설명합니다.

---

## 1. 파일 역할

두 개의 BEM 결과 CSV를 받아,
- 큰 값은 base를 유지하고,
- 작은 값만 refined 결과로 치환
하는 post-process 유틸입니다.

의도:
- 전역 안정성(base) + 저용량 정밀(refined)을 절충

---

## 2. 핵심 함수

## 2.1 `canonical_pair(n1, n2)`

- pair 순서를 canonicalize

## 2.2 `read_caps(path)`

출력:
- `Dict[(net_a, net_b), cap]`

## 2.3 `write_caps(path, caps)`

- 표준 헤더로 CSV 기록

## 2.4 `main()`

입력 옵션:
- `--base`
- `--refined`
- `--out`
- `--threshold` (기본 0.5 fF)
- `--include-ground`
- `--ground-net` (기본 `GND`)

치환 규칙:
1. base pair 순회
2. `base_cap <= threshold`일 때만 후보
3. `include-ground=False`면 GND pair는 제외
4. refined에 해당 pair가 있으면 값 치환

출력 로그:
- base/refined pair 개수
- 치환 개수
- refined 누락 개수

---

## 3. 주의

- 이 도구는 모델 파라미터를 바꾸는 것이 아니라 결과를 합치는 후처리입니다.
- strict한 “단일 물리모델 결과”가 필요하면 사용하지 않는 것이 맞습니다.

