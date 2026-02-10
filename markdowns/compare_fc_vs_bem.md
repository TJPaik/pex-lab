# compare_fc_vs_bem.py 상세 설명

이 문서는 FasterCap 결과와 Python BEM 결과를 비교하는 도구 `compare_fc_vs_bem.py`를 설명합니다.

---

## 1. 파일 역할

입력 CSV 2개:
- FasterCap reference CSV
- BEM prediction CSV

출력 3종:
1. pairwise 비교 CSV
2. summary 텍스트 리포트
3. scatter PNG (전체 + low-cap zoom)

---

## 2. 입력 CSV 계약

헤더:
```csv
net1,net2,coupling_cap_fF
```

pair는 순서와 무관해야 하므로 canonicalization합니다.

---

## 3. 함수별 상세

## 3.1 `canonical_pair(n1, n2)`

출력:
- `(min(n1,n2), max(n1,n2))`

목적:
- `A-B`와 `B-A`를 같은 pair로 취급

---

## 3.2 `read_caps(path)`

출력:
- `Dict[(net_a, net_b), cap_fF]`

동작:
- CSV 읽기
- pair canonicalize
- float로 저장

---

## 3.3 `compute_metrics(fc_vals, bem_vals)`

입력:
- 동일 길이 float 리스트 2개

출력 dict:
- `count`, `mae`, `rmse`, `mape`, `medape`, `corr`

정의:
- MAE: `mean(|b-f|)`
- RMSE: `sqrt(mean((b-f)^2))`
- MAPE: `mean(|b-f|/f * 100)` (`f`가 매우 작은 경우 보호)
- corr: Pearson 상관계수

---

## 3.4 `try_make_scatter(...)`

입력:
- `fc_vals`, `bem_vals`
- `is_gnd`: 각 pair가 GND pair인지 bool 리스트
- `out_png`
- `lowcap_threshold`

출력:
- 상태 문자열

그림 구성:
1. 왼쪽: 전체 log-log scatter
   - signal pair: 파랑
   - GND pair: 빨강(x 마커)
   - 기준선 `y=x`
2. 오른쪽: low-cap linear zoom (`FC <= threshold`)

의도:
- 전체 추세와 저용량 구간 오차를 동시에 보기 위함

---

## 3.5 `main()`

절차:
1. 인자 파싱
2. FasterCap/BEM CSV 읽기
3. common/fc-only/bem-only pair 집합 계산
4. common pair별 delta/relative error 계산
5. 비교 CSV 저장
6. metrics 계산
   - all pairs
   - signal-only (GND 포함 pair 제외)
7. scatter 생성
8. summary 텍스트 기록

summary에는 상위 오차 pair Top10이 함께 저장됩니다.

---

## 4. 출력 파일 포맷

## 4.1 비교 CSV

헤더:
```csv
net1,net2,fastercap_fF,bem_fF,delta_bem_minus_fc_fF,abs_rel_error_percent
```

정렬:
- `|delta|` 큰 순으로 정렬

## 4.2 summary TXT

포함 정보:
- 공통 pair 수
- 단독 pair 수
- all/signal metric
- scatter 생성 결과
- Top |delta| pair

---

## 5. 해석 팁

1. All RMSE vs Signal RMSE를 분리해서 보세요.
- GND 오차가 전체를 끌어올릴 수 있습니다.

2. low-cap zoom subplot을 반드시 같이 보세요.
- 작은 cap에서는 상대오차가 민감합니다.

3. FasterCap-only/BEM-only pair가 0인지 먼저 확인하세요.
- pair mismatch 상태에서 RMSE 비교는 왜곡됩니다.

