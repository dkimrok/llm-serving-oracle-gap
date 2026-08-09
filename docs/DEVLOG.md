# DEVLOG — gap_bench

## D1 (2026-08-04)

### 완료
- 오라클 MILP(formulation.py), 재생 시뮬레이터(simulator.py), 토이 검증(toy_test.py) 구현
- **토이 검증 통과:** 5요청 인스턴스에서 브루트포스 전수탐색(W=8) 최적 == MILP 최적 == 18, 오라클 해 재생 위반 0
- 토이에서 논문 테제의 축소판 확인: FCFS 갭 +16.7% (head-of-line blocking), 완전정보 SJF 갭 0%
- 규모 프로브: n=60 저부하 인스턴스 13.7s 최적 증명 (binvars 7,260) / 갭 0% → **갭은 경합 체제에서만 발생** (RQ3 예고편)
- 경합 프로브(tight): binvars 12,060, 90s 제한 → 발견해 1283, 하한 1177 (MIP 갭 8.3%), FCFS +23.6%(vs 발견해)

### 발견·수정한 버그
1. **simulator: 동일 버킷 기승인 요청의 프리필 토큰 누락** — 승인 검사에서 예산 위반 발생. 수정 후 토이 재검증 통과.
2. **formulation: 제약 구축 루프 O(T×n×|z|)** — 경합 인스턴스에서 모델 구축 자체가 수백초. 버킷별 누산기 O(|z|·d̄)로 재작성.
3. **PuLP HiGHS 래퍼가 시간제한 발견해를 "Optimal"로 오보** — SJF가 오라클을 이기는 가짜 모순(−2.6%) 유발. highspy 직접 호출로 전환, (발견해, 쌍대하한, MIP갭) 삼중 보고. → 설계문서 §3 "[하한, 발견해] 구간 보고" 방침이 필수임을 실증.

### 설계 함의
- 갭 보고는 반드시 구간으로: 예) tight 인스턴스 FCFS 참갭 ∈ [(1586−1283)/1283, (1586−1177)/1177] = [23.6%, 34.8%]
- 휴리스틱 해를 MILP warm start로 주입하면 (a) 풀이 가속, (b) 발견해 ≤ 최선 휴리스틱 보장 → 휴리스틱 갭 음수 원천 차단. **v0.2 최우선 패치.**
- 파일럿 예산 신호: n=60 경합 인스턴스는 90s로 부족. 슬라이스당 5–20분 예산(§9)은 현실적, rolling-horizon은 n=300+에서 필요할 전망.

### 다음 (D2)
- [ ] warm start 구현 (highspy setSolution / MIP start)
- [ ] Azure LLM 트레이스(AzurePublicDataset) 필드·라이선스 확인, 로더 작성 — 실패 시 BurstGPT
- [ ] O-pred 층 (예측 길이 계획 + 수리 규칙) 구현
- [ ] delta ∈ {4,8,16} 민감도 훅

## D2 (2026-08-04, 같은 날 연속 작업)

### 완료
- **트레이스 확보·검증:** AzureLLMInferenceDataset2024 (code 16.8M행 + conv 27.3M행, 각 168h).
  스키마 = {TIMESTAMP, ContextTokens, GeneratedTokens} — 필요 필드 정확히 일치.
  라이선스 데이터 CC-BY-4.0 / 코드 MIT, **인용 의무: DynamoLLM (HPCA 2025)**.
- **특성 산출:** 두 서비스가 자연스러운 두 체제 형성 —
  code: 프리필 지배 (L p50=8, CV 3.23), 시간별 부하 7.5→83.6 req/s (11배 변동)
  conv: 디코드 비중 큼 (L mean=106, p99=692, CV 1.50), 26→75 req/s
  → 주 트레이스 = conv, code는 일반화 검증. 자연 부하 층화 가능 (합성 λ 스케일링과 병행).
- **파이프라인:** prep_trace.py (CSV→정렬 parquet), loader.py (부하 분위 기반 슬라이스 추출 → Instance).
- **실 트레이스 스모크 (400요청 슬라이스, v0.1 모델, iter_ms=25 미보정):**
  conv@busy: FCFS 75.9 vs SJF 34.4 buckets → SJF 실행가능성 논증으로 **FCFS 참갭 ≥ 120% 하한**
  conv@low: 113.2 vs 37.2 / code@busy: 30.6 vs 26.9 (14% — 프리필 지배 체제는 여유 작음, 예측 부합)

### 발견한 모델 한계 (D3 필수 승격)
- **단일버킷 프리필 가정 붕괴:** conv/code의 p99 프롬프트(6.7k–7.7k) > 현실적 버킷 예산
  → 해당 요청 영구 승인 불가. **청크드 프리필**(p_i를 ⌈p_i/B_pf⌉버킷 분할)이 v0.2 필수.
  임시로 bucket_budget=9000 상향으로 스모크 통과 (논문 수치로 사용 금지).

### D3 큐
- [ ] 청크드 프리필 정식화 + 시뮬레이터 반영 (필수)
- [ ] warm start (휴리스틱 해 주입) — 실슬라이스 MILP 필수 전제
- [ ] iter_ms 보정 (vLLM 공개 벤치마크 수치 조사) 또는 민감도 축으로 명시
- [ ] O-pred 층 구현
- [ ] 주의: 컨테이너 리셋 시 traces/ 재생성 필요 (prep_trace.py 실행, ~5분)

## D3 (2026-08-04, 연속)

### 완료 — 청크드 프리필 정식화 (v0.2)
- **설계 선택:** 고정 청크 (요청당 버킷당 B_pf 토큰 결정론 진행). 유연 청크는 선행제약
  이진변수+big-M로 기각. 모델 구조·크기 v0.1 그대로 유지 (계수만 프리필/디코드 2상으로).
- **부수 이득:** TTFT = 승인 + π_i − 도착이 자연 정의 → F2 목적함수 공짜.
- **보수성 2호:** 고정청크 오라클 ≤ 유연청크 최적 → 갭 하한 방향 유지 (선점 금지와 동일 논리).
- 계수 함수(cache_coeff/tok_coeff)를 formulation 단일 소스로 통합 — 시뮬레이터가 임포트,
  오라클·휴리스틱 실행가능영역 일치 보장.
- **warm start 구현·검증:** highspy setSolution, 토이에서 FCFS(42) 시드 → 최적(27) 도달.
- **v0.2 토이 검증 통과:** 다중버킷 프리필 케이스 브루트포스 == MILP == 27, 위반 0,
  FCFS +55.6% / 완전정보 SJF 0%.
- **실슬라이스 첫 MILP (conv@busy n=200, W=150, 예산 4000/청크 2048 — D2에서 불가능했던 조건):**
  파이프라인 종단 작동. 60s 풀이: 발견해 8627 (=SJF warm start, 개선 못함), 하한 6540.
  갭 구간: FCFS [+9.6%, +44.5%], SJF [0%, +31.9%]. 하한이 아직 느슨 → D4 과제.

### 버그·이슈
- loader.py가 v0.1 잔재 (prefill_chunk 미지원) → SliceSpec/Instance 전달 패치.
- 백그라운드 240s 풀이 중 프로세스 사망 (OOM 추정: 트레이스 캐시 ~700MB + B&B 트리).
  → D4: 풀이 전 loader._cache.clear(), 슬라이스 데이터만 유지, 필요 시 스왑/청크 로딩.

### D4 큐
- [ ] 하한 강화: 장시간 풀이(Colab), rolling-horizon, W 민감도 확인
- [ ] 메모리 대책 반영한 슬라이스 러너 (배치 실행 + Drive 체크포인트, Script 37 방식 이식)
- [ ] O-pred 층 구현 (구간 중앙값 예측기부터)
- [ ] iter_ms 보정 조사

## D4 (2026-08-05)

### 완료 — 배치 러너 + Colab 이식 준비
- **run_slices.py:** 그리드 생성(파일럿 6 / 본 50 슬라이스), 슬라이스별
  [휴리스틱 → 최선해 warm start → MILP] 파이프라인, results.csv에 즉시 append
  (매 슬라이스 flush → 세션 중단 안전), 완료 slice_id 자동 skip으로 재개 가능.
- **메모리 대책:** 슬라이스 추출 직후 loader 캐시 해제 + gc — 트레이스 0.7GB를
  분기한정 트리에 양보. (D3의 백그라운드 OOM 재발 방지)
- **로컬 스모크:** 2슬라이스 실행 → 재실행 시 전부 skip 확인. CSV 원장 정상.
- **COLAB_SETUP.md:** 파일 목록 8개, 셀 단위 절차, 파일럿→본실험 전환 기준.

### 발견·수정한 버그
- **warm start 침묵 거부:** SJF 계획의 일부 승인 시각이 admit_window(150) 밖이면
  주입을 통째로 포기 → 슬라이스 1에서 발견해(19289)가 warm 시드(9011)보다 나쁜
  이상 결과. 수정: (a) 휴리스틱 실행 후 warm 계획의 최대 대기를 수용하도록
  admit_window 동적 확장, (b) warm_applied를 결과 CSV에 기록해 침묵 실패 차단.
  수정 후 발견해 == SJF(9011), FCFS 갭 [+30.6%, +66.0%] 정상화.

### 파일럿 초기 관측 (TL=60, 참고용 — 인용 금지)
- conv rank0.3 두 슬라이스: FCFS 갭 하한 +19~31%, milp_gap 21~26% (60s)
  → TL=900에서의 하한 조임 정도가 파일럿의 1차 확인 사항.

### D5 큐 (Colab에서)
- [ ] 파일럿 6슬라이스 (TL=900) → warm_applied/milp_gap/소요시간 점검
- [ ] O-pred 층 (구간 중앙값 예측기) 러너 통합
- [ ] milp_gap이 안 조여지면: rolling-horizon 구현 판단
- [ ] iter_ms 보정 문헌 조사

## D4-b (2026-08-05) — 로컬 실행 전환
- run_slices.py 파라미터화 리팩터: main(pilot, time_limit, n_req, results_csv) +
  argparse CLI (--pilot/--main --tl). env 변수 의존 제거 (Windows 호환).
  리팩터 후 동일 슬라이스 재현 검증 (+19.3% 하한 일치).
- gap_bench_local.ipynb: Drive 마운트 제거, subprocess 기반 검증/준비 셀,
  파라미터 셀([4])에서 PILOT/TIME_LIMIT 직접 제어.
- LOCAL_SETUP.md: venv(Win/mac/Linux), 터미널 전용 경로, 장시간 실행 주의
  (절전 해제, caffeinate), Colab 대비 차이점.
- 실행 환경 결정: 로컬 우선 (세션 제한 없음 → TL 1800 상향 옵션 확보).

## D4-c (2026-08-05) — 폴더 구조 재편 (Duncan 지정 구조)
- gap_bench/ 루트: 노트북 2개(local/colab) + README + docs/ + results/ + traces/
  코드는 local_script/, colab_script/에 각각 (현재 동일 사본 — 수정 시 양쪽 동기화 규칙).
- **경로 루트 앵커링:** loader/prep_trace/run_slices가 스크립트 폴더의 부모를 ROOT로
  잡아 traces/, results/를 해석 — 실행 위치 무관 동작. 새 구조에서 toy_test·로더·
  결과경로 검증 완료.
- 노트북 재작성: [1]셀에서 sys.path에 스크립트 폴더 주입, run_slices.main() 직접 호출.
- 주의 규칙 명문화(README): Drive 폴더 공유 시 로컬·Colab 동시 실행 금지 (원장 경합).
