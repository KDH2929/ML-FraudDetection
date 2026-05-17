param(
    [string]$InputPpt = "C:\Users\KDH\Documents\VisualStudio Code\CatchCatch\캐치캐치_최종3_레이아웃통일.pptx",
    [string]$OutputPpt = "C:\Users\KDH\Documents\VisualStudio Code\CatchCatch\캐치캐치_최종3_레이아웃통일_memberA로드맵추가.pptx"
)

$ErrorActionPreference = "Stop"

$ppLayoutBlank = 12
$msoTextOrientationHorizontal = 1
$msoShapeRectangle = 1
$msoShapeRoundedRectangle = 5
$msoFalse = 0
$msoTrue = -1

function Rgb([int]$r, [int]$g, [int]$b) {
    return ($r -bor ($g -shl 8) -bor ($b -shl 16))
}

$C = @{
    Bg      = Rgb 1 19 25
    Blue    = Rgb 45 116 255
    Blue2   = Rgb 92 155 255
    Soft    = Rgb 189 205 220
    White   = Rgb 242 248 252
    Card    = Rgb 5 28 38
    Card2   = Rgb 9 43 58
    Line    = Rgb 56 85 105
    Green   = Rgb 126 239 172
    Yellow  = Rgb 234 177 37
    Muted   = Rgb 145 164 178
    Red     = Rgb 255 118 118
}

$Font = "맑은 고딕"
$global:SX = 1.0
$global:SY = 1.0
$global:SF = 1.0

function Set-TextRangeStyle($tr, [double]$size, [int]$color, [bool]$bold = $false, [bool]$italic = $false) {
    $tr.Font.NameFarEast = $Font
    $tr.Font.Name = $Font
    $tr.Font.Size = $size
    $tr.Font.Fill.ForeColor.RGB = $color
    $tr.Font.Bold = if ($bold) { $msoTrue } else { $msoFalse }
    $tr.Font.Italic = if ($italic) { $msoTrue } else { $msoFalse }
}

function Add-Text($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string]$text,
    [double]$size = 16, [int]$color = $C.White, [bool]$bold = $false, [bool]$italic = $false,
    [int]$align = 1) {
    $x = $x * $global:SX
    $y = $y * $global:SY
    $w = $w * $global:SX
    $h = $h * $global:SY
    $size = $size * $global:SF
    $shape = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, $x, $y, $w, $h)
    $shape.TextFrame2.MarginLeft = 0
    $shape.TextFrame2.MarginRight = 0
    $shape.TextFrame2.MarginTop = 0
    $shape.TextFrame2.MarginBottom = 0
    $shape.TextFrame2.WordWrap = $msoTrue
    $shape.TextFrame2.TextRange.Text = $text
    Set-TextRangeStyle $shape.TextFrame2.TextRange $size $color $bold $italic
    $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = $align
    $shape.TextFrame.TextRange.Font.Name = $Font
    $shape.TextFrame.TextRange.Font.Size = $size
    $shape.TextFrame.TextRange.Font.Color.RGB = $color
    $shape.TextFrame.TextRange.Font.Bold = if ($bold) { $msoTrue } else { $msoFalse }
    $shape.TextFrame.TextRange.Font.Italic = if ($italic) { $msoTrue } else { $msoFalse }
    return $shape
}

function Add-Rect($slide, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fill, [int]$line = -1, [double]$lineWeight = 1.2, [int]$shapeType = $msoShapeRoundedRectangle) {
    $x = $x * $global:SX
    $y = $y * $global:SY
    $w = $w * $global:SX
    $h = $h * $global:SY
    $lineWeight = $lineWeight * $global:SF
    $shape = $slide.Shapes.AddShape($shapeType, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Fill.Transparency = 0
    if ($line -ge 0) {
        $shape.Line.Visible = $msoTrue
        $shape.Line.ForeColor.RGB = $line
        $shape.Line.Weight = $lineWeight
    } else {
        $shape.Line.Visible = $msoFalse
    }
    return $shape
}

function Add-Badge($slide, [string]$text) {
    $pill = Add-Rect $slide 42 20 138 46 $C.Bg $C.Blue2 1.6 $msoShapeRoundedRectangle
    $pill.Fill.Transparency = 1
    Add-Text $slide 60 31 110 22 $text 15 $C.Blue2 $true $false 2 | Out-Null
}

function Add-Header($slide, [string]$badge, [string]$title, [string]$subtitle) {
    $bg = Add-Rect $slide 0 0 960 540 $C.Bg -1 0 $msoShapeRectangle
    $bg.ZOrder(1) | Out-Null
    Add-Rect $slide 0 0 960 8 $C.Blue -1 0 $msoShapeRectangle | Out-Null
    Add-Badge $slide $badge
    Add-Text $slide 44 84 700 45 $title 31 $C.White $true | Out-Null
    if ($subtitle) {
        Add-Text $slide 46 134 760 31 $subtitle 13.5 $C.Soft $false $true | Out-Null
    }
    Add-Rect $slide 44 174 132 3 $C.Blue -1 0 $msoShapeRectangle | Out-Null
}

function Add-CardTitle($slide, [double]$x, [double]$y, [double]$w, [string]$title, [int]$color = $C.White) {
    Add-Text $slide $x $y $w 24 $title 15.5 $color $true | Out-Null
}

function Add-MetricCard($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string]$value, [string]$label, [string]$note, [int]$valueColor = $C.Blue2) {
    Add-Rect $slide $x $y $w $h $C.Card $C.Line 1.2 | Out-Null
    Add-Text $slide ($x + 15) ($y + 13) ($w - 30) 26 $value 24 $valueColor $true | Out-Null
    Add-Text $slide ($x + 15) ($y + 44) ($w - 30) 18 $label 10.5 $C.White $true | Out-Null
    if ($note) { Add-Text $slide ($x + 15) ($y + 64) ($w - 30) 30 $note 9 $C.Muted | Out-Null }
}

function Add-Table($slide, [double]$x, [double]$y, [double]$w, [double]$h, [object[]]$rows, [int[]]$colWidths, [int]$highlightRow = -1) {
    $rCount = $rows.Count
    $cCount = $rows[0].Count
    $rowH = $h / $rCount
    if (-not $colWidths -or $colWidths.Count -lt $cCount) {
        $colWidths = @()
        for ($i = 0; $i -lt $cCount; $i++) { $colWidths += [int]($w / $cCount) }
    }
    $sumW = 0
    foreach ($cw in $colWidths) { $sumW += $cw }
    $scale = $w / [double]$sumW
    for ($r = 1; $r -le $rCount; $r++) {
        $cx = $x
        for ($c = 1; $c -le $cCount; $c++) {
            $cw = $colWidths[$c - 1] * $scale
            $fill = if ($r -eq 1) { Rgb 92 155 255 } elseif ($r -eq $highlightRow) { Rgb 184 219 255 } else { Rgb 235 242 247 }
            Add-Rect $slide $cx ($y + ($r - 1) * $rowH) $cw $rowH $fill (Rgb 31 60 76) 0.6 $msoShapeRectangle | Out-Null
            $cellText = Add-Text $slide ($cx + 4) ($y + ($r - 1) * $rowH + 3) ($cw - 8) ($rowH - 5) ([string]$rows[$r - 1][$c - 1]) 8.5 (Rgb 3 18 26) ($r -eq 1) $false 2
            $size = if ($r -eq 1) { 8.7 } else { 8.5 }
            Set-TextRangeStyle $cellText.TextFrame2.TextRange ($size * $global:SF) (Rgb 3 18 26) ($r -eq 1) $false
            $cellText.TextFrame.TextRange.Font.Color.RGB = Rgb 3 18 26
            $cx += $cw
        }
    }
}

function Add-Bullets($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string[]]$items, [double]$size = 10.5) {
    $text = ($items | ForEach-Object { "• " + $_ }) -join "`r"
    $shape = Add-Text $slide $x $y $w $h $text $size $C.Soft
    $shape.TextFrame2.TextRange.ParagraphFormat.SpaceAfter = 3
}

function Add-Step($slide, [double]$x, [double]$y, [string]$num, [string]$title, [string]$body, [int]$accent = $C.Blue) {
    Add-Rect $slide $x $y 58 58 $C.Card $accent 2.2 | Out-Null
    Add-Text $slide ($x + 16) ($y + 13) 26 28 $num 19 $C.White $true 0 2 | Out-Null
    Add-Text $slide ($x + 78) ($y + 1) 405 23 $title 16 $C.White $true | Out-Null
    Add-Text $slide ($x + 78) ($y + 28) 445 36 $body 10.7 $C.Soft | Out-Null
}

function New-RoadmapSlide($slide) {
    Add-Header $slide "Member A" "논문 차용 방식과 전략 진화 로드맵" "직접 재현이 아니라, 문헌 개념을 로컬 CLAIM 데이터에 맞게 번역한 검증 흐름입니다."
    $borrowRows = @(
        @("구분", "판단 기준", "이번 분석의 예"),
        @("직접 수식", "논문의 공식을 그대로 코드화", "해당 없음"),
        @("개념 차용", "논문 아이디어를 자체 구현", "doctor-shopping, peer Z-score"),
        @("독자 설계", "논문 근거 없이 EDA·도메인 설계", "hosp_switch, vlid_sum, LOO exposure")
    )
    Add-Table $slide 44 200 865 86 $borrowRows @(100, 305, 460) -1 | Out-Null
    $rows = @(
        @("버전", "핵심 변화", "피처", "AUC", "F1", "Recall", "Precision"),
        @("V1", "베이스라인: 고객 + 청구 기본 집계", "30", "0.9019", "0.5666", "0.6199", "0.5217"),
        @("V2", "+ KMeans(k=5) / PCA(3)", "35", "0.9011", "0.5648", "0.5424", "0.5892"),
        @("V3", "- 비지도 / + claim-domain 파생", "42", "0.9054", "0.5736", "0.5461", "0.6041"),
        @("V4", "+ 행동 집계·peer-Z·고객 청구 이력", "88", "0.9200", "0.6147", "0.6255", "0.6043"),
        @("V5", "- 다중공선성 11개 제거", "77", "0.9188", "0.5971", "0.5756", "0.6203")
    )
    Add-Table $slide 44 310 865 150 $rows @(58, 358, 55, 74, 74, 84, 94) 5 | Out-Null
    Add-Text $slide 44 477 850 32 "평가 대상: labeled 20,607명(SIU=1,806, 8.76%), test set 6,183명. 검증: artifacts/member_a/_compare_v1_to_v5.json, _eda_v1_to_v5.json 기준. 원고의 V2 AUC 0.8989는 현 산출물 기준 0.9011로 정정." 8.8 $C.Muted | Out-Null
}

function New-V1Slide($slide) {
    Add-Header $slide "Member A" "V1 — 베이스라인" "고객 정보와 청구 기본 집계만으로 출발한 기준선입니다."
    $blockRows = @(
        @("블록", "피처 예시", "구현 방식"),
        @("고객 정보", "SEX, AGE, CTPR, OCCP, TOTALPREM", "원본 데이터"),
        @("청구 집계", "claim_cnt, nunique_hosp, paym_rate", "pandas groupby"),
        @("결측 지시", "MATE_OCCP, MINCRDT", "MNAR 2개만 독자 선정")
    )
    Add-Table $slide 44 205 500 110 $blockRows @(100, 250, 150) -1 | Out-Null

    Add-Rect $slide 575 205 290 110 $C.Card $C.Line 1.2 | Out-Null
    Add-CardTitle $slide 595 222 250 "처리 방식: 독자 구현"
    Add-Bullets $slide 595 250 250 52 @(
        "GroupMeanImputer, Median/ModeImputer",
        "TargetEncoder는 train-only 적합",
        "IQRCapper + RobustScaler"
    ) 8.8

    $rows = @(
        @("ANOVA 상위", "F값", "해석"),
        @("nunique_hosp", "4,793.88", "이용 병원 수"),
        @("claim_cnt", "3,639.99", "청구 건수"),
        @("mean_hosp_days", "1,982.25", "평균 입원일수"),
        @("sum_paym_amt", "597.96", "지급액 합"),
        @("is_heed_hosp", "423.02", "유의병원 여부")
    )
    Add-Table $slide 44 350 500 125 $rows @(170, 100, 220) -1 | Out-Null
    Add-MetricCard $slide 575 350 86 64 "336" "TP" "" $C.Green
    Add-MetricCard $slide 675 350 86 64 "308" "FP" "" $C.Red
    Add-MetricCard $slide 775 350 86 64 "206" "FN" "" $C.Yellow
    Add-Text $slide 575 432 290 36 "AUC 0.9019 · F1 0.5666 · Recall 0.6199 · Precision 0.5217" 12 $C.White $true | Out-Null
    Add-Text $slide 44 500 820 18 "한계: '얼마나, 어디서'는 포착하지만 '어떤 패턴으로 비정상적인가'는 약합니다." 10 $C.Yellow $true | Out-Null
}

function New-V2Slide($slide) {
    Add-Header $slide "Member A" "V1→V2 — 비지도 학습 실험" "KMeans/PCA는 EDA에는 쓸모가 있었지만, 최종 모델에는 새 정보를 거의 주지 못했습니다."
    Add-Rect $slide 44 205 380 80 $C.Card $C.Line 1.2 | Out-Null
    Add-CardTitle $slide 64 222 330 "구현: 완전 독자 설계"
    Add-Text $slide 64 252 330 24 "StandardScaler → KMeans(k=5, n_init=10) + PCA(3). DIVIDED_SET==1에서만 fit 후 전체 transform." 9.4 $C.Soft | Out-Null
    Add-Text $slide 64 276 330 12 "중요도 top20 신규 4개: pca_2 #4, kmeans_dist #7, pca_1 #13, pca_0 #14" 7.8 $C.Yellow $true | Out-Null

    $qRows = @(
        @("k", "Sil.", "DBI", "ARI/NMI"),
        @("2", "0.121", "2.876", "0.0118 / 0.0023"),
        @("5", "0.111", "2.182", "0.0281 / 0.0330"),
        @("10", "0.134", "1.555", "-")
    )
    Add-Table $slide 454 205 410 86 $qRows @(45, 65, 65, 210) -1 | Out-Null

    Add-Rect $slide 44 320 250 126 $C.Card2 $C.Blue 1.2 | Out-Null
    Add-Text $slide 64 336 210 20 "k=5 C3: 강한 EDA segment" 13 $C.White $true | Out-Null
    Add-Text $slide 64 365 205 54 "고객 5.1%, SIU 46.2%, 전체 SIU의 27.0%, Lift 5.27. 다만 단독 F1=0.340이고 claim_cnt·nunique_hosp·금액의 재표현입니다." 9.6 $C.Soft | Out-Null

    Add-Rect $slide 322 320 250 126 $C.Card $C.Line 1.2 | Out-Null
    Add-Text $slide 342 336 210 20 "PCA 진단" 13 $C.White $true | Out-Null
    Add-Text $slide 342 365 205 54 "PC0~2 누적 설명분산 30.7%. PC1 SMD=0.870이나 V1 mean_hosp_days SMD=1.085보다 낮고, PC5가 PC2보다 SIU 신호가 큽니다." 9.6 $C.Soft | Out-Null

    $abRows = @(
        @("15회 ablation", "F1 평균", "판단"),
        @("base only", "0.5914", "-"),
        @("+ KMeans only", "0.5941", "+0.0027, 표준편차 내"),
        @("+ PCA only", "0.5906", "-0.0008"),
        @("V2: KMeans+PCA", "0.5905", "-0.0009")
    )
    Add-Table $slide 600 320 264 126 $abRows @(118, 62, 84) -1 | Out-Null
    Add-Text $slide 44 480 820 32 "해석: Gain 중요도는 높았지만 ablation에서 full V2가 base보다 -0.0009 낮았습니다. 즉 모델이 사용한 재표현일 뿐, 검증 성능 기여는 없어서 최종 피처에서 제외했습니다." 10.0 $C.Yellow $true | Out-Null
}

function New-V3Slide($slide) {
    Add-Header $slide "Member A" "V2→V3 — 도메인 파생변수로 전환" "비지도 기하학 대신, 사기 탐지 문헌에서 반복되는 claim-domain 집계로 바꿨습니다."
    Add-Rect $slide 52 207 410 92 $C.Card $C.Line 1.2 | Out-Null
    Add-Text $slide 74 224 370 24 "문헌 근거를 안전하게 해석" 15 $C.White $true | Out-Null
    Add-Text $slide 74 252 360 34 "Herland 2018의 numeric aggregation, du Preez 2025의 청구 금액·빈도 이상치 개념을 우리 청구 컬럼에 맞게 번역했습니다." 10.3 $C.Soft | Out-Null

    $rows = @(
        @("신규 파생", "구현", "근거 유형"),
        @("ma3_max_dmnd_amt", "DMND_AMT max", "개념 차용"),
        @("ma3_std_dmnd_amt", "DMND_AMT std", "개념 차용"),
        @("ma3_mean_paym_dmnd_ratio", "PAYM/DMND mean", "독자 설계"),
        @("ma3_max_nonpay_ratio", "non-pay max", "독자 설계"),
        @("ma3_nunique_acci_dvsn", "ACCI_DVSN nunique", "독자 설계"),
        @("ma3_doc/hosp_siu_expo", "train LOO exposure", "완전 독자 설계")
    )
    Add-Table $slide 52 326 464 150 $rows @(176, 154, 124) -1 | Out-Null

    Add-MetricCard $slide 560 208 132 76 "+0.0065" "AUC" "0.898? 대신 현 기준 0.9011→0.9054" $C.Green
    Add-MetricCard $slide 710 208 132 76 "+0.0088" "F1" "0.5648→0.5736" $C.Green
    Add-MetricCard $slide 560 306 132 76 "+0.0149" "Precision" "0.5892→0.6041" $C.Green
    Add-MetricCard $slide 710 306 132 76 "-11 FP" "오탐" "205→194" $C.Green
    Add-Rect $slide 560 414 282 64 $C.Card2 $C.Blue 1.3 | Out-Null
    Add-Text $slide 581 427 245 38 "신규 중요도: ma3_std_dmnd_amt #3 gain=480, max_dmnd #12, max_nonpay #18" 11.8 $C.White $true | Out-Null
}

function New-V4Slide($slide) {
    Add-Header $slide "Member A" "V3→V4 — 행동 패턴·peer 이상치 확장" "가장 큰 개선 구간입니다. Precision을 유지하면서 Recall을 끌어올렸습니다."
    $rows = @(
        @("핵심 피처", "검증", "블록", "근거 유형"),
        @("claim_agg_n_doc", "F=5,105", "claim_agg", "doctor-shopping 개념"),
        @("cust_claim_first_claim_lag_m", "Gain #3=325", "cust_claim", "독자 설계"),
        @("peer_z_paym_sum_spec", "Gain #4=309", "peer_z", "peer 비교 개념"),
        @("claim_agg_vlid_std", "Gain #8=246", "claim_agg", "집계 개념"),
        @("cust_claim_vlid_sum", "F=5,011 / Gain=223", "cust_claim", "독자 설계")
    )
    Add-Table $slide 44 205 520 135 $rows @(170, 100, 110, 140) -1 | Out-Null

    Add-Rect $slide 600 205 264 135 $C.Card $C.Line 1.2 | Out-Null
    Add-CardTitle $slide 620 222 220 "peer-Z 실제 구현"
    Add-Text $slide 620 254 220 22 "Z = (x_i - μ_spec) / (σ_spec + 1e-6)" 11.2 $C.White $true | Out-Null
    Add-Text $slide 620 284 220 34 "논문 수식 복제가 아니라, '유사 집단 대비 편차' 개념을 HOSP_SPEC_DVSN 기준으로 재해석한 표준 Z-score입니다." 9.5 $C.Soft | Out-Null

    Add-Rect $slide 44 365 250 74 $C.Card2 $C.Blue 1.2 | Out-Null
    Add-Text $slide 64 380 210 20 "문헌 개념 차용" 12.5 $C.White $true | Out-Null
    Add-Text $slide 64 407 205 18 "doctor-shopping, peer comparison, studentized/normalized comparison" 8.8 $C.Soft | Out-Null
    Add-Rect $slide 322 365 250 74 $C.Card2 $C.Yellow 1.2 | Out-Null
    Add-Text $slide 342 380 210 20 "독자 설계" 12.5 $C.White $true | Out-Null
    Add-Text $slide 342 407 205 18 "hosp_switch, recp_gap, weeks_multi_hosp, vlid_sum" 8.8 $C.Soft | Out-Null
    Add-Rect $slide 600 365 264 74 $C.Card2 $C.Green 1.2 | Out-Null
    Add-Text $slide 620 380 220 20 "성능 점프" 12.5 $C.White $true | Out-Null
    Add-Text $slide 620 407 220 18 "신규 11개 top20 진입 · AUC +0.0146 · F1 +0.0411" 8.8 $C.Soft | Out-Null

    Add-Text $slide 44 476 820 32 "해석: 중요도와 ablation이 처음으로 같은 방향을 가리킨 구간입니다. V4 신규 피처 11개가 top20에 들어갔고, Precision은 유지하면서 미탐(FN)을 43건 줄였습니다." 10.0 $C.Yellow $true | Out-Null
}

function New-V5Slide($slide) {
    Add-Header $slide "Member A" "V4→V5 — 다중공선성 제거 실험" "중복 제거는 해석에는 도움이 됐지만, 탐지 목적에서는 손실이 더 컸습니다."
    $rows = @(
        @("제거 피처", "보존 피처", "검증 포인트"),
        @("cust_claim_claim_rows", "claim_cnt", "r=1.000, 완전 중복"),
        @("cust_claim_dmnd_sum", "paym_sum", "r=0.996, 금액 합계 중복"),
        @("cust_claim_top_hosp_share", "hosp_nunique", "음의 강상관, 보존 가능"),
        @("cust_claim_vlid_sum", "cust_claim_vlid_max", "r=0.620, r_SIU 0.4423 vs 0.2680")
    )
    Add-Table $slide 54 210 520 130 $rows @(180, 150, 190) 5 | Out-Null

    Add-Rect $slide 54 372 520 82 $C.Card2 $C.Blue 1.3 | Out-Null
    Add-Text $slide 76 388 470 38 "핵심 발견: V5는 신규 추가 없이 88→77개 제거입니다. V4 top20의 cust_claim_vlid_sum(rank #13, gain=223)을 제거하면서 Recall 손실이 커졌습니다." 12.3 $C.White $true | Out-Null

    Add-MetricCard $slide 620 210 112 74 "0.0012" "AUC 하락" "0.9200→0.9188" $C.Red
    Add-MetricCard $slide 756 210 112 74 "0.0176" "F1 하락" "0.6147→0.5971" $C.Red
    Add-MetricCard $slide 620 310 112 74 "5.0%p" "Recall 하락" "0.6255→0.5756" $C.Red
    Add-MetricCard $slide 756 310 112 74 "+1.6" "Precision %p" "0.6043→0.6203" $C.Green
    Add-MetricCard $slide 620 410 112 74 "27 TP" "탐지 손실" "339→312" $C.Red
    Add-MetricCard $slide 756 410 112 74 "31 FP" "오탐 감소" "222→191" $C.Green
    Add-Text $slide 54 500 800 18 "해석: 보험사기 탐지는 미탐(FN) 비용이 크므로, V5의 보수화보다 V4의 탐지력 우위가 더 중요합니다." 10.5 $C.Yellow $true | Out-Null
}

function New-FinalSlide($slide) {
    Add-Header $slide "Member A" "최종 판단 — V4 채택" "논문을 그대로 재현한 전략이 아니라, 문헌 아이디어를 로컬 CLAIM 컬럼으로 번역한 feature engineering입니다."
    Add-MetricCard $slide 58 210 142 84 "0.9200" "AUC" "5개 버전 중 최고" $C.Yellow
    Add-MetricCard $slide 224 210 142 84 "0.6147" "F1" "최고 F1" $C.Yellow
    Add-MetricCard $slide 390 210 142 84 "0.6255" "Recall" "최고 탐지력" $C.Green
    Add-MetricCard $slide 556 210 142 84 "0.6043" "Precision" "V3 수준 유지" $C.Blue2
    Add-MetricCard $slide 722 210 142 84 "V4" "최종 전략" "member_a_strategy_4" $C.Yellow

    $rows = @(
        @("피처", "논문 근거/출처", "근거 유형", "EDA 검증"),
        @("claim_agg_n_doc", "du Preez p.5 doctor shopping", "개념 차용", "F=5,105 #1"),
        @("peer_z_n_hosp_spec", "du Preez p.5 peer comparison", "개념 차용", "F=3,870"),
        @("ma3_std_dmnd_amt", "Herland p.11 std + 금액 이상치", "개념 차용", "집계 방식 검증"),
        @("cust_claim_vlid_sum", "논문 근거 없음", "완전 독자 설계", "F=5,011, r_SIU=0.4423"),
        @("hosp_switch_count", "논문 근거 없음", "완전 독자 설계", "F=4,564")
    )
    Add-Table $slide 58 334 806 138 $rows @(180, 270, 140, 216) -1 | Out-Null
    Add-Rect $slide 58 494 806 35 $C.Card2 $C.Green 1.3 | Out-Null
    Add-Text $slide 76 501 760 22 "최종 1줄: V2는 중요도만 높고 성능 기여가 없었고, V4는 신규 11개 top20 진입과 최대 성능 도약이 함께 확인되어 최종 채택했습니다." 9.8 $C.White $true | Out-Null
}

$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($InputPpt, $false, $false, $false)
try {
    $global:SX = [double]$pres.PageSetup.SlideWidth / 960.0
    $global:SY = [double]$pres.PageSetup.SlideHeight / 540.0
    $global:SF = [Math]::Min($global:SX, $global:SY)
    $pres.Slides.Item(15).Delete()
    $insertAt = 15
    $builders = @(
        ${function:New-RoadmapSlide},
        ${function:New-V1Slide},
        ${function:New-V2Slide},
        ${function:New-V3Slide},
        ${function:New-V4Slide},
        ${function:New-V5Slide},
        ${function:New-FinalSlide}
    )
    for ($i = 0; $i -lt $builders.Count; $i++) {
        $s = $pres.Slides.Add($insertAt + $i, $ppLayoutBlank)
        & $builders[$i] $s
    }
    if (Test-Path $OutputPpt) { Remove-Item -LiteralPath $OutputPpt -Force }
    $pres.SaveAs($OutputPpt)
}
finally {
    $pres.Close()
    $app.Quit()
}

Write-Output $OutputPpt
