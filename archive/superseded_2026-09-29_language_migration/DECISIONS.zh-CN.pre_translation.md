# 决策记录


## D1 (2026-09-29) 切分沿用官方 960/120/120+400

```json
{
  "id": "D1",
  "date": "2026-09-29",
  "question": "提案 §3.2 的 30/80/60 三分法 vs 官方 960/120/120+400",
  "decision": "A -- 沿用官方切分 train 960 / dev 120 / calibration 120 / test 400",
  "decided_by": "project owner",
  "evidence": {
    "independent_unit": "case (每 case 5 题, 同 case 内不独立)",
    "scaling_law_verified": "test 400 cases SE=1.067pp; dev 120 cases SE=2.140pp; 1/sqrt(cases) 预测 1.948pp (实测为其 1.10x)",
    "proposal_medium_M": {
      "tasks": 80,
      "approx_cases": 16,
      "se_pp": 5.33,
      "bonferroni_H1000_pp": 21.63
    },
    "official_dev": {
      "cases": 120,
      "se_pp": 2.14,
      "bonferroni_H1000_pp": 12.38
    },
    "usable_headroom_pp": 20.0,
    "note": "起点 0.8150, 理论上限 1.0 -> 可用空间约 18.5pp, 上界取 20pp"
  },
  "consequence": "medium M 的判定阈值 21.63pp > 全部可用空间, 该切分在 H≈1000 下无法分辨真改进与噪声. 官方 dev 的 12.38pp < 可用空间, 可判定.",
  "what_is_given_up": "提案的三分法(proxy/medium/frozen)本身; 改用 dev 选型 + calibration 校准 + test 终评, 与提案 Rule 1 的意图(冻结集只用于预注册检查点)一致, 仅规模不同."
}
```

## D2 (2026-09-29) QLoRA 的必要性: 显存不构成约束

```json
{
  "id": "D2",
  "date": "2026-09-29",
  "question": "提案 §2.1/§2.3: QLoRA (4-bit) 是否为 12 GB 下的必需路径",
  "status": "纸面算术已证伪, 实测进行中",
  "claim_1_full_param_infeasible": {
    "verdict": "HOLDS",
    "evidence": "0.6B fp32 AdamW = 8.94 GiB (weights 2.24 + grads 2.24 + m/v 4.47) > 12.28 GiB card"
  },
  "claim_2_4bit_required": {
    "verdict": "DOES NOT FOLLOW from claim 1",
    "evidence": {
      "base": "Qwen3-0.6B: 28 layers, hidden 1024, GQA 16/8, intermediate 3072",
      "lora_r16_all_linear_trainable": "168.82M = 28.1% of backbone",
      "estimated_footprint": {
        "base_bf16_gib": 1.12,
        "lora_params_grads_gib": 0.62,
        "lora_adamw_fpv_gib": 1.26,
        "activations_ctx_gib": 1.89,
        "total_gib": 4.84
      },
      "qlora_estimate_gib": 4.0,
      "saving_from_4bit_gib": 0.84,
      "card_gib": 12.28
    },
    "reading": "4-bit 省下 0.84 GiB, 而卡有 12.28 GiB. 显存不构成约束.",
    "the_real_cost": "4-bit 把权重舍入误差注入每一次前向, 该误差进入 Floor B (run-to-run SD); Floor B 决定 epsilon; epsilon 越大可检出改进越少. 提案 §2.3 把量化列为 'fixed baseline' 视作免费, 但它实际是在决定 accept rule 要容忍多少噪声.",
    "decision_deferred_to": "Floor B 双臂实测 (BF16-LoRA vs QLoRA), 判据预注册后再跑"
  },
  "consequence": "不因显存而采用 QLoRA. 是否采用由数值决定, 即 Floor B 测量. 在此之前 epsilon 不得引用 QLoRA 路径的 Floor B."
}
```

## D3 (2026-09-29) 手稿溯源: 抓出并修正一处 Floor B 污染

```json
{
  "id": "D3",
  "date": "2026-09-29",
  "question": "手稿的每个数字是否可溯源到产物, 且未被他臂污染",
  "status": "发现并修正一处真实污染; 溯源链升级",
  "finding": {
    "what": "手稿初稿把 Floor B = 0.40 pp 当作本目标的 Floor B 引用, 并据此给出 ε(dev) = 12.38 pp 作为既定阈值",
    "truth": "0.40 pp 测自 laya-multilingual 322M 全参臂 (另一条训练路径). 校准文件自己写着 'Do not inherit it silently'. 12.38 pp 确实代入了这个借来的 Floor B (见 DAY1_BASELINE_2026-09-29.md:67,72-73), 且该文档同时标注 'Floor B on QLoRA 种子 = 未测'",
    "direction_of_error": "乐观. Floor B 以平方项进入, 借来的值偏小 => 真实阈值只会更大. 即 12.38 pp 是下界",
    "how_found": "measurement/audit_manuscript_numbers.py 比对 12.38 与校准文件的 4.95 时报错, 追查出两个 ε 来自两个不同目标"
  },
  "corrections": [
    "手稿 §2 改为带'是否可迁移'列的表: Floor A 可迁移(切分的属性), Floor B 不可迁移(优化器的属性), ε(dev) 标注 provisional",
    "手稿显式引用校准文件自身的告诫原文, 使警告随数字一同传播",
    "τ_test = 3.11 pp 与 accept-rule 审计标注为他臂实测, 且本种子的 test 阈值在测得前不可给出",
    "§6 状态表把 'Floor B on this seed' 标为 blocking 而非 done, 与 §2 的 provisional 保持一致"
  ],
  "provenance_upgrade": {
    "was": "基线数字的唯一来源是评测器 stdout 的人工转录 (docs/DAY1_PROGRESS.md) -> 只能做存在性检查",
    "weakness_proved_by": "负控制 N4: 手稿中 0.8150 出现 6 次, 改 1 处后审计仍 PASS —— 存在性检查抓不到错值",
    "now": "measurement/recompute_day1_baseline.py 从 day1_raw_dev.json 的逐题 logits, 调用冻结评测器自己的 compute_metrics/probabilities/chance_corrected_skill 重算两个臂 -> 12 个指标全部与记录一致, 落盘为 day1_baseline_metrics.json. 审计改为按值比对",
    "why_not_reimplement": "重实现 ECE 就是第二个 ECE 定义, 恰在手稿最敏感处漂移. 必须复用冻结评测器的度量函数",
    "bonus": "两臂来自同一份 logits, 故 raw vs L1 是同输入的配对比较 —— §3.2 的'校准对立'因此是干净的"
  },
  "guard_has_teeth": {
    "file": "measurement/test_manuscript_audit.py",
    "cases": 5,
    "result": "5/5 注入全部被捕获; 还原后 sha 字节一致",
    "test_bug_found": "负控制初版只替换第一处, 导致 N2/N4 假通过. 已改为全处替换并断言出现次数>=2, 使过时的用例会响亮失败而不是静默失去测试能力"
  },
  "gates": "9/9 PASS"
}
```
