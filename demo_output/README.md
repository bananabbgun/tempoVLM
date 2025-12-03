# TempoVLM Complete Demo Results

## 生成時間: 2025-12-03 02:19:14

## 總共 1 個場景

| 場景 | 時序一致性 | 深度排序 | 軌跡預測 | 遮擋測試 |
|------|-----------|---------|---------|---------|
| scene0000_00 | ✅ | ✅ | ✅ | ✅ |

## 每個場景包含:

1. **temporal_consistency.mp4** - 時序一致性對比影片
   - Base Model vs Unified Model 特徵相似度曲線

2. **depth_ordering.mp4** - 深度排序測試
   - 三個區域 (左/中/右) 深度比較

3. **trajectory.mp4** - 軌跡預測
   - 俯視圖顯示 GT 軌跡

4. **occlusion_test.mp4** - 遮擋測試 ⭐ NEW
   - 遮擋配置: Frame 15-25, ratio=0.4, type=black
   - GT / 遮擋 / 注入後 描述對比

5. **occlusion_results.json** - 遮擋測試詳細結果
   - 每幀的異常分數、品質、描述文字

## 統計摘要

### scene0000_00
- 時序一致性改善: 2.88%
- 遮擋檢測率: 10.0%
- 成功注入數: 13
