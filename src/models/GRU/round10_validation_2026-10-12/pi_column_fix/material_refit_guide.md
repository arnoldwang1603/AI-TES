# material refit guide

本列来自预注册，仅说明输出参照的附加拟合参数。direct 的 none 不代表换材料后训练数据缩放器或 GRU 权重无需重新拟合，也不代表材料迁移已验证。physics_input_refit 单独列出物理输入需要更新的物性。

| output_form | 换材料时每种输出形式要重拟合什么 | refit_rule_source | status |
|---|---|---|---|
| direct | none | batches.P3.re_fit_column | protocol annotation only; no new-material validation |
| anchor | mu/sigma | batches.P3.re_fit_column | protocol annotation only; no new-material validation |
| EMA | tau | batches.P3.re_fit_column | protocol annotation only; no new-material validation |
| ratio | mu/sigma | batches.P3.re_fit_column | protocol annotation only; no new-material validation |
| physics residual | k, rho, cp or one diffusivity | batches.P3.re_fit_column | protocol annotation only; no new-material validation |
