# c16 validation points with material refit

Retained real ep2 smoke values; annotation-only preview from the unchanged registered arm definitions. No new predictions or formal scientific verdict. 本列来自预注册，仅说明输出参照的附加拟合参数。direct 的 none 不代表换材料后训练数据缩放器或 GRU 权重无需重新拟合，也不代表材料迁移已验证。physics_input_refit 单独列出物理输入需要更新的物性。

| arm | channel | set | model_MAE_C | seed_file_sd_C | n_seed_files | reference | reference_residual_std_C | reference_residual_mean_C | target_std_C | n_training_files | n_training_steps | output_form | physics_input_refit | refit_rule_source | 换材料时每种输出形式要重拟合什么 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_mask | T_avg | val | 223.224 | 88.1007 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| B1_mask | T_inner | val | 0.589809 | 0.548222 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| B1_mask | T_outer | val | 176.426 | 103.694 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| B2_mask | T_avg | val | 265.157 | 92.3695 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| B2_mask | T_inner | val | 0.587943 | 0.542257 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| B2_mask | T_outer | val | 226.565 | 114.413 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| C11 | T_avg | val | 0.408677 | 0.358114 | 17 | physics | 0.46569 | 0.199868 | 106.72 | 169 | 243360 | physics residual | none | batches.P3.re_fit_column | k, rho, cp or one diffusivity |
| C11 | T_inner | val | 0.583345 | 0.547667 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| C11 | T_outer | val | 0.540502 | 0.459351 | 20 | physics | 0.709983 | 0.429772 | 114.351 | 284 | 408960 | physics residual | none | batches.P3.re_fit_column | k, rho, cp or one diffusivity |
| C12 | T_avg | val | 0.408797 | 0.358689 | 17 | physics | 0.46569 | 0.199868 | 106.72 | 169 | 243360 | physics residual | k, rho, cp or one diffusivity | batches.P3.re_fit_column | k, rho, cp or one diffusivity |
| C12 | T_inner | val | 0.57929 | 0.548084 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | k, rho, cp or one diffusivity | batches.P3.re_fit_column | mu/sigma |
| C12 | T_outer | val | 0.540626 | 0.459782 | 20 | physics | 0.709983 | 0.429772 | 114.351 | 284 | 408960 | physics residual | k, rho, cp or one diffusivity | batches.P3.re_fit_column | k, rho, cp or one diffusivity |
| C13 | T_avg | val | 266.981 | 93.1002 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | k, rho, cp or one diffusivity | batches.P3.re_fit_column | none |
| C13 | T_inner | val | 0.585288 | 0.542918 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | k, rho, cp or one diffusivity | batches.P3.re_fit_column | mu/sigma |
| C13 | T_outer | val | 190.774 | 101.143 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | k, rho, cp or one diffusivity | batches.P3.re_fit_column | none |
| C14 | T_avg | val | 263.726 | 92.2976 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| C14 | T_inner | val | 273.813 | 89.2272 | 20 | none | 115.033 | 293.44 | 115.033 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| C14 | T_outer | val | 227.693 | 114.498 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| C15 | T_avg | val | 265.157 | 92.3695 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| C15 | T_inner | val | 0.498343 | 0.449733 | 20 | inlet | 0.593088 | -0.137615 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| C15 | T_outer | val | 226.565 | 114.413 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| C2 | T_avg | val | 55.9122 | 35.8291 | 17 | initial | 70.8827 | 18.8952 | 106.72 | 169 | 243360 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| C2 | T_inner | val | 0.587207 | 0.543099 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| C2 | T_outer | val | 42.7249 | 29.3052 | 20 | initial | 56.4632 | 15.6443 | 114.351 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| C3 | T_avg | val | 223.048 | 89.8209 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| C3 | T_inner | val | 0.587677 | 0.542579 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| C3 | T_outer | val | 84.8632 | 55.6643 | 20 | inlet | 94.8474 | -13.8692 | 114.351 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| ENC_16_16 | T_avg | val | 265.304 | 92.3568 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| ENC_16_16 | T_inner | val | 0.587837 | 0.542309 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| ENC_16_16 | T_outer | val | 226.857 | 114.472 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| ENC_linear | T_avg | val | 265.492 | 92.3751 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| ENC_linear | T_inner | val | 0.5879 | 0.542281 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| ENC_linear | T_outer | val | 226.887 | 114.6 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| G7a | T_avg | val | 265.133 | 92.3635 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| G7a | T_inner | val | 0.587932 | 0.542279 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| G7a | T_outer | val | 226.5 | 114.4 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| G7b | T_avg | val | 265.089 | 92.3185 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| G7b | T_inner | val | 0.58793 | 0.542244 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| G7b | T_outer | val | 226.718 | 114.462 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| G7c | T_avg | val | 265.227 | 92.3887 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| G7c | T_inner | val | 0.58796 | 0.542252 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| G7c | T_outer | val | 226.765 | 114.499 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| G7d | T_avg | val | 263.198 | 92.2769 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| G7d | T_inner | val | 0.587731 | 0.542452 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| G7d | T_outer | val | 222.723 | 114.016 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| H0_tanh | T_avg | val | 265.158 | 92.3701 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| H0_tanh | T_inner | val | 0.587943 | 0.542257 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| H0_tanh | T_outer | val | 226.569 | 114.416 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| H0_zero | T_avg | val | 265.643 | 92.4151 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| H0_zero | T_inner | val | 0.587885 | 0.542283 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| H0_zero | T_outer | val | 226.725 | 114.467 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| H8_a12_l0 | T_avg | val | 265.159 | 92.3695 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| H8_a12_l0 | T_inner | val | 0.569391 | 0.51661 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| H8_a12_l0 | T_outer | val | 226.564 | 114.413 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| H8_a12_l1 | T_avg | val | 313.778 | 94.7938 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| H8_a12_l1 | T_inner | val | 0.646733 | 0.547129 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| H8_a12_l1 | T_outer | val | 229.155 | 114.246 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| H8_a1_l1 | T_avg | val | 313.8 | 94.7955 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| H8_a1_l1 | T_inner | val | 0.586765 | 0.549744 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| H8_a1_l1 | T_outer | val | 229.15 | 114.246 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| H8_a6_l0 | T_avg | val | 265.158 | 92.3695 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| H8_a6_l0 | T_inner | val | 0.578392 | 0.530408 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| H8_a6_l0 | T_outer | val | 226.565 | 114.413 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| H8_a6_l1 | T_avg | val | 313.791 | 94.7948 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| H8_a6_l1 | T_inner | val | 0.594922 | 0.559758 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| H8_a6_l1 | T_outer | val | 229.152 | 114.246 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| H8_inv_kc | T_avg | val | 265.073 | 92.3685 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| H8_inv_kc | T_inner | val | 0.587657 | 0.542367 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| H8_inv_kc | T_outer | val | 226.76 | 114.418 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| NO_TIME | T_avg | val | 356.875 | 93.3053 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| NO_TIME | T_inner | val | 0.595758 | 0.542327 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| NO_TIME | T_outer | val | 232.689 | 113.197 | 20 | none | 114.351 | 279.72 | 114.351 | 284 | 408960 | direct | none | batches.P3.re_fit_column | none |
| R1 | T_avg | val | 223.023 | 89.7934 | 17 | none | 106.72 | 283.567 | 106.72 | 169 | 243360 | direct | none | batches.P3.re_fit_column | none |
| R1 | T_inner | val | 0.587681 | 0.542475 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| R1 | T_outer | val | 81.5784 | 52.5932 | 20 | ratio_kelvin | 88.1679 | 4.16576 | 114.351 | 284 | 408960 | ratio | none | batches.P3.re_fit_column | mu/sigma |
| R2 | T_avg | val | 55.0585 | 29.027 | 17 | ratio_kelvin | 65.8798 | 6.85749 | 106.72 | 169 | 243360 | ratio | none | batches.P3.re_fit_column | mu/sigma |
| R2 | T_inner | val | 0.587217 | 0.542838 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| R2 | T_outer | val | 81.5466 | 52.5951 | 20 | ratio_kelvin | 88.1679 | 4.16576 | 114.351 | 284 | 408960 | ratio | none | batches.P3.re_fit_column | mu/sigma |
| R2C | T_avg | val | 60.4423 | 38.3507 | 17 | ratio_celsius20 | 69.0374 | -19.3429 | 106.72 | 169 | 243360 | ratio | none | batches.P3.re_fit_column | mu/sigma |
| R2C | T_inner | val | 0.586752 | 0.543595 | 20 | inlet | 0.63902 | -0.149335 | 115.033 | 284 | 408960 | anchor | none | batches.P3.re_fit_column | mu/sigma |
| R2C | T_outer | val | 84.0481 | 52.8983 | 20 | ratio_celsius20 | 95.8118 | -19.5967 | 114.351 | 284 | 408960 | ratio | none | batches.P3.re_fit_column | mu/sigma |
| R3 | T_avg | val | 55.0435 | 29.0589 | 17 | ratio_kelvin | 65.8798 | 6.85749 | 106.72 | 169 | 243360 | ratio | none | batches.P3.re_fit_column | mu/sigma |
| R3 | T_inner | val | 46.1972 | 26.6556 | 20 | ratio_kelvin | 57.803 | 12.0329 | 115.033 | 284 | 408960 | ratio | none | batches.P3.re_fit_column | mu/sigma |
| R3 | T_outer | val | 81.5356 | 52.6519 | 20 | ratio_kelvin | 88.1679 | 4.16576 | 114.351 | 284 | 408960 | ratio | none | batches.P3.re_fit_column | mu/sigma |
