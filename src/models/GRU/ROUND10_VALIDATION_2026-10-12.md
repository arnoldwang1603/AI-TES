# Round 10 包验证记录（开发者交接）

版本日期沿用交接包：2026-10-12。本文记录的是实现与运行路径验证。六个新钩子已按 H8 → H7 → NO_TIME → H10 → H5 → H6 完成；每个钩子后都执行了七项旧检查。完整冒烟已退出 0，69/69 臂通过。

**未启动正式 Round 10 训练；未读取真实测试 CSV；未执行真实 B6/F11 评分。** 这里的 B6/F11 检查使用合成数据和临时访问台账，未消耗包内的真实测试窗口。两轮训练结果仅用于验证运行、保存和验证集复现，不用于模型优劣判决。

本文适合继续开发或审计的人。组员安装、启动、进度和交回内容见 [ROUND10_RUNBOOK.md](ROUND10_RUNBOOK.md)。规格见 [ROUND10_IMPLEMENTATION_SPEC.md](ROUND10_IMPLEMENTATION_SPEC.md)。可携带验证证据统一放在 [round10_validation_2026-10-12](round10_validation_2026-10-12/)，不把试跑 checkpoint 放进交付包。

## 1. 最强的默认行为回归证据

[independent_validation_regression.json](round10_validation_2026-10-12/independent_validation_regression.json) 对比的是冻结的原 P0 包和新增钩子后的默认实现：两个实现分别拟合常数，加载同一个历史冠军 checkpoint，在其原验证划分的全部 20 个文件上分别执行完整 rollout。每个文件 1440 步、三个温度通道均纳入对比。

- `ok=true`，`source_verified=true`；原 P0 的 13 个源文件通过来源校验。
- `TF32=legacy`；评分集合为 `validation`；`test_csv_access=false`。
- 全 20 文件、全部步和通道的 `max_delta_scaled=0.0`；验收阈值为 `1e-6`。
- `max_weighted_loss_delta=0.0`；验收阈值为 `1e-7`。
- checkpoint SHA-256：`72442eb28660fb7376008565bfad65860656a718ac0d3fb3439d2e9abe8e3a69`。
- 原 P0 源清单 SHA-256：`3ca25577a24bb39cbf8bc55762c3f88f076962063887c4864926117f4581895f`。

上述差额的单位是缩放后的温度和加权损失，不是 °C。结论只覆盖这个 checkpoint、原验证划分、默认旋钮和记录的数值环境；它证明新增默认路径保持原 P0 行为，不是重新计算 Round 9 测试集结果。

## 2. 按实现顺序执行的检查

七项回归入口保持 `check_h9 → check_h1 → check_h2 → check_h3 → check_h4 → check_a4 → check_a5`。六轮均为 0 FAIL；可携带证据目录保存每个钩子之后七项检查的 `.txt` 转存。开发时原日志位于 `round10_checks/{H8,H7,NO_TIME,H10,H5,H6}_after_*.log`，这些原日志路径仅是本地开发位置。

七项检查在这些日志中的规模为：H9 17 项、H1 18 项、H2 23 项、H3 18 项、H4 13 项、A4 14 项。A5 在 H8 后为 19 项；加入封存卫生断言后，H7 及其后五轮均为 20 项。后来的重复日志不作为额外独立证据计数。

旧检查已按 Round 10 封存要求改造：真实数据只使用训练/验证文件；历史测试预测或 metadata 可以回放，但不打开真实测试 CSV。H9/H3 通过不代表此次重新计算了 case 56 或历史测试 MAE。

各新钩子的独立检查：

- H8：3/3，通过实际 `1/KC` 权重、未拟合 KC 拒绝，以及名称、配置、重载和导出接线。
- H7：5/5，覆盖默认 256-256、线性、16-16、tanh、零初态；零初态编码器为 0 参数。
- NO_TIME：3/3，验证输入仅一列、锚读取第 0 列入口、配置与导出接线。
- H10：5/5，前 10 步权重和文件掩码共用分母，掩码通道梯度为零，尖峰指标及 60 步 T0 扫描接口存在。
- H5：11/11，覆盖 R1/R2/R3 Kelvin、R2 摄氏 20°C 下限及 `anchor_initial`。各五文件的训练/评估双路径恒等还原，最大误差不超过 `3.91e-5°C`；T_avg 比值统计排除被掩训练文件。
- H6：21/21，覆盖残差、输入加残差、仅输入；物理列在 CASE_FLAG 前，残差头零初始化，T_avg 常数拟合遵守掩码。零头验证得到 T_outer `0.543198°C`、T_avg `0.409175°C`，与 E1 行差额小于 `1e-3°C`。

H6 保留标准化常数 `mu`、`sd` 和 `z0=-mu/sd`，重建写为 `phys + mu + sd*(z0+head)`；零头因此对应原物理参照。H5 的 C2 使用 `anchor_initial`，分别以 T_outer(0)、T_avg(0) 为参照；旧 `anchor` 模式保持原定义。

## 3. F1、流水线与分析支持

初次 S0 的汇总和检查转存保存在可携带证据目录。开发时原始位置是 `runs/_round10_results/S0`；以下均通过：

- F4/F1 划分：14 项；种子 43/44、10 个留一家族折、保守近重复闭包和 304 个保留源文件哈希。
- F1/G7：23/23；全部 10 折的 scaler 和锚逐位等于原 284 文件拟合；G7(c) 20 步恒 T0 前缀读出被丢弃，原预测长度保持；G7(d) 20 个合成恒温文件仅加入训练，常数和 scaler 在增强前拟合。
- F1 fold9 全掩码处理：5/5；T_avg 保持 NaN/JSON null，`Scored_N_files_T_avg=0`，整体描述分只平均已计分通道；合成完整汇总和三张比较图均完成，无 NaN 坐标轴退出。
- 分析纯数组检查：15 项；判决检查 13/13；流水线依赖、续跑与完整性检查 17/17。

F1 的常数冻结有意使用注册的完整训练拟合池，可能包含该折留出的家族；这些文件只用于 scaler/锚等固定常数，优化仍排除被留家族。这是已声明的协议，不能把它描述为完全不使用留出家族信息。

`prepare_frozen_constants.py` 可在无训练、无测试读取的条件下创建或核验 shipped full-train sidecar。现有 sidecar与 B2 的 scaler、常数、实际权重一致时保持其原来源和哈希，不覆盖。

## 4. 最新 CPU 测试边界复核

在物理测试评分 API 和 F11 上界来源修正后，重新执行了相关 CPU/合成检查：

- `check_paper_report`：22/22。覆盖写前台账、窗口顺序、相同窗口恢复、并发锁、B6 披露范围以及未经登记的测试评分拒绝。
- `check_f11`：22/22。覆盖直接和挑战者/F10 两分支、固定 20/30 模型成员、相同配方与逐位一致真值、缺种子/缺案例拒绝、C16 已确认臂映射、67 文件标尺和 21 案例 oracle、误差预算、图和预测包。带正负误差的解析例子验证先平均预测再计算 MAE。
- `check_f11_physics_boundary`：9/9。调用真实 `eval_checkpoint.evaluate_one` 入口，模拟数据加载和 CPU 推理；残差、输入加残差、仅输入均显式向 70 个 Dataset 传入预加载物理序列。无权限、B6 权限、缺失/不完整映射和长度错误均被拒绝；验证路径保持原行为。

对应最新 `.txt` 转存位于可携带证据目录；原始 `scratch/validation_latest_check_*.log` 仅是本地开发日志。

最终封存审查又加入测试源内容冻结：首次 B6 的有效窗口内才给完整 70 CSV 建立字节哈希；B6 恢复、F11 评分前及完成前核对同一内容，拒绝源内容变化、缺源、未封存旧缓存和嵌套额外 CSV。F11 真实入口拒绝更换冻结物理参数或标签约定。更新后的 [报告边界检查](round10_validation_2026-10-12/final_paper_report_cpu.txt) 为 **37/37 PASS**。此外，[旧 runner 路径检查](round10_validation_2026-10-12/final_runner_legacy_cpu.txt) 为 **5/5 PASS**，确认 P1 预检不读测试数据，旧 gate API 也使用新的验证回归。

F11 的 `residual_upper_bound_RMSE_C` 来自冻结约定表的 `tavg_rmse_best_mix_C`，即份额拟合后的逐文件残差上界；原始未拟合份额的 `physics_yardstick_RMSE_C` 独立保留。合成检查用 `0.03` 和 `0.4` 两个不同值核验来源。

shipped 物理缓存仅含训练/验证文件。物理臂的测试评分必须在 F11 窗口由 `paper_report` 生成完整物理映射，再传入 `evaluate_one(..., physics_overrides=...)`；不会回退读取旧 Test 缓存。

## 5. Dry-run 与完整训练冒烟

[round10_all_dry_run.txt](round10_validation_2026-10-12/round10_all_dry_run.txt) 的 `--all --dry-run` 已打印全部可能分支的 run 名，当前实际可执行规模为 `420..540` run。它没有训练或测试访问；历史交接里的规模估算不应当作已执行数量。

本地开发日志 `scratch/round10_all_arm_smoke.log` 结束于 `SMOKE PASS 69/69`。可携带的逐臂验证结果见 [smoke_results.json](round10_validation_2026-10-12/smoke_results.json) 及同目录 CSV：

- 69 条记录、seed 7、每臂 2 轮，全部 `ok=true`。
- 全部 `sealed=true`、`no_test_key=true`、`sidecar=true`、`repro=true`。
- 验证重新评分与原 run metadata 的最大 `max_delta` 为 `1.1368683772161603e-13`；多数记录为 0。它是极小舍入差，不能改写成所有记录逐位零差。

冒烟覆盖常规、条件、确认与最终配方的命名/训练/保存/验证路径，不执行真实研究判决，也不读取真实 B6/F11。69 个临时 ep2 目录已在完成分析 CLI 集成验证后安全清理；逐臂检查记录与功能图保留。

五个分析 CLI 随后在真实 ep2 产物上完成代表性集成检查，全部 PASS，且 `test_csv_attempts=0`。记录见 [analysis_cli_results.json](round10_validation_2026-10-12/analysis_cli_results.json)：C16 使用 28 个 smoke run，error_budget 使用 2 个，G1 使用 1 个，E3/G11 使用 2 个，F1 envelope 使用 11 个。六张示例图随证据保存在 `round10_validation_2026-10-12/analysis_figures/`；它们验证输出路径和布局，不作为正式研究图或模型排名。

## 6. 环境与兼容性边界

[preflight.json](round10_validation_2026-10-12/preflight.json) 的实际环境是 Python `3.12.12`、PyTorch `2.8.0+cu126`、CUDA `12.6`、NVIDIA RTX 4090。已执行快照记录 `test_access=false`，训练来源文件和包输入哈希核验通过；核验文件数以最终封存后的 JSON 为准。

最新 `check_python_compat` 用 Python 3.9 grammar 解析 90 个源文件，通过；最终新增源文件后的数量以可携带证据中的最新检查转存为准。**语法检查不等于在 Python 3.9/3.10/3.11 上实际运行。** 本机 CUDA 冒烟使用的是 3.12.12；组员在目标服务器仍需先运行 runbook 的 preflight 和 smoke，再启动正式全流程。

## 7. 最终交付状态与运行边界

### PI 结果表漏列修复（2026-10-11，整包标签沿用 2026-10-12）

此前 `batches.P3.re_fit_column` 只有预注册内容，没有进入 P1 或 C16 导出。现在 `round10_refit.py` 直接解析该字段，按实际配置逐通道附加输出形式与重拟合说明；物理输入另列维护要求。P1 的内壁锚与外壁/平均温度直接头因此不会被整行归成“无”。F11 C16 表沿冻结的训练参照行继承说明，未从测试标签拟合新参数。

- [新增出口检查](round10_validation_2026-10-12/pi_column_fix/check_material_refit.txt)：40/40 PASS，包含真实 P1/C16 CSV、Markdown 输出、F11 alias 继承、跨种子冲突拒绝、动态预注册文本传播及原预注册字节不变。
- 同时复跑统计判决 13/13、流水线 17/17、分析 15/15、F11 22/22、报告门 37/37，以及 Python 3.9 语法检查（93 个源文件），均通过。记录在同目录 `pi_fix_*.txt`。
- [五种输出说明表](round10_validation_2026-10-12/pi_column_fix/material_refit_guide.csv) 和 [列输出示例](round10_validation_2026-10-12/pi_column_fix/c16_validation_points_with_material_refit.md) 已生成。示例使用保留的真实 ep2 验证汇总，其原有 84 行数值逐项保持不变；它不是正式多种子结果。EMA 只有协议说明，没有新增训练臂或 MAE。
- 该列说明的是预注册的输出参照维护要求；`direct: none` 不表示新材料上缩放器/GRU 免训练，也不表示材料迁移成功已经验证。之前的模型训练与 69 臂冒烟证据仍单独保留。

- 漏列修复后重新封存并执行完整 preflight，退出 0；全部独立检查、旧 P0 验证回归、311 个训练源文件和 731 个固定包输入核验通过。记录见 [preflight.json](round10_validation_2026-10-12/preflight.json)、[regression.json](round10_validation_2026-10-12/regression.json) 和 [最终预检日志](round10_validation_2026-10-12/final_preflight.txt)。原 729 输入的快照保留在 `pi_column_fix/before_fix/`。
- 最终版 `--all --dry-run` 退出 0，仍为实际分支 420–540 run；原预注册 SHA-256 保持 `4999e24fcde3b13d8912855cc821c1af2ae0f4ac1a0532f11110a43e98f3c5ca`。
- 已清理 69 个本次临时冒烟目录，剩余 0；真实 `prereg/` 内没有生成测试访问台账、状态或源封存文件。
- 原 AI-TES 目录由专用安装器同步并逐文件核对；实际写入、目标 dry-run 与文件哈希以 [安装回执](round10_validation_2026-10-12/install_receipt.json) 为准。安装器拒绝覆盖接手后发生变化的原文件，保留既有运行目录；未执行 Git 操作。
- 本次 PI 列补充使用独立目标快照保护；补充安装回执为 `round10_validation_2026-10-12/install_pi_column_receipt.json`，初次安装回执保持不变。
- 正式 S1–S9、多种子统计判决、真实 B6/F11：本次未执行，留给组员按 runbook 正式启动。


### Linux 服务器换行校验修复（2026-10-12）

服务器 Case 1.csv 的 SHA-256 为 `3ecae761f836a468f5472b4e61a5097b909fd81fae79135159d14d08a8902e14`，逐字节等于原封存文件仅将 CRLF 转为 LF 后的版本。原始封存哈希 `24f9d57f50d5e7b4d9741029372e8af2434c31d64b93a7f046af89d0c59d9cf4` 保留。

- `round10_integrity.matches_frozen_sha256` 先核对原始字节；只对明确后缀的有效 UTF-8 文本接受 LF/CRLF 两种表示。它不解析或重排 CSV/JSON，不改数字、空格、列名、BOM 或末尾空行，也不写源文件。
- 接入预检与正式流水线的包/训练数据校验、F4/F1 划分与来源检查、预注册划分检查、独立 P0 源码校验。原预注册、311 文件来源清单与冻结 P0 源哈希清单保持原字节。
- checkpoint、NPZ、测试源封存、访问台账、报告指纹和续跑状态继续使用原始字节哈希。
- 新增检查 **26/26 PASS**：真实 311 训练文件的 LF 副本通过真实校验入口；增删行、数值、格式、空格、BOM 和列名修改仍被拒绝；全部 733 个固定输入通过 LF/CRLF 表示模拟，二进制输入保持精确核对；冻结 P0 源码 LF 副本通过且真实源码改动被拒绝。
- 修复后完整 preflight 退出 0，核对 **733 个包输入、311 个训练源文件、95 个 Python 源文件的 3.9 语法**；独立旧 P0 回归的预测与加权损失差仍为 0。`--all --dry-run` 退出 0，仍为 420–540 run。
- 证据在 `round10_validation_2026-10-12/newline_fix/`；检查日志保留此前追加内容，最新执行位于文件末尾。此次是本机换行模拟和预检，没有声称 Linux 整体运行已通过，没有重跑训练或打开真实 Test CSV。
- 原目录补丁由限定文件名单的安装器同步，回执为 `round10_validation_2026-10-12/install_newline_receipt.json`。服务器更新这版代码后重跑 preflight，无需转换 CSV、替换数据或重建封存哈希。

### A5 校准报告随包交付与外部路径修复（2026-10-12）

服务器 A5 的 19/20 失败来自缺少开发者 wiki 工作树内的校准报告。现将已有 Round 9 派生报告 `long_table.csv` 与 `pair_full70.csv` 连同来源和哈希纳入 `round10_resources/band_calibration/`，A5 默认读取包内文件，流水线固定该路径。A4 的物理缓存与 H9/旧调度器的冠军默认路径也统一到已有包内资源。没有重新标定阈值或读取原始 Test CSV。

- 新增可携带资源检查 8/8 PASS；A5 20/20 PASS；Python 3.9 语法检查覆盖 96 个源文件。
- 创建独立副本：738 个固定包输入和 311 个训练 CSV，全部文本转换为 LF；没有复制真实 Test 数据，仅保留数据加载器要求的空目录。
- 完整 `--preflight` 退出 0。审计钩子覆盖 43 个 Python 进程，禁止访问原 AI-TES、原工作包以及 wiki 开发工作树等路径，禁止路径访问尝试为 0。独立旧 P0 的 20 个验证文件预测差与加权损失差均为 0。
- 这次隔离验证实际运行在 Windows / Python 3.12.12；LF 文件模拟验证不等于服务器 Linux / Python 3.10 已运行通过。服务器更新完整包后仍需重跑预检。
- 证据保存在 `round10_validation_2026-10-12/portable_fix/`；补丁安装回执为 `round10_validation_2026-10-12/install_portable_receipt.json`。本次没有重跑全臂训练，既有 69/69 冒烟证据保持独立；正式 Round 10 与真实 B6/F11 尚未执行。

### 默认数据根目录传递修复（2026-10-12）

服务器未设置 DATA_ROOT 时，父进程来源校验使用 AI-TES/data，但 command() 曾给子进程注入空字符串；F4/F1 的 os.environ.get 默认值不会在空字符串时生效，导致从当前目录查找训练 CSV。此前隔离预检显式指定 DATA_ROOT，未覆盖这一启动方式。

- 流水线集中解析数据根目录：未设置和空值使用仓库 data，自定义目录保留并转绝对路径。检查、训练子进程和分析 manifest 使用相同路径；独立 F4/F1 入口也接受空值回退。
- 新增 5/5 回归检查，真实启动子进程覆盖未设置、空值、自定义目录和相对目录，核对训练环境、manifest 及显式覆盖。
- 使用标准 AI-TES/src/models/GRU 布局的独立 LF 副本，不设置 DATA_ROOT，完整 preflight 退出 0；739 个固定包输入、311 个训练源文件核验通过，48 个受审计 Python 进程没有访问被禁的原仓库或开发路径。独立旧 P0 在 20 个验证文件上的预测与加权损失差均为 0。
- F4/F1 在该整包预检中 14/14 PASS，并另外以显式空 DATA_ROOT 独立执行，仍为 14/14 PASS。Python 3.9 语法检查覆盖 97 个源文件；完整 --all --dry-run 退出 0，仍为 420–540 run。
- 证据见 round10_validation_2026-10-12/data_root_fix/，安装回执为 install_data_root_receipt.json。实际测试平台仍是 Windows / Python 3.12.12，不代表服务器 Linux / Python 3.10 已通过；此次没有读取真实 Test CSV、训练模型、修改原数据或原预注册。

### 同类路径与环境问题并行审计（2026-10-12）

针对服务器连续出现的可携带性问题，从资源入口、子进程环境及 Linux/Python 版本差异三路检查，确认并修复三类遗漏：

1. PHYSICS_CACHE_DIR、CHAMPION_RUN_DIR、ROUND9_RUNS_ROOT 显式空值未回退，且相对路径在子进程切换 cwd 后改变含义。现集中解析包内默认值，并在切换目录前变成绝对路径；训练、检查和分析 manifest 使用一致路径。
2. 冒烟训练已解析路径，但 verify_sealed_run 的 checkpoint 复验原样继承环境；P1 兼容入口也存在此问题。两处现统一环境路径传递，复验保留当前臂的数据及物理缓存覆盖值。
3. analysis_common 的独立分析入口遇到空 DATA_ROOT 会选当前目录。现与流水线一致，空值回退至仓库 data。

- 新增 check_resource_paths 12/12 PASS：从其他 cwd 真实启动子进程，覆盖未设置、空值、带空格绝对路径、相对路径；检查 manifest 一致性、独立分析默认路径，以及合成 sealed run 的复验调用和 P1 入口。复验夹具的 evaluator 被替换为合成指标，不是重新训练或真实模型实验。
- 在标准仓库布局的独立 LF 包中，从 GRU 目录外启动、DATA_ROOT 未设置、三个资源环境变量为空，完整 preflight 退出 0。740 个固定包输入及 311 个训练源文件核对通过；53 个受审计 Python 进程没有访问禁止的原仓库/wiki 开发路径。旧 P0 在 20 个验证文件上的预测差及加权损失差均为 0。
- 完整 dry-run 退出 0，仍为 420–540 run。98 个源文件通过 Python 3.9 语法检查；另对原 739 个固定输入及 311 个训练 CSV 逐级核对大小写，没有发现 Linux 文件名大小写不一致。没有发现新的主流程包外依赖，残留 wiki 工作树引用属于开发生成工具或文档。
- Test_70_cases 目录仍属于 runbook 所要求的数据布局；预检不会读取其中 CSV。本次隔离包只创建空目录，未复制真实 Test 数据。
- 实际执行平台为 Windows / Python 3.12.12。对 Python 3.10 与 PyTorch 2.9 的接口检查不等于服务器运行验证；服务器仍应预检和冒烟。没有修改训练/打分数学逻辑、原数据或原预注册，也没有重跑全臂训练。

证据见 round10_validation_2026-10-12/env_audit/，本次安装回执为 install_env_audit_receipt.json。

### Linux PID 0 锁误判修复（2026-10-12）

服务器 check_paper_report 在当前 PID 锁拦截通过后，以 pid=0 模拟进程退出并尝试恢复。POSIX 的 kill(0, 0) 查询当前进程组，旧 pid_alive 因而返回 True，抛出 A test report is already running；Windows 上该夹具却通过。上一轮审计注意到了 PID 0 语义差异，但漏查测试实际使用它，未及时消除该跨平台假设。

- pid_alive 先将 PID 转为整数，对 <=0 的非单进程标识直接返回 False，避免探测进程组。正数 PID 的存活、无进程和权限不足处理保留；报告窗口与流水线共用该函数。
- 报告检查新增显式 POSIX mock，覆盖零/负 PID 不调用系统探测、存活 PID、已退出 PID 和权限不足。恢复夹具改用本次实际创建并等待退出的子进程 PID，同时检查当前进程仍存活、并发锁仍拦截。
- check_paper_report 40/40 PASS。完整 LF 独立副本预检退出 0，覆盖目录外启动、默认数据目录和空资源变量；740 个固定输入、311 个训练源核验通过，54 个受审计 Python 进程无禁止路径访问。独立旧 P0 的 20 个验证文件预测和损失差均为 0，完整 --all --dry-run 仍为 420–540 run。
- 本机实际平台仍为 Windows/Python 3.12；新增 POSIX 分支检查是模拟，不冒充 Linux 全流程实跑。没有读取真实 Test CSV、运行正式报告窗口或训练模型；原数据与预注册未改。

证据在 round10_validation_2026-10-12/pid_fix/，安装回执 install_pid_fix_receipt.json。服务器应更新完整补丁（含封存清单）后重跑预检，不需要修改锁或跳过检查。
