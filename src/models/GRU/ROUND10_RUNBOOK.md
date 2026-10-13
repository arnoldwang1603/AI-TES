# Round 10 v4 — 组员运行说明

这份说明用于实验室 GPU 服务器。你负责更新完整包、通过预检和全臂冒烟，再启动一次全量流水线，最后把完整产物交回。34 个基础科学实验臂全部固定跑 20 个种子；程序自动完成后续确认和报告，不需要中途人工挑臂或决定补跑。

**版本状态：本机 v4 完整预检与 `--all --dry-run` 已通过；全臂冒烟按负责人要求放到实验机器上跑。** 本版计划覆盖 96 条冒烟路径，必须在服务器看到 `SMOKE PASS 96/96` 后再启动全量。v3 的 `SMOKE PASS 69/69` 属于旧版证据。详见 `ROUND10_V4_VALIDATION_2026-10-12.md`。

## 1. 停止旧任务，更新完整包

如果旧终端还在跑 v3，按一次 `Ctrl+C`，等流水线和它启动的训练进程退出。保留原有 `runs/`、状态、日志和 `prereg/` 中的测试访问记录。

先等负责人在自己的电脑上提交并推送 v4，再在服务器执行：

```bash
cd ~/AI-TES
git status --short
git pull --ff-only
cd src/models/GRU
```

`git pull --ff-only` 是更新命令：下载远端提交并直接前移当前分支；遇到本地改动冲突或分支分叉会停止。更新报错时保留输出交给负责人处理，完成更新后再往下走。所有提交、推送由负责人自行操作。

沿用此前已经通过服务器 CUDA 检查和冒烟的 Python 环境。新服务器需要可用的 CUDA/PyTorch，以及 `round10_requirements.txt` 中的依赖；由维护该服务器环境的人安装。预检会检查 GPU 和剩余磁盘，至少需要 30 GiB；完整实验应预留足够保存上千个训练目录的空间。

完整包应包含 `tes_gru/`、`splits/`、`prereg/`、`round10_checks/`、`round10_resources/`。服务器不需要另装 wiki 仓库或复制 `.claude/worktrees`。

## 2. 核对数据位置，登记 v4 重启

默认数据目录如下，保留原来的 CSV：

```text
AI-TES/
  data/
    Latest Database (Use this for training)/
    Test_70_cases/
  src/models/GRU/
```

默认读取 `AI-TES/data/`。只在数据另放一块盘时设置：

```bash
export DATA_ROOT="/absolute/path/to/AI-TES/data"
```

新终端只需要激活运行环境，以及按需设置 `DATA_ROOT`。物理缓存、历史校准报告和旧模型回归资源默认使用包内版本。

在 GRU 目录执行：

```bash
python round10_pipeline.py --restart-v4
```

这个命令审计旧流水线状态及测试访问记录，保存 `runs/_round10_v4_results/migration_v3_audit.json`。它保留旧产物，不训练、不评分、不自动复用旧 run。若旧版已经进入真实 B6/F11 测试访问，它会拒绝重启；把完整输出交给负责人，不要删除记录绕过检查。重复执行同一迁移命令应安全且幂等。

v4 正式 run 使用 `r10v4_` 前缀，结果写到 `runs/_round10_v4_results/`。旧版的 `r10_` run 和 `runs/_round10_results/` 留在原处；旧版已完成的少量种子也不并入 v4。

## 3. 预检、查看计划、全臂冒烟

按顺序执行，前一条成功后再执行下一条：

```bash
python round10_pipeline.py --preflight
python round10_pipeline.py --all --dry-run
python round10_pipeline.py --smoke --parallel 3
```

- **预检**检查环境、GPU、磁盘、固定输入哈希、独立检查及验证集上的旧代码回归，不训练。
- **dry-run**列出所有可能的 run 名和执行数量，不训练。静态计划为 **830–1090 个正式训练 run**；所有可能条件名称的并集可以更大，不能直接把打印行数当作一次运行数量。
- **冒烟**覆盖基础臂和后续确认、F4、F1、最终配方路径，各跑一个种子、两个 epoch，检查封存产物、常数 sidecar 及 checkpoint 验证重打分。本版应输出 `SMOKE PASS 96/96`。试跑成功后自动清理本次冒烟训练目录，保留 `runs/_round10_v4_smoke/` 中的检查记录。

输入哈希不一致时，程序会指出文件。将报错交给负责人核对数据版本；不要重新生成封存哈希或自行更换数据。固定文本输入兼容 CRLF/LF 换行，其他内容变化仍会停止。

## 4. 启动全量

```bash
python round10_pipeline.py --all --parallel 3
```

这是正式运行的唯一启动命令。程序会依次完成：

| 阶段 | 自动执行的内容 |
|---|---|
| S0 / S1 / S1b | 预检；标签与控制实验；预注册 B6 报告、桥接和探针 |
| S2 / S3 / S4 / S5 | 锚尺度、损失、前视、输出形式、比例、全部编码器及 G7、物理参照实验 |
| S5b / S6 | 分析全部比较；按固定规则选取方案，在 h160 容量下确认 |
| S7 | F4 两个新划分的配对复验 |
| S7b | 根据 F4 判决冻结最终配方，再做 F1 留一家族：direct 与最终配方在 h128 下各跑十家族、每家族十种子；配方相同时只跑一组 |
| S8 / S9 | F9/F10 的二十个新种子；F11 最终报告、曲线、误差预算和集成 |

所有基础科学臂固定 20 seeds，G7c/G7d 也固定执行。只有机器桥接 RETRAIN 保留 10。C17 入选者各 20；F4 每个新划分、每个配方 10；F1 每个家族、每个配方 10；F9/F10 使用同组 20 个新种子。

F4 的原划分 seed 42 复用主实验完整的 20 个训练种子；新增划分 seed 43、44 各跑 10 个训练种子。每个划分内，direct 和候选方案使用相同种子配对、各自判决；三个划分不混成一个统计样本池。复用原划分不增加上表的正式训练数量。

程序仍会按照预先写好的规则选择最终配方，但组员无需看中间结果再决定加跑。即使某种输出形式没有入选，它的完整基础比较和阴性、不确定结果也会保留。

F1 每折分别使用训练集、内部验证集和整类留出集：拟合常数只用该折训练数据，早停只看内部验证集，留出家族只在训练结束后评分。F1 结果不会重新改变已经冻结的配方。这是已选定配方的留一家族评估；前面的主验证/F4 接触过各家族数据，因此不能表述为整个研究流程从未见过该家族。物理臂沿用固定 nominal 物理参考，残差归一化等数据统计量才在各折训练集重新拟合。

正式运行可能持续较久，请保持服务器会话可用；实际用时以服务器进度为准，两个 epoch 的冒烟时间不能直接当作正式时长。如果三任务并行显存不足，停止并等待子进程退出后，用 `--parallel 1` 重跑同一正式命令。

## 5. 进度、报错和恢复

终端显示当前阶段、臂、种子与完成数。主要产物：

- `runs/_round10_v4_results/pipeline_state.json`：阶段状态、run 清单和冻结决定。
- `runs/_round10_v4_results/decision_log.md`：自动判决、区间及依据。
- `runs/_round10_v4_results/<stage>/`：阶段表、图、清单和分析日志。
- `sweep_logs/pipeline-v4-<arm>-seed<seed>.log`：每个训练进程的详细日志。
- `runs/_round10_v4_results/S9/`：最终 F11 报告及附表。
- `runs/_round10_v4_smoke/`：本版逐臂冒烟检查记录。

P1、各阶段结果和 C16/F11 表含给 PI 的逐通道“换材料时要重拟合什么”说明；它是方法适用条件说明，不是已经完成换材料实验的结果。

要暂停时按一次 `Ctrl+C`，等待所有子进程退出。保持代码、数据和运行目录不变，之后直接恢复：

```bash
cd ~/AI-TES/src/models/GRU
python round10_pipeline.py --all --parallel 3
```

程序检查已完成产物，从未完成的 run 或阶段继续。恢复已有 v4 时不需要重新执行 `--restart-v4`，也不要另起第二个全量进程。B6/F11 的中断恢复使用同一批冻结输入和访问账本。

报错时记录终端最后的完整错误及程序指出的日志。环境或磁盘问题处理后重跑同一命令；遇到哈希、封存、checkpoint、测试访问或统计输入不一致，交给负责人处理。保留状态文件、锁文件、失败 run 和访问记录。

### 特定故障恢复：S1 后出现 `Index contains duplicate entries, cannot reshape`

这是验证结果汇总把同名、不同目录的文件当成同一个 case 的程序错误。负责人发布的 2026-10-12 case-ID 修复同时更新三个读取入口、回归检查和包封存，并提供专用恢复脚本。不要自行改哈希或删除状态；已完成的训练可以保留。

仅当 S0/S1 完成、S1b 因上述错误失败且尚未进入 B6 时，更新后执行：

```bash
cd ~/AI-TES
git pull --ff-only
cd src/models/GRU
python round10_case_id_hotfix.py --apply
python round10_pipeline.py --preflight
python round10_pipeline.py --all --parallel 3
```

每条成功后再运行下一条。恢复脚本核对旧/新封存、原协议、数据、阶段状态及已完成 S1 产物，备份原状态后只迁移包身份；拒绝不符合这次故障条件的状态。随后 S0/S1 显示 `already complete`，从失败的 S1b 汇总继续。不要运行 `--restart-v4`，也不必为这个汇总修复重训或重新做全臂冒烟。若脚本拒绝恢复，保留输出交给负责人，不手改状态绕过。

这次修复不改变模型、权重、验证划分、训练参数、种子、统计阈值或测试访问规则。修复说明和审核范围见 `ROUND10_CASE_ID_HOTFIX_2026-10-12.md`。专用脚本不带 `--apply` 时仅检查，不迁移状态。

如果先前版本的恢复脚本报 `S1b validation paths differ from the frozen validation split`，它可能是把带训练目录前缀的导出路径与不带前缀的 split 键直接比较。负责人已修正为训练程序实际导出的路径格式；更新完整补丁后重新执行上述恢复命令即可。该拒绝发生在状态迁移和备份写入前，勿为此删除状态或重训。更新后若仍不匹配，错误会列出具体 run 以及缺失/多出的路径，交给负责人核对。

### 特定故障恢复：A7 报 scaler 缺少 `feature_names_in_`

如果 S1 已完成，且 `a7_bridge.py.log` 的异常为 `AttributeError: 'MinMaxScaler' object has no attribute 'feature_names_in_'`，使用负责人发布的 A7 修复。该属性是可选列名信息；修复按训练/推断已固定的五列顺序读取 scale，不改训练、模型、统计阈值或验证数据。

这次故障发生在 B6 完成之后，使用下面的 A7 专用恢复命令，不使用前面的 case-ID 恢复命令：

```bash
cd ~/AI-TES
git pull --ff-only
cd src/models/GRU
python round10_a7_scaler_hotfix.py --apply
python round10_pipeline.py --preflight
python round10_pipeline.py --all --parallel 3
```

每条成功再执行下一条。恢复脚本核对原包、170 个 S1 run、40 个 B6 checkpoint 指纹、已完成 B6 报告及访问记录，备份原状态和 A7 失败日志，只迁移包身份。B6 报告内容只作哈希验证；不重新读取测试数据。继续运行时 S0/S1 显示 `already complete`，B6 也应显示 `already complete`，然后继续 G1/A7。保留原 run、结果和访问账本，无需重训或重新跑全臂冒烟。若恢复脚本拒绝，保留具体报错交给负责人。

本次合并更新还修正后续 P4 的 T0 稳定性判定：扫描出现无效值或缺失采样行时不能被判为通过；种子数、0.5°C 阈值和训练配置不变。适用于服务器仍停在上述原始 A7 故障、尚未应用第一版 A7 修复的情况。说明和验证范围见 `ROUND10_A7_SCALER_HOTFIX_2026-10-12.md`。

## 6. 完成后交回

以 S9 完成、`pipeline_state.json` 中包括 S7b 在内的全部阶段均为完成状态为准。交回这些完整目录：

1. **`runs/`**：正式 checkpoint、`run_config.json`、常数 sidecar、验证预测、训练历史、v4 结果、迁移回执、冒烟记录，以及保留的旧版产物。
2. **`sweep_logs/`**：所有训练日志。
3. **`prereg/`**：v3 原协议、v4 协议与修订说明、固定输入哈希、测试访问账本和源文件封存记录。
4. **运行说明**：实际启动命令、环境信息、出现过的异常以及如何恢复。

组员交回整包即可，无需手动选赢家或重新运行测试评分。负责人据此确认这一轮完成，再交接 convection；本包本身不自动开始 convection 训练。

## 7. 临时结果需要处理吗

通常不需要手动清理。`--all --dry-run` 只打印计划，不生成训练结果。成功的全臂冒烟会删除它创建的 `r10v4smoke_*_ep2_*` 训练目录，保留 `runs/_round10_v4_smoke/` 中的检查记录；后者体积小，交接时一起带上。冒烟被中断后，重新执行完整冒烟会核对并清理对应的旧试跑目录，再重新训练。

保留 `runs/_round10_v4_results/`、`r10v4_` 正式训练目录、旧版正式 run、`sweep_logs/` 和 `prereg/` 中的测试访问记录。它们用于恢复、审计和论文结果，不能当作临时文件清掉。`scratch/` 是临时目录，通常无需处理；确需腾空间时等所有本包进程结束后再处理。

本机开发时创建的隔离验证副本由发布负责人清理；服务器不需要复制这些隔离目录或 wiki 仓库。
