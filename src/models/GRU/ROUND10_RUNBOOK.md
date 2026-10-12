# Round 10 — 组员运行说明

你负责在实验室 GPU 服务器上运行整包，并把产物交回。进入 `AI-TES/src/models/GRU/` 后，按下面三条命令执行。阶段、条件臂、统计判决和最终配方都由流水线自动处理。

## 1. 安装环境

建议新建 Python 3.11 环境，使用支持 CUDA 的 NVIDIA GPU：

```bash
conda create -n tes-round10 python=3.11 -y
conda activate tes-round10
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r round10_requirements.txt
```

PyTorch 安装命令来自 [官方 2.8.0 / CUDA 12.6 安装说明](https://pytorch.org/get-started/previous-versions/#v280)。如果服务器已有管理员维护的 CUDA/PyTorch 环境，可先用该环境运行预检和冒烟。磁盘至少留出 30 GiB，正式实验建议预留更多空间。

本机已在 Python 3.12.12、PyTorch 2.8.0+cu126、RTX 4090 上完成全臂冒烟，代码通过 Python 3.9 语法检查。Python 3.10/3.11 的实际运行仍需下面的服务器冒烟确认。

## 2. 放好包和数据

保留整个 GRU 文件夹，尤其是 `tes_gru/`、`splits/`、`prereg/`、`round10_checks/` 和 `round10_resources/`。资源包已包含训练/验证物理缓存、冻结常数、旧代码回归副本和 Round 9 的分析 checkpoint。

数据目录保持原结构：

```text
AI-TES/
  data/
    Latest Database (Use this for training)/   # 311 个源 CSV
    Test_70_cases/                            # 70 个测试 CSV
  src/models/GRU/                             # 当前运行目录
```

默认从这个 `AI-TES/data/` 读取。如果数据放在另一块盘，只设置数据根目录，保留它下面两层文件夹的名称：

```bash
# Linux / bash：替换为服务器上的真实路径
export DATA_ROOT="/absolute/path/to/AI-TES/data"
```

```powershell
# Windows / PowerShell：替换为本机真实路径
$env:DATA_ROOT = 'D:\AI-TES\data'
```

使用新终端，只激活运行环境和设置 `DATA_ROOT` 即可。包会核对 311 个训练源文件与固定输入的哈希；不一致时停止并指出文件。不要重新生成哈希、修改数据、划分、预注册或模型参数。测试评分只在自动到达 B6 和 F11 时开放，并记录访问及中断恢复。

Windows 与 Linux 的 Git 换行转换（CRLF/LF）已兼容：固定文本输入只允许这两种换行表示，数值、列名、空格和其它内容变化仍会停止。无需为服务器转换原始 CSV 或换一批数据。若旧版预检因此报错，更新整包后重跑预检；checkpoint、测试源封存和运行记录仍按原始字节严格核验。

## 3. 三条命令

在 `AI-TES/src/models/GRU/` 中依次执行：

```bash
python round10_pipeline.py --preflight
python round10_pipeline.py --smoke --parallel 3
python round10_pipeline.py --all --parallel 3
```

1. **预检**：检查环境、GPU、磁盘、输入哈希、全部独立检查和旧代码验证回归，不训练。
2. **冒烟**：69 个臂各跑种子 7、2 轮，检查封存产物、常数 sidecar 和 checkpoint 验证重打分；成功后自动清理本次试跑目录。等到 `SMOKE PASS 69/69` 再执行正式命令。
3. **正式运行**：从 S0 到 S9 自动运行。无条件分支共 420 个训练 run；条件臂和确认全部触发时最多 540 个。实际数量由验证侧规则决定。不同服务器的用时不同，预留一到两天并保持终端会话存活。

如果共享 GPU 上三任务并行导致显存不足，停止后把最后一条命令改为 `--parallel 1` 重跑；不改其它参数。每个臂的种子可并行，依赖阶段按顺序执行。

## 4. 进度和中断恢复

终端会打印当前阶段、臂、种子与完成数。详细训练日志在 `sweep_logs/pipeline-<arm>-seed<seed>.log`；分析脚本日志在对应的 `runs/_round10_results/<stage>/` 下。

- `runs/_round10_results/pipeline_state.json`：已完成阶段、run 和冻结决定。
- `runs/_round10_results/decision_log.md`：自动判决及使用的数字、区间和规则。
- `runs/_round10_results/<stage>/`：该阶段的表、图、清单和分析日志。
- `runs/_round10_results/S1b/P1_arm_levels.csv` 和同名 `.md`：给 PI 的逐通道结果，含“换材料时每种输出形式要重拟合什么”列。C16 的验证表及最终 F11 C16 表也保留此列；`physics_input_refit` 单独说明物理输入的更新要求。同目录 `material_refit_guide.csv/.md` 汇总五种输出形式的预注册说明。
- `runs/_round10_smoke/smoke_results.csv`：服务器冒烟的逐臂检查记录。

需要中断时按一次 Ctrl+C，等进程结束。保持包、数据和 `runs/` 原样，再执行同一条正式命令即可续跑。流水线会检查已完成产物，并从未完成的 run 或阶段继续；B6/F11 的恢复使用同一批冻结输入。

报错后保留完整错误和对应日志。环境或磁盘问题处理后重跑同一命令；哈希、封存、checkpoint 或统计输入不一致时，把日志交给负责人处理。不要删除状态文件、锁文件、失败的正式 run 或测试访问记录。

## 5. 完成后交回什么

以终端完成 S9、`pipeline_state.json` 中全部阶段完成为准。交回：

- **整个 `runs/` 文件夹**：正式 checkpoint、`run_config.json`、常数 sidecar、验证预测、训练历史，以及 `_round10_results/` 和冒烟检查记录。
- **整个 `sweep_logs/` 文件夹**：训练控制台日志。
- **整个 `prereg/` 文件夹**：固定规则、包哈希，以及本次生成的 `test_set_access.log`、`test_access_state.json`、`test_source_seal.json`。
- **环境与问题记录**：预检已保存环境信息；另附实际启动命令和任何异常说明。

最终 F11 的链表、物理/线性底线、C16 测试点、误差预算、地板和集成结果位于 `runs/_round10_results/S9/`。一起交回完整文件夹即可，组员不需要挑选赢家或手动补评分。
