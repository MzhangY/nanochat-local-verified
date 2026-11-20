# nanochat：无模型/语料下载的微型 CPU 验证

这是 karpathy/nanochat 的学习验证 Fork，不是原创大模型；保留 Andrej Karpathy、上游贡献者及 MIT LICENSE。固定上游提交 `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`。

## 真实结果（2026-11-20，Windows / Python 3.13.5 / torch 2.9.1+cpu）

- 分词器、引擎、SDPA 回退、内存任务数据的 4 个测试文件：**30 通过、0 失败、10 跳过**。10 项要求 Flash Attention 3 硬件，未伪装为通过。
- 首轮缺 NumPy 收集失败，补齐独立环境后复跑；首轮证据保留。
- 上游真实 GPT：2 层、110646 参数、上下文 32，CPU 两线程、80 步。
- 同一微型合成重复 token 序列 loss：**3.47307968 → 0.00041325**，训练 token 准确率 100%。
- 保存/加载权重后 logits 完全一致；少量内置中英/Emoji 训练分词器并往返通过。

**只证明能拟合微型合成序列，不代表聊天质量、泛化能力或完整大模型训练成功。** 用标准 `torch.optim.AdamW` 的 eager CPU 路线，未验证上游 Muon/分布式/torch.compile/GPU 优化器或完整预训练、SFT、评测。

未运行模型生成代码执行的 `test_execution.py`，未下载外部模型/语料/评测集，未执行整段 `runcpu.sh` / `speedrun.sh`，未调用付费 API。

## 独立环境复现（不动系统 PyTorch）

```powershell
python -m venv .venv-local-validation
.\.venv-local-validation\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-local-validation\Scripts\python.exe -m pip install -r validation/requirements-tests.txt
.\validation\verify.ps1 -PythonPath .\.venv-local-validation\Scripts\python.exe
```

只跑 80 步合成训练：

```powershell
.\validation\verify.ps1 -PythonPath .\.venv-local-validation\Scripts\python.exe -SmokeOnly
```

本机 rustbpe 0.1.0 有 Python 3.13 Windows wheel，无需 Rust/Cargo。只安装此 CPU 验证范围必需依赖，不等于满足上游全部路线；精确环境见 `validation/evidence/nanochat-freeze.txt`。

## 证据和边界

`validation/synthetic-smoke.json`、`validation/results.json`、`validation/evidence/` 保留训练和测试/跳过证据。未修改上游运行时代码，未创建上游 PR。临时 home/config 和 offline/socket 限制不是 OS 安全沙箱或完整安全审计。
