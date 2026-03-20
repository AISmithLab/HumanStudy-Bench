<div align="center">
  <img src="docs/img/new-HS-bench_logo.png" alt="HumanStudy-Bench Logo" width="300">

  <h1>HumanStudy-Bench: Towards AI Agent Design for Participant Simulation</h1>

  [![Leaderboard & Results](https://img.shields.io/badge/Leaderboard_%26_Results-hs--bench.clawder.ai-orange)](https://www.hs-bench.clawder.ai)
  [![Read the Paper](https://img.shields.io/badge/Paper-arXiv%3A2602.00685-b31b1b)](https://arxiv.org/abs/2602.00685)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Community Edition](https://img.shields.io/badge/Community_Edition-Contribute_New_Studies-brightgreen)](https://github.com/HumanStudy-Hub/HumanStudy-Bench)

</div>

---

> **New contributions are now accepted at the [Community Edition](https://github.com/HumanStudy-Hub/HumanStudy-Bench).** Please fork and submit PRs there.

> LLMs are increasingly used to simulate human participants in social science research, but existing evaluations conflate base model capabilities with agent design choices, making it unclear whether results reflect the model or the configuration.

## Overview

HumanStudy-Bench treats participant simulation as an *agent design problem* and provides a **standardized testbed** — combining an **Execution Engine** that reconstructs full experimental protocols from published studies and a **Benchmark** with standardized evaluation metrics — for *replaying human-subject experiments end-to-end* with alignment evaluation at the level of scientific inference.

This repository contains the **original 12 foundational studies** and benchmark code as described in the paper. For the paper's exact results, use tag `v1.0.0`.

## Community Edition

**Want to contribute a study or use the latest community-expanded benchmark?**

Head to the **[Community Edition](https://github.com/HumanStudy-Hub/HumanStudy-Bench)** — an open, community-driven repo where anyone can submit new studies via PR. The community version includes the same 12 foundational studies plus all community contributions.

[![Contribute on HumanStudy-Hub](https://img.shields.io/badge/Contribute-HumanStudy--Hub-brightgreen)](https://github.com/HumanStudy-Hub/HumanStudy-Bench)

## Reproduction (Paper Results)

To reproduce the exact benchmark and results reported in the paper:

```bash
git checkout v1.0.0
```

## Citation & Hugging Face

If you use HumanStudy-Bench, please cite:

```bibtex
@misc{liu2026humanstudybenchaiagentdesign,
      title={HumanStudy-Bench: Towards AI Agent Design for Participant Simulation},
      author={Xuan Liu and Haoyang Shang and Zizhang Liu and Xinyan Liu and Yunze Xiao and Yiwen Tu and Haojian Jin},
      year={2026},
      eprint={2602.00685},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.00685},
}
```

**Hugging Face:** Benchmark and resources are available on the [Hugging Face Hub](https://huggingface.co/) — `fuyyckwhy/HS-Bench-results`.

## License

MIT License. See [LICENSE](LICENSE) for details.
