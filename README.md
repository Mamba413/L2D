Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text
----

This repository contains the implementation of [Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text](https://openreview.net/forum?id=2ZUPeEM3FH), accepted by ICLR 2026. Our method provides a geometric projection viewpoint on the effectiveness of rewritten-based detectors and foster the power of such a method via distance learning. We build upon and extend code from [AdaDetectGPT](https://github.com/Mamba413/AdaDetectGPT).

![](figure/workflow.png)

## TODO

- pretrain a check-point, and save it to huggingface
- use other to rewrite

## 🛠️ Installation

### Requirements
- Python 3.10.8
- PyTorch 2.7.0
- CUDA-compatible GPU (experiments conducted on H20-NVLink with 96GB memory)

### Setup
```bash
./setup.sh
```

## Usage


### Experiment on one dataset with pretrained checkpoint

Recommended for off-the-shelf usage. 

```sh
python scripts/detect_l2d.py --from_pretrain mamba413/L2D --eval_dataset example/demo --output_file example/demo --eval_only 
```

Pretreined details:
- rewritten model: `google/gemma-2-9b-it`
- training dataset texts from GPT-4o, Gemini-2.5, and Claude-3.5. 
- The pre-trained LoRA adapter in huggingface:


### Experiment on your data 

Recommended in the following cases: 

- to improve performance on your specific domain
- fairly compare with other training based methods



## 🎁 Additional Resources

The `scripts/` directory contains implementations of various LLM detection methods from the literature. These implementations are modified from their official versions or the repo of [AdaDetectGPT](https://github.com/Mamba413/AdaDetectGPT) to provide:

- Consistent input/output formats
- Simplified method comparison

The provided methods are summarized below.

| Method                      | Script File            | Paper/Website                                                   |
| --------------------------- | ---------------------- | --------------------------------------------------------------- |
| **AdaDetectGPT**            | `detect_gpt_ada.py`    | [arXiv:2510.01268](https://arxiv.org/abs/2510.01268)            |
| **BARTScore**               | `detect_bartscore.py`  | [EMNLP-main.463](https://aclanthology.org/2023.emnlp-main.463/) |
| **Binoculars**              | `detect_binoculars.py` | [arXiv:2401.12070](https://arxiv.org/abs/2401.12070)            |
| **Fast-DetectGPT**          | `detect_gpt_fast.py`   | [arXiv:2310.05130](https://arxiv.org/abs/2310.05130)            |
| **GLTR**                    | `detect_gltr.py`       | [arXiv:1906.04043](https://arxiv.org/abs/1906.04043)            |
| **IDE**                     | `detect_ide.py`        | [arXiv:2306.05540](https://arxiv.org/abs/2306.04723)            |
| **ImBD**                    | `detect_ImBD.py`       | [arXiv:2412.10432](https://arxiv.org/abs/2412.10432)            |
| **Likelihood**              | `detect_likelihood.py` | [arXiv:2306.05540](TOADD)                                       |
| **LRR**                     | `detect_lrr.py`        | [arXiv:2306.05540](https://arxiv.org/abs/2306.05540)            |
| **RADAR**                   | `detect_radar.py`      | [arXiv:2307.03838](https://arxiv.org/abs/2307.03838)            |
| **RADIAR**                  | `detect_radiar.py`     | [arXiv:2307.03838](https://arxiv.org/abs/2401.12970)            |
| **RoBERTa OpenAI Detector** | `detect_roberta.py`    | [arXiv:1908.09203](https://arxiv.org/abs/1908.09203)            |

### Reproduce guidance

- `exp_diverse.sh`: generate Table 1, Tables B1-B4
- `exp_prompt.sh`: generate Table 2
- `exp_attack`: Figure 4

After running the above code, please use `python script/report_results.py` to see the results. Use either the `report_black_prompt_results` or the `report_diverse_results` functions.

### 📖 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{zhou2026learn,
  title={Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text},
  author={Hongyi Zhou and Jin Zhu and Erhan Xu and Kai Ye and Ying Yang and Chengchun Shi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  abbr={ICLR},
  year={2026},
}
```

If you have any question, please feel free to open an [issue](https://github.com/Mamba413/L2D/issues).
