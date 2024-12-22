This is the replicate package for AVR study.

**Contact**: Feel free to contact Junji Yu (junjiyu@tju.edu.cn) and Honglin Shu (shu.honglin.167@s.kyushu-u.ac.jp) if you have any further questions about the code.

# 1. How to reporduce
For TFix, VRepair, VulRepair, CodeBERT, GraphCodeBERT, CodeT5, CodeReviewer, PolyCoder, CodeGen, GPT2-CSRC, and VQM techniques, please use the following commands:
```sh
cd VQM
pip install -r requirements.txt
cd VQM/transformers
pip install .
```
For VulMaster, please use the following commands:
```sh
cd VulMaster
pip install -r requirement.txt
```
For LLMs (CodeLlama, Llama3, DeepSeek, ChatGPT-3.5-turbo, and ChatGPT-4o), please use the following commands:
```sh
pip install -r requirement.txt
```

# 2. Dataset
We construct the Automated Vulnerability Repair dataset based on the [REEF](https://github.com/ASE-REEF/REEF-data).

Due to file size limitation, dataset for VulMaster can be download from [Zenodo](https://zenodo.org/records/14542701)

# 3. Models
All the models' checkpoints in our experiments can be downloaded from [Zenodo](https://zenodo.org/records/14542701).