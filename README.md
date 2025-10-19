# SPINE: Segmentation-guided Processing and Integration of multimodal spinal MRI for Natural-language Enhanced report generation

We propose **SPINE**, a segmentation-guided, multimodal framework for generating spinal MRI reports using 3D vision–language models. The framework integrates **T1- and T2-weighted MRI** with **anatomical segmentation** to enhance spatial and contextual understanding of spinal structures.

---

## About the project

SPINE leverages multimodal input (T1, T2, and segmentation) to generate radiology reports for spinal MRI. Experiments were conducted on two public datasets: **515 axial** and **190 sagittal** cases. Three input configurations were explored, with **T1+T2+segmentation (V3)** achieving the best performance on the axial dataset.  

For sagittal data, structured gradings were converted into narrative reports using GPT-4o and Grok-3, enabling controlled language generation. Structured supervision improved semantic consistency and accuracy. These results emphasize the value of combining anatomical priors with structured language for reliable spinal MRI reporting.

---

## Demo

Follow these steps to run the demo:

### 1. Run Initialization Script
```bash
sh modify_m3d.sh
```
### 2. Generate Reports
```bash
python3 -u demo_csv.py \
--model_name_or_path PATH_TO_MERGED_WEIGHTS \
--data_root PATH_TO_IMAGES \
--amos_validation_cap_data_path PATH_TO_CSV_FILE \
--output_dir PATH_TO_OUTPUT_DIR
```
Replace `PATH_TO_MERGED_WEIGHTS` with your fine-tuned model path.  

---
## Experimental Setup

- **Datasets**
  - Axial dataset: 515 patients  
  - Sagittal dataset: 190 patients  
  - Results were averaged across:
    - **10 folds** for axial dataset
    - **5 folds** for sagittal dataset
  - 📎 [Axial dataset link](https://doi.org/10.17632/zbf6b4pttk.2)  
  - 📎 [Sagittal dataset link](https://zenodo.org/records/10159290)

- **Modalities**: T1, T2, segmentation  
- **Language Supervision**: Structured vs. natural language  
- **Evaluation**: Report generation quality and consistency

---

## Acknowledgements

This project builds upon open source projects: 

- [M3D](https://github.com/BAAI-DCAI/M3D) — Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models
- [LLaVA](https://github.com/haotian-liu/LLaVA) — Large Language and Vision Assistant

We thank the original authors for making their work and weights publicly available.

