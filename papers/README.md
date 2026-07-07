# Local Research Papers

Place PDF or TXT documents in this folder before ingestion.

The actual research papers used during local testing are not committed because they may be third-party copyrighted material. Keeping this folder empty in Git also keeps the repository safe if it is later made public.

## Paper Titles Used During Local Testing

These are the paper titles used locally. Reviewers can search these titles and download them from the original publisher or source pages where available:

- A novel CNN architecture for accurate early detection and classification of Alzheimer's disease using MRI data
- Accurate Detection of Alzheimer's Disease Using Lightweight Deep Learning Model on MRI Data
- Advanced interpretable diagnosis of Alzheimer's disease using SECNN-RF framework with explainable AI
- Advancements in deep learning for early diagnosis of Alzheimer's disease using multimodal neuroimaging challenges and future directions
- Alzheimer's Disease Detection Through Whole-Brain 3D-CNN MRI
- Classifying and diagnosing Alzheimer's disease with deep learning using 6735 brain MRI Images
- Deep Multi-Branch CNN Architecture for Early Alzheimer's Detection from Brain MRIs
- Deep learning techniques for Alzheimer's disease detection in 3D imaging A systematic review
- Intelligent Diagnosis of Alzheimer's Disease Based on Machine Learning
- MRI-Driven Alzheimer's Disease Diagnosis Using Deep Network Fusion and Optimal Selection of Feature

Recommended local flow:

```bat
copy path\to\papers\*.pdf papers\
python ingest.py --reset
```
