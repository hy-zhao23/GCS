# Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution

[[Paper](https://openreview.net/pdf?id=CvttyK4XzV)] | [[Project Page](https://hy-zhao23.github.io/projects/gcs/)]

## Environment Installation

```bash
micromamba env create -f environment.yml
```

**NOTES:** Experimental variables are set up in `.env` and `utils/config.py`. For optimal efficiency, we recommend using a GPU with at least 80GB of memory. We highly recommend using parallel programming when handling training tasks.

## Data Collection
1. Set up concepts required in `preprocess/concept_gen.json`
2. Set up promp for each concept in `preprocess/prompts.json`
3. Run `preprocess/generate_text.py` to generate text data for each concept

## Latent Representation Extraction

Run `scripts/get_hs_activations.py` to extract latent representations from the model.

## Probing Classifier Training

Run `scripts/get_probing_classifier.py` to train probing classifiers for each concept.

## GCS Computation

Run `scripts/get_sampled_classifier.py` to compute the GCS for each concept.

## Evaluation

Run `scripts/evaluation.py` to evaluate the performance of the probing classifier.

- Cosine similarity between trained probing classifiers and the sampled classifiers
- Accuracy of the probing classifier
- Cosine similarity among concept vectors
- PCA visualization of concept vectors

## Emotion Steering

Run `scripts/emotion_steering.py` to steer the emotion of the model.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{
zhao2025beyond,
title={Beyond Single Concept Vector: Modeling Concept Subspace in {LLM}s with Gaussian Distribution},
author={Haiyan Zhao and Heng Zhao and Bo Shen and Ali Payani and Fan Yang and Mengnan Du},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=CvttyK4XzV}
}
```




