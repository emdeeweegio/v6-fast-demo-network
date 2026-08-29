
# 20kLogRegChallenge

**Python package name:** The importable module is `logreg_challenge_20k` (folder `logreg_challenge_20k/`), not `20kLogRegChallenge`, because Python identifiers cannot start with a digit. MockClient and Docker use `logreg_challenge_20k`; the project folder and Docker image can still be named `20kLogRegChallenge`.

This algorithm is designed to be run with the [vantage6](https://vantage6.ai)
infrastructure for distributed analysis and learning. See the "20kChallenge
logistic regression (BEACH-schema)" section in the repo root
[README.md](../../README.md) for how to generate data, build the image, and
run it against this harness.

### Basis: 20kChallenge / Personal Health Train

The distributed logistic regression (ADMM) approach in this project is **conceptually based on** the work in:

- **GitHub:** [RadiationOncologyOntology/20kChallenge](https://github.com/RadiationOncologyOntology/20kChallenge/)
- **Manuscript:** *Distributed learning on 20 000+ lung cancer patients – The Personal Health Train* (Deist et al.; see authors on the [20kChallenge README](https://github.com/RadiationOncologyOntology/20kChallenge/)).

That repository implements training/validation of logistic regression and summary statistics in a **master/site** architecture (MATLAB + Varian Learning Portal). This vantage6 algorithm is a Python adaptation for a similar distributed setting. For data format and RDF triples, see the wiki linked from the upstream repo.