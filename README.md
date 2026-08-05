![ML from Scratch](https://raw.githubusercontent.com/ather-ops/Supervised-ML-from-scratch/main/Assets/ml.png)

# Supervised Machine Learning from Scratch

Building every ML algorithm from the ground up — no scikit-learn, no TensorFlow, no abstractions. Every gradient, every loss function, every normalization step is written by hand using only Python, NumPy, Pandas, and Matplotlib.

The goal is simple: if you can build it from scratch, you truly understand it.

This repository is the mathematical foundation half of a two-repository system. The companion repository [ML-with-Scikit-Learn](https://github.com/ather-ops/ML-with-Scikit-Learn) applies the same concepts using production-grade tools. Study this one first.


```bash
git clone https://github.com/ather-ops/Machine-Learning-from-scratch.git
cd Machine-Learning-from-scratch
pip install -r requirements.txt
python "Linear-Regression/student_scores_example.py"
```

To run every model at once once `all_models.py` is complete:

```bash
python all_models.py
```

---

## Dependencies

```text
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
```

No scikit-learn. No shortcut libraries. That is the point.

---

## Related Repository

| Repository | Purpose |
|------------|---------|
| [Supervised-Machine-Learning-from-scratch](https://github.com/ather-ops/Machine-Learning-from-scratch) | This repo — mathematical foundation, pure NumPy |
| [Supervised-ML-with-Scikit-Learn](https://github.com/ather-ops/ML-with-Scikit-Learn) | Same algorithms using sklearn, pipelines, real projects |

---

## License

MIT. Use freely.

---

## Author

[ather-ops](https://github.com/ather-ops)
