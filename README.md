# TIDE

This code contains Pytorch implementation of [TIDE](https://arxiv.org/pdf/2605.00499):

> Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation.
> Zhiying Deng, Yuan Fu, Usman Farooq, Ziwei Tian, Wei Liu, Jianjun Li.
> The 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2026).

TIDE employs a dual-expert framework to disentangle users’ stable repurchase habits and curiosity-driven exploration, avoiding the intent entanglement caused by unified representations. Specifically, TIDE uses a Hawkes-enhanced Habit Expert with Fourier-based frequency encoding to capture non-monotonic replenishment patterns. Additionally, TIDE introduces a Pattern-Guided Exploration Expert to retrieve latent collaborative interests from a global item-pattern bipartite graph, alleviating the sparsity of exploratory behaviors.

## Environments  

RTX4090.

torch 2.9.1+cuda 12.8.

python 3.11.14.

numpy 2.3.5.

scipy 1.17.0.

scikit-learn 1.8.0.

We suggest you create a new environment with `conda create -n TIDE python=3.11`
And then conduct: `pip install -r requirements.txt`

## Running the code
```python
$ python main.py --dataset beauty
$ python main.py --dataset grocery
$ python main.py --dataset home
$ python main.py --dataset sports
```

## Reference

```
@article{deng2026time,
  title={Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation},
  author={Deng, Zhiying and Fu, Yuan and Farooq, Usman and Tian, Ziwei and Liu, Wei and Li, Jianjun},
  journal={arXiv preprint arXiv:2605.00499},
  year={2026}
}
```

