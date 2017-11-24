# pregel
Tensorflow implementation of Graph Convolutional Network

## Setup

* `sudo pip3 install -r requirements.txt`

## Run

* `python3 main.py -h` to view all the config parameters. Update the default parameters in the `main.py` file.
* `python3 main.py` to run the models.

## References

This work is an attempt to reproduce some of the works related to graph convolutional networks: 

#### [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```

* [Author's implementation in Tensorflow](https://github.com/tkipf/gcn)
* [Author's implementation in Keras](https://github.com/tkipf/keras-gcn)
* [Author's implementation in PyTorch](https://github.com/tkipf/pygcn)

#### [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)

```
@article{hamilton2017representation,
  title={Representation Learning on Graphs: Methods and Applications},
  author={Hamilton, William L and Ying, Rex and Leskovec, Jure},
  journal={arXiv preprint arXiv:1709.05584},
  year={2017}
}
```

* [Author's implementation in Tensorflow](https://github.com/tkipf/gae)
