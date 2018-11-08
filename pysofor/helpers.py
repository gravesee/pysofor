from dataclasses import dataclass
import numpy as np
import math
from typing import *

@dataclass
class ExNode:
    size: int = 0


InNode = NewType("InNode", Any)


@dataclass
class InNode:
    left: InNode
    right: InNode
    normal: np.ndarray
    intercept: np.ndarray

def random_intercept(x: np.ndarray):
    lo = np.apply_along_axis(np.min, 0, x)
    hi = np.apply_along_axis(np.max, 0, x)
    return (hi - lo) * np.random.rand(lo.size) + lo


def H(i: int) -> float:
    return math.log(i) + 0.5772156649


def c(n: int) -> float:
    return 2 * H(n - 1) - (2 * (n - 1) / n)


class ITree:
    lim: int
    nodes: InNode
    phi: int

    def __init__(self, x: np.ndarray, phi: int):
        self.lim = math.ceil(math.log2(phi))
        self.phi = phi
        self.nodes = self.itree(x)

    def itree(self, x: np.ndarray, e: int = 0):
        if (e > self.lim) | (x.shape[0] <= 1):
            return ExNode(size=max(1, x.shape[0]))
        else:
            n = np.random.rand(x.shape[1])
            p = random_intercept(x)
            f = (x - p).dot(n) <= 0

            return(
                InNode(
                    left=self.itree(x=x[np.where(f), :].reshape(-1, x.shape[1]), e=e + 1),
                    right=self.itree(x=x[np.where(~f), :].reshape(-1, x.shape[1]), e=e + 1),
                    normal=n,
                    intercept=p
                )
            )

    def pathlength(self, sample: np.ndarray, node: Union[ExNode, InNode] = None, e: int = 0):
        if node is None:
            node = self.nodes

        if isinstance(node, ExNode):
            return e + c(self.phi)
        else:
            if (sample - node.intercept).dot(node.normal) <= 0:
                return self.pathlength(sample, node.left, e + 1)
            else:
                return self.pathlength(sample, node.right, e + 1)

    def pathlength_multiple(self, sample: np.ndarray):
        with Pool(5) as p:
            p.map(self.pathlength, sample)
        return np.array(p.join())
