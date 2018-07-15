from typing import Callable, Tuple, TypeVar, Iterator

import numpy as np
from itertools import chain


def ceildiv(n, d):
    return -(-n // d)


T = TypeVar('T')

# Adapted from https://github.com/numpy/numpy/issues/2269#issuecomment-14436725
def find(a: 'np.ndarray[T]', predicate: 'Callable[[np.ndarray[T]], np.ndarray[bool]]',
         chunk_size=1024) -> Iterator[Tuple[Tuple[int], T]]:
    """
    Find the indices of array elements that match the predicate.

    Parameters
    ----------
    a : array_like
        Input data, must be 1D.

    predicate : function
        A function which operates on sections of the given array, returning
        element-wise True or False for each data value.

    chunk_size : integer
        The length of the chunks to use when searching for matching indices.
        For high probability predicates, a smaller number will make this
        function quicker, similarly choose a larger number for low
        probabilities.

    Returns
    -------
    index_generator : generator
        A generator of (indices, data value) tuples which make the predicate
        True.

    See Also
    --------
    where, nonzero

    Notes
    -----
    This function is best used for finding the first, or first few, data values
    which match the predicate.

    Examples
    --------
    >>> a = np.sin(np.linspace(0, np.pi, 200))
    >>> result = find(a, lambda arr: arr > 0.9)
    >>> next(result)
    ((71, ), 0.900479032457)
    >>> np.where(a > 0.9)[0][0]
    71


    """
    if a.ndim != 1:
        raise ValueError('The array must be 1D, not {}.'.format(a.ndim))

    i0 = 0
    chunk_inds = chain(range(chunk_size, a.size, chunk_size), [None])

    for i1 in chunk_inds:
        chunk = a[i0:i1]
        for idx in predicate(chunk).nonzero()[0]:
            yield (idx + i0, ), chunk[idx]
        i0 = i1
