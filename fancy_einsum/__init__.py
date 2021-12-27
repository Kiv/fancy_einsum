"""
Concerns:
- Nice error handling (map back to original strings)
- Complete feature set:
    - Ellipsis
    - Optional RHS
"""
import re
import sys
import string

# This part follows einops
_backends = {}

def get_backend(tensor):
    for framework_name, backend in _backends.items():
        if backend.is_appropriate_type(tensor):
            return backend

    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)

    for BackendSubclass in backend_subclasses:
        if BackendSubclass.framework_name not in _backends:
            # check that module was already imported. Otherwise it can't be imported
            if BackendSubclass.framework_name in sys.modules:
                backend = BackendSubclass()
                _backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    return backend

    raise RuntimeError('Tensor type unknown: {}'.format(type(tensor)))



class AbstractBackend:
    framework_name = None

    def is_appropriate_type(self, tensor):
        raise NotImplementedError

    def einsum(self, equation, *operands):
        raise NotImplementedError


class TorchBackend(AbstractBackend):
    framework_name = 'torch'

    def __init__(self):
        import torch
        self.torch = torch

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.torch.Tensor)

    def einsum(self, equation, *operands):
        return self.torch.einsum(equation, *operands)


class NumpyBackend(AbstractBackend):
    framework_name = 'numpy'

    def __init__(self):
        import numpy
        self.np = numpy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.np.ndarray)

    def einsum(self, equation, *operands):
        return self.np.einsum(equation, *operands)

# end part following einops



_part_re = re.compile(r'\.{3}|\w+|,|->')

def convert_equation(equation):
    SPECIAL = ['...', ',', '', '->']
    terms = _part_re.findall(equation)
    if '->' not in terms:
        # Infer RHS side
        # Important that we sort alphabetically by long names, not short ones
        rhs = ['->']
        if '...' in terms:
            rhs.append('...')
        rhs.extend(sorted(term for term in terms if
            term not in SPECIAL and terms.count(term) == 1))
        terms.extend(rhs)
    
    # First pass: prefer to map long names to first letter, uppercase if needed
    # so "time" becomes t if possible, then T.
    short_to_long = {}
    long_to_short = {}
    def try_make_abbr(s):
        base = s[0]
        if base not in short_to_long:
            short_to_long[base] = s
            long_to_short[s] = base
            return True
        inverted = base.upper() if base == base.lower() else base.lower()
        if inverted not in short_to_long:
            short_to_long[inverted] = s
            long_to_short[s] = inverted
            return True
        return False

    # Handle multiple long with same first letter. Second one gets first available letter
    conflicts = []
    for term in terms:
        if term not in SPECIAL and term not in long_to_short and not try_make_abbr(term):
            conflicts.append(term)
    if conflicts:
        available = [c for c in string.ascii_uppercase + string.ascii_lowercase if c not in short_to_long]
        solution = list(zip(conflicts, available))
        short_to_long.update(solution)
        long_to_short.update((l, s) for s, l in solution.items())

    new_equation = ''.join(term if term in SPECIAL else long_to_short[term] for term in terms)
    return new_equation


def einsum(equation, *operands):
    backend = get_backend(operands[0])
    new_equation = convert_equation(equation)
    return backend.einsum(new_equation, *operands)