# Copyright 2023 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Jordan-Wigner transform on fermionic operators."""

from openfermion.transforms import jordan_wigner as openfermion_jordan_wigner
from tangelo.toolboxes.operators import FermionOperator, QubitOperator

#@profile
def memoize_mapping(func):
    """ Store the results of the decorated function for fast lookup """
    cache = {}
    def wrapper(*args, **kwargs):

        op, other_args = args[0], args[1:]
        #print(op, other_args, kwargs)

        qop = QubitOperator().to_openfermion()
        print('WOO')

        # Traverse operator, assign 1 term at a time for mapping
        f_tmp = FermionOperator()
        for k, v in op.terms.items():
            f_tmp.terms = {k: v}
            str_args = (str(f_tmp.terms), str(other_args), str(kwargs)) if other_args else (str(f_tmp.terms), str(kwargs))
            if str_args not in cache:
                cache[str_args] = func(f_tmp, other_args, **kwargs) if other_args else func(f_tmp, **kwargs)
            #qop += cache[(str(args), str(kwargs))]
            qop += cache[str_args]
        return QubitOperator.from_openfermion(qop)
    return wrapper

#@memoize_mapping
@profile
def jordan_wigner(operator):
    r"""Apply the Jordan-Wigner transform to a FermionOperator,
    InteractionOperator, or DiagonalCoulombHamiltonian to convert to a
    QubitOperator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Returns:
        QubitOperator: An instance of the QubitOperator class.

    Warning:
        The runtime of this method is exponential in the maximum locality of the
        original FermionOperator.

    Raises:
        TypeError: Operator must be a FermionOperator,
        DiagonalCoulombHamiltonian, or InteractionOperator.
    """

    # V0
    # qubit_operator = openfermion_jordan_wigner(operator)
    # return qubit_operator

    # # V1
    cache = {}
    op, other_args = operator, ()
    qop = QubitOperator().to_openfermion()

    # Traverse operator, assign 1 term at a time for mapping
    f_tmp = FermionOperator()
    for k, v in op.terms.items():
        f_tmp.terms = {k: v}
        str_args = str(f_tmp.terms)
        if str_args not in cache:
            a = openfermion_jordan_wigner(f_tmp)
            cache[str_args] = a
        qop += cache[str_args]
    return QubitOperator.from_openfermion(qop)
