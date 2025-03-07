from __future__ import annotations


__doc__ = """
.. currentmodule:: pytato.transform.calls

.. autofunction:: inline_calls
.. autofunction:: concatenate_calls
.. autofunction:: tag_all_calls_to_be_inlined
.. autofunction:: zero_unused_call_bindings

.. autoclass:: CallSiteLocation
"""

__copyright__ = "Copyright (C) 2022 Kaushik Kulkarni"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import itertools
import logging
import numpy as np
from functools import partialmethod, reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Generator,
    List,
    Never,
    Sequence,
    Set,
    Tuple,
    cast,
)
from typing_extensions import Self

import attrs
from immutabledict import immutabledict

import pymbolic.primitives as prim
from pytools import memoize_method, memoize_on_first_arg

import pytato.scalar_expr as scalar_expr
from pytato.analysis import collect_nodes_of_type
from pytato.array import (
    AbstractResultWithNamedArrays,
    Array,
    AxisPermutation,
    BasicIndex,
    Concatenate,
    DataWrapper,
    DictOfNamedArrays,
    Einsum,
    IndexBase,
    IndexLambda,
    InputArgumentBase,
    Placeholder,
    Reshape,
    Roll,
    ShapeComponent,
    ShapeType,
    SizeParam,
    Stack,
    concatenate,
    zeros,
)
from pytato.function import Call, FunctionDefinition, NamedCallResult


if TYPE_CHECKING:
    from collections.abc import Mapping
from pytato.tags import (
    ConcatenatedCallInputConcatAxisTag,
    ConcatenatedCallOutputSliceAxisTag,
    FunctionIdentifier,
    ImplStored,
    InlineCallTag,
    UseInputAxis,
)
from pytato.transform import (
    ArrayOrNames,
    CachedMapper,
    CachedWalkMapper,
    CombineMapper,
    CopyMapper,
    Deduplicator,
    InputGatherer,
    TransformMapperCache,
    TransformMapperWithExtraArgs,
    _verify_is_array,
)
from pytato.transform.lower_to_index_lambda import to_index_lambda
from pytato.utils import are_shape_components_equal


if TYPE_CHECKING:
    from pytato.loopy import LoopyCallResult

logger = logging.getLogger(__name__)

ArrayOnStackT = Tuple[Tuple[Call, ...], Array]


# {{{ inlining

class PlaceholderSubstitutor(CopyMapper):
    """
    .. attribute:: substitutions

        A mapping from the placeholder name to the array that it is to be
        substituted with.

    .. note::

        This mapper does not deduplicate subexpressions that occur in both the mapped
        expression and the substitutions. Must follow up with a
        :class:`pytato.transform.Deduplicator` if duplicates need to be removed.
    """

    def __init__(self, substitutions: Mapping[str, Array]) -> None:
        # Ignoring function cache, since we don't support functions anyway
        super().__init__()
        self.substitutions = substitutions

    def map_placeholder(self, expr: Placeholder) -> Array:
        # Can't call rec() to remove duplicates here, because the substituted-in
        # expression may potentially contain unrelated placeholders whose names
        # collide with the ones being replaced
        return self.substitutions[expr.name]

    def map_function_definition(
            self, expr: FunctionDefinition) -> FunctionDefinition:
        # Only operates within the current stack frame
        return expr


class Inliner(CopyMapper):
    """
    Primary mapper for :func:`inline_calls`.
    """
    def __init__(
            self,
            _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
            _function_cache: TransformMapperCache[FunctionDefinition, []] | None = None
            ) -> None:
        # Must disable collision/duplication checking because we're combining
        # expressions that were previously in two different call stack frames
        # (and were thus cached separately)
        super().__init__(
            err_on_collision=False,
            err_on_created_duplicate=False,
            _cache=_cache,
            _function_cache=_function_cache)

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            _function_cache=cast(
                "TransformMapperCache[FunctionDefinition, []]", self._function_cache))

    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        if expr.tags_of_type(InlineCallTag):
            substitutor = PlaceholderSubstitutor(expr.bindings)

            return DictOfNamedArrays(
                {name: _verify_is_array(self.rec(substitutor(ret)))
                 for name, ret in expr.function.returns.items()},
                tags=expr.tags
            )
        else:
            return super().map_call(expr)

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        new_call_or_inlined_expr = self.rec(expr._container)
        assert isinstance(new_call_or_inlined_expr, AbstractResultWithNamedArrays)
        if isinstance(new_call_or_inlined_expr, Call):
            return new_call_or_inlined_expr[expr.name]
        else:
            return new_call_or_inlined_expr[expr.name].expr


class InlineMarker(CopyMapper):
    """
    Primary mapper for :func:`tag_all_calls_to_be_inlined`.
    """
    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        rec_expr = super().map_call(expr)
        if rec_expr.tags_of_type(InlineCallTag):
            return rec_expr
        else:
            return rec_expr.tagged(InlineCallTag())


def inline_calls(expr: ArrayOrNames) -> ArrayOrNames:
    """
    Returns a copy of *expr* with call sites tagged with
    :class:`pytato.tags.InlineCallTag` inlined into the expression graph.
    """
    inliner = Inliner()
    return inliner(expr)


def tag_all_calls_to_be_inlined(expr: ArrayOrNames) -> ArrayOrNames:
    """
    Returns a copy of *expr* with all reachable instances of
    :class:`pytato.function.Call` nodes tagged with
    :class:`pytato.tags.InlineCallTag`.

    .. note::

       This routine does NOT inline calls, to inline the calls
       use :func:`tag_all_calls_to_be_inlined` on this routine's
       output.
    """
    return InlineMarker()(expr)

# }}}


# {{{ _collect_used_call_inputs

class _UsedCallInputCollector(CachedWalkMapper[[]]):
    def __init__(
            self,
            _fn_input_gatherers:
                dict[FunctionDefinition, InputGatherer] | None = None,
            _visited_functions: set[Any] | None = None
            ) -> None:
        if _fn_input_gatherers is None:
            _fn_input_gatherers = {}

        self.call_to_used_inputs: dict[Call, set[Placeholder]] = {}
        self._fn_input_gatherers = _fn_input_gatherers

        super().__init__(_visited_functions=_visited_functions)

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            _fn_input_gatherers=self._fn_input_gatherers,
            _visited_functions=self._visited_functions)

    def get_cache_key(self, expr: ArrayOrNames) -> ArrayOrNames:
        return expr

    def get_function_definition_cache_key(
            self, expr: FunctionDefinition) -> FunctionDefinition:
        return expr

    # type-ignore-reason: CachedWalkMapper's method takes in variadic args, kwargs
    def map_named_call_result(
            self, expr: NamedCallResult,  # type: ignore[override]
            ) -> None:
        call = expr._container
        try:
            input_gatherer = self._fn_input_gatherers[call.function]
        except KeyError:
            input_gatherer = InputGatherer()
            self._fn_input_gatherers[call.function] = input_gatherer

        used_inputs = self.call_to_used_inputs.setdefault(call, set())
        used_inputs |= input_gatherer(call.function.returns[expr.name])

        super().map_named_call_result(expr)


def _collect_used_call_inputs(
        expr: ArrayOrNames) -> immutabledict[Call, frozenset[Placeholder]]:
    """
    Returns a mapping from :class:`~pytato.function.Call` to the set of input
    :class:`~pt.array.Placeholder`\ s belonging to its function definition that are
    actually used by the expression. In other words, it returns the inputs
    corresponding to the call bindings that would remain in the DAG if the call was
    inlined.
    """
    collector = _UsedCallInputCollector()
    collector(expr)

    return immutabledict({
        call: frozenset(inputs)
        for call, inputs in collector.call_to_used_inputs.items()})

# }}}


# {{{ zero_unused_call_bindings

class _UnusedCallBindingZeroer(CopyMapper):
    """
    Mapper to replace unused bindings of :class:`~pytato.function.Call` with zeros
    of appropriate shape.
    """
    def __init__(
            self,
            call_to_used_inputs: Mapping[Call, frozenset[Placeholder]],
            _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
            _function_cache: TransformMapperCache[FunctionDefinition, []] | None = None
            ) -> None:
        super().__init__(_cache=_cache, _function_cache=_function_cache)
        self.call_to_used_inputs = call_to_used_inputs

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        return type(self)(
            call_to_used_inputs=self.call_to_used_inputs,
            _function_cache=self._function_cache)

    def map_call(self, expr: Call) -> Call:
        new_function = self.rec_function_definition(expr.function)
        new_bindings = {}
        for name, bnd in expr.bindings.items():
            if isinstance(bnd, Array):
                if (
                        expr.function.get_placeholder(name)
                        in self.call_to_used_inputs[expr]):
                    new_bnd = self.rec(bnd)
                else:
                    new_bnd = zeros(bnd.shape, bnd.dtype)
            else:
                new_bnd = bnd
            new_bindings[name] = new_bnd
        if (
                new_function is expr.function
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return Call(new_function, immutabledict(new_bindings), tags=expr.tags)


def zero_unused_call_bindings(expr: ArrayOrNames) -> ArrayOrNames:
    """
    Replaces :class:`~pytato.function.Call` bindings that are not used by the
    expression with arrays of zeros of the appropriate shape. This can be necessary
    for certain transformations such as concatenation, where otherwise bindings
    may be retained in the DAG when they should be dropped.
    """
    call_to_used_inputs = _collect_used_call_inputs(expr)
    return _UnusedCallBindingZeroer(call_to_used_inputs)(expr)

# }}}


# {{{ Concatenatability

@attrs.define(frozen=True)
class Concatenatability:
    """
    Describes how a particular array expression can be concatenated.
    """


@attrs.define(frozen=True)
class ConcatableAlongAxis(Concatenatability):
    """
    Used to describe an array expression that is concatenatable along *axis*.
    """
    axis: int


@attrs.define(frozen=True)
class ConcatableIfConstant(Concatenatability):
    """
    Used to describe an array expression in a function body that can be
    concatenated only if the expression is the same across call-sites.
    """

# }}}


# {{{ concatenate_calls

@attrs.define(frozen=True)
class CallSiteLocation:
    r"""
    Records a call-site's location in a :mod:`pytato` expression.

    .. attribute:: call

        The instance of :class:`~pytato.function.Call` being called at this
        location.

    .. attribute:: stack

        The call sites within which this particular call is called.
        For eg. if ``stack = (c1, c2)``, then :attr:`call` is called within
        ``c2``\ 's function body which itself is called from ``c1``\ 's
        function body.
    """
    call: Call
    stack: Tuple[Call, ...]


class CallSiteDependencyCollector(
        CombineMapper[FrozenSet[CallSiteLocation], Never]):
    r"""
    Collects all the call sites in a :mod:`pytato` expression along with their
    interdependencies.

    .. attribute:: stack

        The stack of calls at which the calls are being collected. This
        attribute is used to specify :attr:`CallSiteLocation.stack` in the
        :class:`CallSiteLocation`\ s being built. Must be altered (by creating
        a new instance of the mapper) before entering the function body of a
        new :class:`~pytato.function.Call`.

    .. attribute:: call_site_to_dep_call_sites

        A mapping from call site to the call sites on which it depends, for each
        call site present in the expression.
    """
    def __init__(self, stack: Tuple[Call, ...]) -> None:
        self.stack = stack
        self.call_site_to_dep_call_sites: \
            Dict[CallSiteLocation, CallSiteLocation] = {}
        super().__init__()

    def combine(self, *args: FrozenSet[CallSiteLocation]
                ) -> FrozenSet[CallSiteLocation]:
        return reduce(lambda a, b: a | b, args, frozenset())

    def map_size_param(self, expr: SizeParam) -> FrozenSet[CallSiteLocation]:
        return frozenset()

    def map_call(self, expr: Call) -> FrozenSet[CallSiteLocation]:
        cs = CallSiteLocation(expr, self.stack)

        new_mapper_for_fn = CallSiteDependencyCollector(stack=self.stack + (expr,))
        dependent_call_sites = self.combine(
            *[
                self.rec(bnd) for bnd in expr.bindings.values()
                if isinstance(bnd, Array)],
            *[new_mapper_for_fn(ret)
              for ret in expr.function.returns.values()])

        self.call_site_to_dep_call_sites[cs] = dependent_call_sites
        self.call_site_to_dep_call_sites.update(
            new_mapper_for_fn.call_site_to_dep_call_sites)

        return self.combine(frozenset([cs]), dependent_call_sites)


class _NamedCallResultReplacerPostConcatenate(CopyMapper):
    """
    Mapper to replace instances of :class:`~pytato.function.NamedCallResult` as
    per :attr:`replacement_map`.

    .. attribute:: current_stack

        Records the stack to track which function body the mapper is
        traversing. Must be altered (by creating a new instance) before
        entering the function body of a new :class:`~pytato.function.Call`.
    """
    def __init__(
            self,
            replacement_map: Mapping[
                Tuple[
                    NamedCallResult,
                    Tuple[Call, ...]],
                Array],
            current_stack: Tuple[Call, ...],
            _cache: TransformMapperCache[ArrayOrNames, []] | None = None,
            _function_cache: TransformMapperCache[FunctionDefinition, []] | None = None
            ) -> None:
        super().__init__(_cache=_cache, _function_cache=_function_cache)
        self.replacement_map = replacement_map
        self.current_stack = current_stack

    def clone_for_callee(
            self, function: FunctionDefinition) -> Self:
        raise AssertionError("Control should not reach here."
                             " Call clone_with_new_call_on_stack instead.")

    def clone_with_new_call_on_stack(self, expr: Call) -> Self:
        # type-ignore-reason: Mapper class does not define these attributes.
        return type(self)(  # type: ignore[call-arg]
            self.replacement_map,  # type: ignore[attr-defined]
            self.current_stack + (expr,),  # type: ignore[attr-defined]
            _function_cache=self._function_cache
        )

    def map_function_definition(
            self, expr: FunctionDefinition) -> FunctionDefinition:
        # No clone here because we're cloning in map_call instead
        new_returns = {name: self.rec(ret)
                       for name, ret in expr.returns.items()}
        if all(
                new_ret is ret
                for ret, new_ret in zip(
                    expr.returns.values(),
                    new_returns.values())):
            return expr
        else:
            return attrs.evolve(expr, returns=immutabledict(new_returns))

    def map_call(self, expr: Call) -> AbstractResultWithNamedArrays:
        new_mapper = self.clone_with_new_call_on_stack(expr)
        new_function = new_mapper.rec_function_definition(expr.function)
        new_bindings = {
            name: self.rec(bnd) if isinstance(bnd, Array) else bnd
            for name, bnd in expr.bindings.items()}
        if (
                new_function is expr.function
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return Call(new_function, immutabledict(new_bindings), tags=expr.tags)

    def map_named_call_result(self, expr: NamedCallResult) -> Array:
        try:
            new_expr = self.replacement_map[expr, self.current_stack]
            if isinstance(new_expr, NamedCallResult):
                return super().map_named_call_result(new_expr)
            else:
                return self.rec(new_expr)
        except KeyError:
            return super().map_named_call_result(expr)


def _have_same_axis_length(arrays: Collection[Array],
                           iaxis: int) -> bool:
    """
    Returns *True* only if every array in *arrays* have the same axis length
    along *iaxis*.
    """
    axis_length = next(iter(arrays)).shape[iaxis]
    return all(are_shape_components_equal(other_ary.shape[iaxis],
                                          axis_length)
               for other_ary in arrays)


def _have_same_axis_length_except(arrays: Collection[Array],
                                  iaxis: int) -> bool:
    """
    Returns *True* only if every array in *arrays* have the same
    dimensionality and have axes with the same lengths except along the
    *iaxis*-axis.
    """
    ndim = next(iter(arrays)).ndim
    return (all(ary.ndim == ndim for ary in arrays)
            and all(_have_same_axis_length(arrays, idim)
                    for idim in range(ndim)
                    if idim != iaxis))


@attrs.define(frozen=True)
class _InputConcatabilityGetterAcc:
    r"""
    Return type for :class:`_InputConcatabilityGetter`. An instance of this class is
    returned after mapping a :class:`~pytato.Array` expression.

    .. attribute:: seen_inputs

        A :class:`frozenset` of all :class:`pytato.InputArgumentBase`
        predecessors of a node.

    .. attribute:: input_concatability

        Records the constraints that come along with concatenating the array
        being mapped. The constraints are recorded as a mapping from the axes
        of the array being mapped to the axes of the input arguments. This
        mapping informs us which axes in the :class:`InputArgumentBase`\ s'
        must be concatenated to soundly concatenate a particular axis in the
        array being mapped. The axes in this mapping are represented using
        :class:`int`. If certain axes are missing in this mapping, then
        concatenation cannot be performed along those axes for the mapped
        array.
    """
    seen_inputs: FrozenSet[InputArgumentBase]
    input_concatability: Mapping[Concatenatability,
                                 Mapping[InputArgumentBase, Concatenatability]]

    def __post_init__(self) -> None:
        assert all(
            frozenset(input_concat.keys()) == self.seen_inputs
            for input_concat in self.input_concatability.values())

    __attrs_post_init__ = __post_init__


class NonConcatableExpression(RuntimeError):
    """
    Used internally by :class:`_ScalarExprConcatabilityMapper`.
    """


class _InvalidConcatenatability(RuntimeError):
    """
    Used internally by :func:`_get_ary_to_concatenatabilities`.
    """


class _ScalarExprConcatabilityMapper(scalar_expr.CombineMapper):
    """
    Maps :attr:`~pytato.array.IndexLambda.expr` to the axes of the bindings
    that must be concatenated to concatenate the IndexLambda's
    :attr:`iaxis`-axis.

    .. attribute:: allow_indirect_addr

        If *True* indirect access are allowed. However, concatenating along the
        iaxis-axis would be sound only if the binding which is being indexed
        into is same for all the expressions to be concatenated.
    """
    def __init__(self, iaxis: int, allow_indirect_addr: bool) -> None:
        self.iaxis = iaxis
        self.allow_indirect_addr = allow_indirect_addr
        super().__init__()

    def combine(self, values: Collection[Mapping[str, Concatenatability]]
                ) -> Mapping[str, Concatenatability]:
        result: Dict[str, Concatenatability] = {}
        for value in values:
            for bnd_name, iaxis in value.items():
                try:
                    if result[bnd_name] != iaxis:
                        # only one axis of a particular binding can be
                        # concatenated. If multiple axes must be concatenated
                        # that means the index lambda is not
                        # iaxis-concatenatable.
                        raise NonConcatableExpression
                except KeyError:
                    result[bnd_name] = iaxis

        return immutabledict(result)

    def map_variable(self, expr: prim.Variable) -> Mapping[str, Concatenatability]:
        if expr.name == f"_{self.iaxis}":
            raise NonConcatableExpression
        else:
            return immutabledict()

    def map_constant(self, expr: Any) -> Mapping[str, Concatenatability]:
        return immutabledict()

    map_nan = map_constant

    def map_subscript(self, expr: prim.Subscript
                      ) -> Mapping[str, Concatenatability]:
        name: str = expr.aggregate.name
        rec_indices: List[Mapping[str, Concatenatability]] = []
        for iaxis, idx in enumerate(expr.index_tuple):
            if idx == prim.Variable(f"_{self.iaxis}"):
                rec_indices.append({name: ConcatableAlongAxis(iaxis)})
            else:
                rec_idx = self.rec(idx)
                if rec_idx:
                    if not self.allow_indirect_addr:
                        raise NonConcatableExpression
                    else:
                        # indirect accesses cannot be concatenated in the general
                        # case unless the indexee is the same for the
                        # expression graphs being concatenated.
                        pass
                rec_indices.append(rec_idx)

        combined_rec_indices = dict(self.combine(rec_indices))

        if name not in combined_rec_indices:
            combined_rec_indices[name] = ConcatableIfConstant()

        return immutabledict(combined_rec_indices)


@memoize_on_first_arg
def _get_binding_to_concatenatability_scalar_expr(
        expr: scalar_expr.ScalarExpression,
        iaxis: int,
        allow_indirect_addr: bool) -> Mapping[str, Concatenatability]:
    mapper = _ScalarExprConcatabilityMapper(iaxis, allow_indirect_addr)
    return mapper(expr)  # type: ignore[no-any-return]



def _get_binding_to_concatenatability(expr: scalar_expr.ScalarExpression,
                                      iaxis: int,
                                      allow_indirect_addr: bool,
                                      ) -> Mapping[str, Concatenatability]:
    """
    Maps *expr* using :class:`_ScalarExprConcatabilityMapper`.
    """
    if np.isscalar(expr):
        # In some cases expr may just be a number, which can't be memoized on
        return {}

    return _get_binding_to_concatenatability_scalar_expr(
        expr, iaxis, allow_indirect_addr)


def _combine_input_accs(
    operand_accs: Tuple[_InputConcatabilityGetterAcc, ...],
    concat_to_operand_concatabilities: Mapping[Concatenatability,
                                               Tuple[Concatenatability, ...]
                                               ],
) -> _InputConcatabilityGetterAcc:
    """
    For an index lambda ``I`` with operands ``I1, I2, .. IN`` that specify their
    concatenatability constraints using *operand_accs*, this routine returns
    the axes concatenation constaints of ``I``.

    :arg concat_to_operand_concatabilities: Mapping of the form ``concat_I ->
        (C_I1, C_I2, ..., C_IN)`` specifying the concatabilities of the
        operands ``I1, I2, .., IN`` in order to concatenate the
        ``I`` axis via the criterion ``conncat_I``.
    """

    input_concatabilities: Dict[Concatenatability, Mapping[InputArgumentBase,
                                                       Concatenatability]] = {}
    seen_inputs: FrozenSet[InputArgumentBase] = reduce(
        frozenset.union,
        (operand_acc.seen_inputs for operand_acc in operand_accs),
        frozenset())

    # The core logic here is to filter the iaxis in out_axis_to_operand_axes
    # so that all the operands agree on how the input arguments must be
    # concatenated.

    for out_concat, operand_concatabilities in (concat_to_operand_concatabilities
                                                .items()):
        is_i_out_axis_concatenatable = True
        input_concatability: Dict[InputArgumentBase, Concatenatability] = {}

        for operand_concatability, operand_acc in zip(operand_concatabilities,
                                                      operand_accs,
                                                      strict=True):
            if operand_concatability not in (
                    operand_acc.input_concatability):
                # required operand concatability cannot be achieved
                # => out_concat cannot be concatenated
                is_i_out_axis_concatenatable = False
                break

            for input_arg, input_concat in (
                    operand_acc
                    .input_concatability[operand_concatability]
                    .items()):
                try:
                    if input_concatability[input_arg] != input_concat:
                        is_i_out_axis_concatenatable = False
                        break
                except KeyError:
                    input_concatability[input_arg] = input_concat
            if not is_i_out_axis_concatenatable:
                break

        if is_i_out_axis_concatenatable:
            input_concatabilities[out_concat] = immutabledict(input_concatability)

    return _InputConcatabilityGetterAcc(seen_inputs,
                                        immutabledict(input_concatabilities))


@attrs.define(frozen=True)
class FunctionConcatenability:
    r"""
    Records a valid concatenatability criterion for a
    :class:`pytato.function.FunctionDefinition`.

    .. attribute:: output_to_concatenatability

        A mapping from the name of a
        :class:`FunctionDefinition`\ 's returned array to how it should be
        concatenated.

    .. attribute:: input_to_concatenatability

        A mapping from a :class:`FunctionDefinition`\ 's parameter to how it
        should be concatenated.

    .. note::

        A :class:`FunctionDefinition` typically has multiple valid
        concatenability constraints. This class only records one of those valid
        constraints.
    """
    output_to_concatenatability: Mapping[str, Concatenatability]
    input_to_concatenatability: Mapping[str, Concatenatability]

    def __str__(self) -> str:
        outputs = []
        for name, concat in self.output_to_concatenatability.items():
            outputs.append(f"{name} => {concat}")

        inputs = []
        for name, concat in self.input_to_concatenatability.items():
            inputs.append(f"{name} => {concat}")

        output_str = "\n".join(outputs)
        input_str = "\n".join(inputs)

        return (f"Outputs:\n--------\n{output_str}\n"
                f"========\nInputs:\n-------\n{input_str}\n"
                "========")


def _combine_named_result_accs_simple(
        named_result_accs: Mapping[str, _InputConcatabilityGetterAcc]
) -> Tuple[FunctionConcatenability, ...]:
    """
    Combines the concantenatability constraints of named results of a
    :class:`FunctionDefinition` and returns a :class:`tuple` of the valid
    *simple* concatenatable constraints (i.e., concatenation of all inputs/outputs
    along the same axis).
    """
    valid_concatenatabilities: List[FunctionConcatenability] = []

    input_args = reduce(
        frozenset.union,
        [
            acc.seen_inputs
            for acc in named_result_accs.values()],
        frozenset())

    candidate_concat_axes = reduce(
        frozenset.union,
        [
            frozenset(acc.input_concatability.keys())
            for acc in named_result_accs.values()],
        frozenset())

    # print(f"{candidate_concat_axes=}")

    for i_concat_axis in candidate_concat_axes:
        # if isinstance(i_concat_axis, ConcatableAlongAxis) and i_concat_axis.axis == 0:
        #     for acc in named_result_accs.values():
        #         for ary, concat in acc.input_concatability[i_concat_axis].items():
        #             print(f"{type(ary).__name__=}, {ary.name=}, {ary.shape=}, {id(ary)=}, {concat=}")
        #         print("")
        if (
                all(
                    i_concat_axis in acc.input_concatability
                    for acc in named_result_accs.values())
                and all(
                    (
                        i_input_axis == i_concat_axis
                        or isinstance(i_input_axis, ConcatableIfConstant))
                    for acc in named_result_accs.values()
                    for i_input_axis in (
                        acc.input_concatability[i_concat_axis].values()))):
            output_concats = {name: i_concat_axis for name in named_result_accs}
            input_concats = {pl.name: i_concat_axis
                             for pl in input_args
                             if isinstance(pl, Placeholder)}
            valid_concatenatabilities.append(
                FunctionConcatenability(immutabledict(output_concats),
                                        immutabledict(input_concats)))

    return valid_concatenatabilities


# FIXME: Find a more efficient way to do this. The number of candidates
# explodes when the function being concatenated has more than a few outputs
def _combine_named_result_accs_exhaustive(
        named_result_accs: Mapping[str, _InputConcatabilityGetterAcc]
) -> Generator[
        FunctionConcatenability,
        None,
        None]:
    """
    Combines the concantenatability constraints of named results of a
    :class:`FunctionDefinition` and returns a :class:`tuple` of the valid
    concatenatable constraints.
    """
    potential_concatenatable_output_axes = itertools.product(*[
        [(name, concat) for concat in acc.input_concatability]
        for name, acc in named_result_accs.items()])

    for output_concats in potential_concatenatable_output_axes:
        is_concatenatable = True
        input_concatability: Dict[InputArgumentBase, Concatenatability] = {}

        for result_name, iresult_axis in output_concats:
            for input_arg, i_input_axis in (
                    named_result_accs[result_name]
                    .input_concatability[iresult_axis]
                    .items()):
                try:
                    if input_concatability[input_arg] != i_input_axis:
                        is_concatenatable = False
                        break
                except KeyError:
                    input_concatability[input_arg] = i_input_axis

            if not is_concatenatable:
                break

        if is_concatenatable:
            pl_concatabilities = {pl.name: concat
                                  for pl, concat in input_concatability.items()
                                  if isinstance(pl, Placeholder)}
            yield FunctionConcatenability(immutabledict(output_concats),
                                          immutabledict(pl_concatabilities))


class _InputConcatabilityGetter(
        CachedMapper[ArrayOrNames, Never, [ArrayOrNames, ...]]):
    """
    Maps :class:`pytato.array.Array` expressions to
    :class:`_InputConcatenatabilityGetterAcc` that summarizes constraints
    induced on the concatenatability of the inputs of the expression by  the
    expression's concatenatability.
    """
    def get_cache_key(
                self, expr: ArrayOrNames, *exprs_from_other_calls: ArrayOrNames
            ) -> tuple[ArrayOrNames, ...]:
        return (expr, *exprs_from_other_calls)

    def _map_input_arg_base(
            self,
            expr: InputArgumentBase,
            *exprs_from_other_calls: InputArgumentBase,
            ) -> _InputConcatabilityGetterAcc:
        input_concatenatability: Dict[Concatenatability,
                                      Mapping[InputArgumentBase,
                                          Concatenatability]] = {}
        for idim in range(expr.ndim):
            input_concatenatability[ConcatableAlongAxis(idim)] = immutabledict(
                {expr: ConcatableAlongAxis(idim)})

        input_concatenatability[ConcatableIfConstant()] = immutabledict(
            {expr: ConcatableIfConstant()})

        return _InputConcatabilityGetterAcc(frozenset([expr]),
                                            immutabledict(input_concatenatability))

    map_placeholder = _map_input_arg_base
    map_data_wrapper = _map_input_arg_base

    def _map_index_lambda_like(
            self,
            expr: Array,
            *exprs_from_other_calls: Array,
            allow_indirect_addr: bool) -> _InputConcatabilityGetterAcc:
        expr = to_index_lambda(expr)
        exprs_from_other_calls = tuple(
            to_index_lambda(ary) for ary in exprs_from_other_calls)

        input_accs = tuple(
            self.rec(
                expr.bindings[name],
                *[
                    ary.bindings[name]
                    for ary in exprs_from_other_calls])
            for name in sorted(expr.bindings.keys()))
        expr_concat_to_input_concats: Dict[Concatenatability,
                                           Tuple[Concatenatability, ...]] = {}

        for iaxis in range(expr.ndim):
            for ary in (expr,) + exprs_from_other_calls:
                # If the array has length 1 along this axis, the index may have been
                # dropped from the scalar expression, in which case
                # _get_binding_to_concatenatability will fail to determine the
                # concatenatability. If that happens, we have to look at the other
                # expressions in the hope that one of them has a non-1 length
                if ary.shape[iaxis] == 1:
                    continue
                try:
                    bnd_name_to_concat = _get_binding_to_concatenatability(
                        ary.expr, iaxis, allow_indirect_addr)
                    expr_concat_to_input_concats[ConcatableAlongAxis(iaxis)] = (
                        tuple(concat
                              for _, concat in sorted(bnd_name_to_concat.items(),
                                                      key=lambda x: x[0]))
                    )
                except NonConcatableExpression:
                    # print(f"{iaxis=}")
                    # print(f"{ary.expr=}")
                    # print(f"{ary.shape=}")
                    break

        expr_concat_to_input_concats[ConcatableIfConstant()] = tuple(
            ConcatableIfConstant() for _ in expr.bindings)

        return _combine_input_accs(input_accs, expr_concat_to_input_concats)

    map_index_lambda = partialmethod(_map_index_lambda_like,
                                      allow_indirect_addr=False)
    map_einsum = partialmethod(_map_index_lambda_like,
                                allow_indirect_addr=False)
    map_basic_index = partialmethod(_map_index_lambda_like,
                                     allow_indirect_addr=False)
    map_roll = partialmethod(_map_index_lambda_like,
                              allow_indirect_addr=False)
    map_stack = partialmethod(_map_index_lambda_like,
                               allow_indirect_addr=False)
    map_concatenate = partialmethod(_map_index_lambda_like,
                                     allow_indirect_addr=False)
    map_axis_permutation = partialmethod(_map_index_lambda_like,
                                          allow_indirect_addr=False)
    map_reshape = partialmethod(_map_index_lambda_like,
                                 allow_indirect_addr=False)

    map_contiguous_advanced_index = partialmethod(_map_index_lambda_like,
                                                   allow_indirect_addr=True)
    map_non_contiguous_advanced_index = partialmethod(_map_index_lambda_like,
                                                       allow_indirect_addr=True)

    def map_named_call_result(
            self,
            expr: NamedCallResult,
            *exprs_from_other_calls: NamedCallResult,
            ) -> _InputConcatabilityGetterAcc:
        raise NotImplementedError("nested functions aren't supported.")

        # FIXME: Update the code below to work after changing
        # _InputConcatabilityGetter to look at all function calls instead of just
        # the template call
        assert isinstance(expr._container, Call)
        valid_concatenatabilities = _get_valid_concatenatability_constraints_simple(
            expr._container.function)

        expr_concat_possibilities = {
            valid_concatenability.output_to_concatenatability[expr.name]
            for valid_concatenability in valid_concatenatabilities
        }

        input_concatenatabilities: Dict[Concatenatability,
                                        Mapping[InputArgumentBase,
                                                Concatenatability]] = {}
        rec_bindings = {bnd_name: self.rec(binding)
                        for bnd_name, binding in expr._container.bindings.items()}
        callee_acc = self.rec(expr._container.function.returns[expr.name])
        seen_inputs: Set[InputArgumentBase] = set()

        for seen_input in callee_acc.seen_inputs:
            if isinstance(seen_input, Placeholder):
                seen_inputs.update(rec_bindings[seen_input.name].seen_inputs)
            elif isinstance(seen_input, (DataWrapper, SizeParam)):
                seen_inputs.add(seen_input)
            else:
                raise NotImplementedError(type(seen_input))

        for concat_possibility in expr_concat_possibilities:
            caller_input_concatabilities: Dict[InputArgumentBase,
                                               Concatenatability] = {}

            is_concat_possibility_valid = True
            for callee_input_arg, callee_input_concat in (
                    callee_acc.input_concatability[concat_possibility].items()):
                caller_acc = rec_bindings[callee_input_arg.name]
                if isinstance(callee_input_arg, Placeholder):
                    if callee_input_concat in caller_acc.input_concatability:
                        for caller_input_arg, caller_input_concat in (
                                caller_acc
                                .input_concatability[callee_input_concat]
                                .items()):
                            try:
                                if (caller_input_concatabilities[caller_input_arg]
                                        != caller_input_concat):
                                    is_concat_possibility_valid = False
                                    break
                            except KeyError:
                                caller_input_concatabilities[callee_input_arg] = (
                                    caller_input_concat)
                        if not is_concat_possibility_valid:
                            break
                    else:
                        is_concat_possibility_valid = False
                        break
                elif isinstance(callee_input_arg, (DataWrapper, SizeParam)):
                    try:
                        if (caller_input_concatabilities[callee_input_arg]
                                != callee_input_concat):
                            is_concat_possibility_valid = False
                            break
                    except KeyError:
                        caller_input_concatabilities[callee_input_arg] = (
                            callee_input_concat)
                else:
                    raise NotImplementedError(type(callee_input_arg))

            if is_concat_possibility_valid:
                input_concatenatabilities[concat_possibility] = immutabledict(
                    caller_input_concatabilities)

        return _InputConcatabilityGetterAcc(frozenset(seen_inputs),
                                            immutabledict(input_concatenatabilities))

    def map_loopy_call_result(
            self,
            expr: "LoopyCallResult",
            *exprs_from_other_calls: "LoopyCallResult",
            ) -> _InputConcatabilityGetterAcc:
        raise ValueError("Loopy Calls are illegal to concatenate. Maybe"
                         " rewrite the operation as array operations?")


def _verify_arrays_can_be_concated_along_axis(
        arrays: Collection[Array],
        fields_that_must_be_same: Collection[str],
        iaxis: int) -> None:
    """
    Performs some common checks if *arrays* from different function bodies can be
    concatenated.

    .. attribute:: arrays

        Corresponding expressions from the function bodies for call-site that
        are being checked for concatenation along *iaxis*.
    """
    if not _have_same_axis_length_except(arrays, iaxis):
        raise _InvalidConcatenatability("Axis lengths are incompatible.")
    for field in fields_that_must_be_same:
        if len({getattr(ary, field) for ary in arrays}) != 1:
            raise _InvalidConcatenatability(f"Field '{field}' varies across calls.")


def _verify_arrays_same(arrays: Collection[Array]) -> None:
    if len(set(arrays)) != 1:
        raise _InvalidConcatenatability("Arrays are not the same.")


def _get_concatenated_shape(arrays: Collection[Array], iaxis: int) -> ShapeType:
    # type-ignore-reason: mypy expects 'ary.shape[iaxis]' as 'int' since the
    # 'start' is an 'int'
    concatenated_axis_length = sum(ary.shape[iaxis]  # type: ignore[misc]
                                   for ary in arrays)
    template_ary = next(iter(arrays))

    return tuple(dim
                 if idim != iaxis
                 else concatenated_axis_length
                 for idim, dim in enumerate(template_ary.shape)
                 )


class _ConcatabilityCollector(CachedWalkMapper):
    def __init__(
            self,
            current_stack: Tuple[Call, ...],
            _visited_functions: set[Any] | None = None
            ) -> None:
        self.ary_to_concatenatability: Dict[ArrayOnStackT, Concatenatability] = {}
        self.current_stack = current_stack
        self.call_sites_on_hold: Set[Call] = set()
        super().__init__(_visited_functions=_visited_functions)

    # type-ignore-reason: CachedWalkMaper takes variadic `*args, **kwargs`.
    def get_cache_key(self,  # type: ignore[override]
                      expr: ArrayOrNames,
                      *args: Any,
                      ) -> Tuple[ArrayOrNames, Any]:
        return (expr, args)

    # type-ignore-reason: CachedWalkMaper takes variadic `*args, **kwargs`.
    def get_function_definition_cache_key(
            self,  # type: ignore[override]
            expr: FunctionDefinition,
            *args: Any,
            ) -> tuple[ArrayOrNames, Any]:
        return (expr, args)

    def _record_concatability(self, expr: Array,
                              concatenatability: Concatenatability,
                              ) -> None:
        key = (self.current_stack, expr)
        assert key not in self.ary_to_concatenatability
        self.ary_to_concatenatability[key] = concatenatability

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        raise AssertionError("Control should not reach here."
                             " Call clone_with_new_call_on_stack instead.")

    def clone_with_new_call_on_stack(self, expr: Call) -> Self:
        # type-ignore-reason: Mapper class does not define these attributes.
        return type(self)(  # type: ignore[call-arg]
            self.current_stack + (expr,),  # type: ignore[attr-defined]
            _visited_functions=self._visited_functions
        )

    def _map_input_arg_base(self,
                            expr: InputArgumentBase,
                            concatenatability: Concatenatability,
                            exprs_from_other_calls: Tuple[Array, ...],
                            ) -> None:
        if isinstance(concatenatability, ConcatableIfConstant):
            _verify_arrays_same((expr,) + exprs_from_other_calls)
        elif isinstance(concatenatability, ConcatableAlongAxis):
            # FIXME: Probably needs some extra handling for broadcastable arrays
            _verify_arrays_can_be_concated_along_axis(
                (expr,) + exprs_from_other_calls,
                ["dtype", "name"],
                concatenatability.axis)
        else:
            raise NotImplementedError(type(concatenatability))

        self._record_concatability(expr, concatenatability)

    map_placeholder = _map_input_arg_base  # type: ignore[assignment]
    map_data_wrapper = _map_input_arg_base  # type: ignore[assignment]

    def _map_index_lambda_like(self,
                               expr: Array,
                               concatenatability: Concatenatability,
                               exprs_from_other_calls: Tuple[Array, ...],
                               allow_indirect_addr: bool,
                               ) -> None:
        self._record_concatability(expr, concatenatability)

        idx_lambda = to_index_lambda(expr)
        idx_lambdas_from_other_calls = tuple(to_index_lambda(ary)
                                             for ary in exprs_from_other_calls)

        if isinstance(concatenatability, ConcatableIfConstant):
            _verify_arrays_same((idx_lambda,) + idx_lambdas_from_other_calls)
            for bnd_name in idx_lambda.bindings:
                self.rec(
                    idx_lambda.bindings[bnd_name], concatenatability,
                    tuple(
                        ary.bindings[bnd_name]
                        for ary in idx_lambdas_from_other_calls))
        elif isinstance(concatenatability, ConcatableAlongAxis):
            _verify_arrays_can_be_concated_along_axis(
                (idx_lambda, ) + idx_lambdas_from_other_calls,
                ["dtype"],
                concatenatability.axis)
            if len({
                    ary.expr
                    for ary in (idx_lambda,) + idx_lambdas_from_other_calls
                    if ary.shape[concatenatability.axis] != 1}) != 1:
                raise _InvalidConcatenatability(
                    "Cannot concatenate the calls; required fields are not the same.")
            bnd_name_to_concat = None
            for ary in (idx_lambda,) + idx_lambdas_from_other_calls:
                if ary.shape[concatenatability.axis] > 1:
                    bnd_name_to_concat = _get_binding_to_concatenatability(
                        ary.expr, concatenatability.axis, allow_indirect_addr)
                    break
            if bnd_name_to_concat is None:
                bnd_name_to_concat = _get_binding_to_concatenatability(
                    idx_lambda.expr, concatenatability.axis, allow_indirect_addr)
            for bnd_name, bnd_concat in bnd_name_to_concat.items():
                self.rec(idx_lambda.bindings[bnd_name], bnd_concat,
                         tuple(ary.bindings[bnd_name]
                               for ary in idx_lambdas_from_other_calls))
        else:
            raise NotImplementedError(type(concatenatability))

    map_index_lambda = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_einsum = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_basic_index = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_roll = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_stack = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_concatenate = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_axis_permutation = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)
    map_reshape = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=False)

    map_contiguous_advanced_index = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=True)
    map_non_contiguous_advanced_index = partialmethod(  # type: ignore[assignment]
        _map_index_lambda_like, allow_indirect_addr=True)

    # type-ignore-reason: CachedWalkMapper.map_call takes in variadic args, kwargs
    def map_call(self,  # type: ignore[override]
                 expr: Call,
                 exprs_from_other_calls: Tuple[Call, ...]) -> None:
        if not all(
                (self.current_stack, named_result) in self.ary_to_concatenatability
                for named_result in expr.values()):
            self.call_sites_on_hold.add(expr)
        else:
            self.call_sites_on_hold -= {expr}
            # FIXME The code below bypasses caching of function definitions
            new_mapper = self.clone_with_new_call_on_stack(expr)
            for name, val_in_callee in expr.function.returns.items():
                new_mapper(val_in_callee,
                           self.ary_to_concatenatability[(self.current_stack,
                                                          expr[name])],
                           tuple(other_call.function.returns[name]
                                 for other_call in exprs_from_other_calls)
                           )

            if new_mapper.call_sites_on_hold:
                raise NotImplementedError("Call sites that do not all use all"
                                          " the returned values not yet"
                                          " supported for concatenation.")

            for ary, concat in new_mapper.ary_to_concatenatability.items():
                assert ary not in self.ary_to_concatenatability
                self.ary_to_concatenatability[ary] = concat

            for name, binding in expr.bindings.items():
                if not isinstance(binding, Array):
                    continue
                concat = (
                    new_mapper
                    .ary_to_concatenatability[(self.current_stack + (expr,),
                                               expr.function.get_placeholder(name))]
                )
                self.rec(binding,
                         concat,
                         tuple(other_call.bindings[name]
                               for other_call in exprs_from_other_calls))

    # type-ignore-reason: CachedWalkMapper's method takes in variadic args, kwargs
    def map_named_call_result(self, expr: NamedCallResult,  # type: ignore[override]
                              concatenatability: Concatenatability,
                              exprs_from_other_calls: Tuple[Array, ...],
                              ) -> None:
        self._record_concatability(expr, concatenatability)
        if any(not isinstance(ary, NamedCallResult)
               for ary in exprs_from_other_calls):
            raise _InvalidConcatenatability()

        # type-ignore-reason: mypy does not respect the conditional which
        # asserts that all arrays in `exprs_from_other_calls` are
        # NamedCallResult.
        self.rec(expr._container,
                 tuple(ary._container  # type: ignore[attr-defined]
                       for ary in exprs_from_other_calls)
                 )

    def map_loopy_call_result(self, expr: "LoopyCallResult"
                              ) -> None:
        raise ValueError("Loopy Calls are illegal to concatenate. Maybe"
                         " rewrite the operation as array operations?")


# Memoize the creation of concatenated input arrays to avoid copies
class _InputConcatenator:
    def __init__(self, inherit_axes: bool):
        self.inherit_axes = inherit_axes

    @memoize_method
    def __call__(self, arrays, axis):
        if self.inherit_axes:
            concat_axis_tag = UseInputAxis(0, axis)
        else:
            concat_axis_tag = ConcatenatedCallInputConcatAxisTag()
        return concatenate(
                arrays,
                axis
            ).with_tagged_axis(axis, frozenset({concat_axis_tag})).tagged(
                ImplStored())


# Memoize the creation of sliced output arrays to avoid copies
class _OutputSlicer:
    def __init__(self, inherit_axes: bool):
        self.inherit_axes = inherit_axes

    @memoize_method
    def _get_slice(
            self,
            ary: Array,
            axis: int,
            start_idx: ShapeComponent,
            end_idx: ShapeComponent):
        indices = [slice(None) for i in range(ary.ndim)]
        indices[axis] = slice(start_idx, end_idx)
        if self.inherit_axes:
            slice_axis_tag = UseInputAxis(None, axis)
        else:
            slice_axis_tag = ConcatenatedCallOutputSliceAxisTag()
        sliced_ary = ary[tuple(indices)].with_tagged_axis(
            axis, frozenset({slice_axis_tag})).tagged(ImplStored())
        assert isinstance(sliced_ary, BasicIndex)
        return sliced_ary

    def __call__(self, ary, axis, slice_sizes):
        start_indices: List[ShapeComponent] = []
        end_indices: List[ShapeComponent] = []
        if len(slice_sizes) > 0:
            start_indices.append(0)
            end_indices.append(slice_sizes[0])
            for islice in range(1, len(slice_sizes)):
                start_indices.append(end_indices[-1])
                end_indices.append(end_indices[-1] + slice_sizes[islice])
        return [
            self._get_slice(ary, axis, start_idx, end_idx)
            for start_idx, end_idx in zip(start_indices, end_indices)]


class _FunctionConcatenator(TransformMapperWithExtraArgs[Tuple[Array, ...]]):
    def __init__(self,
                 current_stack: Tuple[Call, ...],
                 input_concatenator: _InputConcatenator,
                 ary_to_concatenatability: Mapping[ArrayOnStackT, Concatenatability],
                 _cache: TransformMapperCache[
                    ArrayOrNames, [Tuple[Array, ...]]] | None = None,
                 _function_cache: TransformMapperCache[
                    FunctionDefinition, [Tuple[Array, ...]]] | None = None
                 ) -> None:
        super().__init__(_cache=_cache, _function_cache=_function_cache)
        self.current_stack = current_stack
        self.input_concatenator = input_concatenator
        self.ary_to_concatenatability = ary_to_concatenatability

    def get_cache_key(
                self, expr: ArrayOrNames, exprs_from_other_calls: tuple[Array, ...]
            ) -> tuple[ArrayOrNames, ...]:
        return (expr, *exprs_from_other_calls)

    def clone_for_callee(self, function: FunctionDefinition) -> Self:
        raise AssertionError("Control should not reach here."
                             " Call clone_with_new_call_on_stack instead.")

    def clone_with_new_call_on_stack(self, expr: Call) -> Self:
        return type(self)(
            self.current_stack + (expr,),
            self.input_concatenator,
            self.ary_to_concatenatability,
            _function_cache=self._function_cache
        )

    def _get_concatenatability(self, expr: Array) -> Concatenatability:
        return self.ary_to_concatenatability[(self.current_stack, expr)]

    def map_placeholder(self,
                        expr: Placeholder,
                        exprs_from_other_calls: Tuple[Array, ...]
                        ) -> Array:
        concat = self._get_concatenatability(expr)
        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            new_shape = _get_concatenated_shape(
                (expr,) + exprs_from_other_calls, concat.axis)
            return Placeholder(name=expr.name,
                               dtype=expr.dtype,
                               shape=new_shape,
                               tags=expr.tags,
                               axes=expr.axes,
                               non_equality_tags=expr.non_equality_tags)
        else:
            raise NotImplementedError(type(concat))

    def map_data_wrapper(self,
                         expr: DataWrapper,
                         exprs_from_other_calls: Tuple[Array, ...]
                         ) -> Array:
        concat = self._get_concatenatability(expr)
        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            return self.input_concatenator(
                (expr,) + exprs_from_other_calls, concat.axis)
        else:
            raise NotImplementedError(type(concat))

    def map_index_lambda(self,
                         expr: IndexLambda,
                         exprs_from_other_calls: Tuple[Array, ...]
                         ) -> Array:
        concat = self._get_concatenatability(expr)
        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, IndexLambda)
                       for ary in exprs_from_other_calls)

            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are IndexLambda.
            new_bindings = {
                bnd_name: self.rec(
                           subexpr,
                           tuple(ary.bindings[bnd_name]  # type: ignore[attr-defined]
                                 for ary in exprs_from_other_calls))
                for bnd_name, subexpr in expr.bindings.items()
            }
            new_shape = _get_concatenated_shape((expr,) + exprs_from_other_calls,
                                                concat.axis)
            return IndexLambda(expr=expr.expr,
                               shape=new_shape,
                               dtype=expr.dtype,
                               bindings=immutabledict(new_bindings),
                               var_to_reduction_descr=expr.var_to_reduction_descr,
                               tags=expr.tags,
                               axes=expr.axes,
                               non_equality_tags=expr.non_equality_tags)
        else:
            raise NotImplementedError(type(concat))

    def map_einsum(self, expr: Einsum,
                   exprs_from_other_calls: Tuple[Array, ...]) -> Array:
        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, Einsum) for ary in exprs_from_other_calls)

            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Einsum.
            new_args = [self.rec(arg,
                                 tuple(ary.args[iarg]  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls))
                        for iarg, arg in enumerate(expr.args)]

            return Einsum(expr.access_descriptors,
                          tuple(new_args),
                          expr.redn_axis_to_redn_descr,
                          tags=expr.tags,
                          axes=expr.axes,
                          non_equality_tags=expr.non_equality_tags)
        else:
            raise NotImplementedError(type(concat))

    def _map_index_base(self, expr: IndexBase,
                        exprs_from_other_calls: Tuple[Array, ...]) -> Array:
        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, IndexBase) for ary in exprs_from_other_calls)

            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are IndexBase.
            new_indices = [
                self.rec(idx,
                         tuple(ary.indices[i_idx]  # type: ignore[attr-defined]
                               for ary in exprs_from_other_calls))
                if isinstance(idx, Array)
                else idx
                for i_idx, idx in enumerate(expr.indices)
            ]
            new_array = self.rec(expr.array,
                                 tuple(ary.array  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls))

            return type(expr)(array=new_array,
                              indices=tuple(new_indices),
                              tags=expr.tags,
                              axes=expr.axes,
                              non_equality_tags=expr.non_equality_tags)
        else:
            raise NotImplementedError(type(concat))

    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base
    map_basic_index = _map_index_base

    def map_roll(self,
                 expr: Roll,
                 exprs_from_other_calls: Tuple[Array, ...]
                 ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert concat.axis != expr.axis
            assert all(isinstance(ary, Roll) for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Roll.
            return Roll(self.rec(expr.array,
                                 tuple(ary.array  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls)),
                        shift=expr.shift,
                        axis=expr.axis,
                        tags=expr.tags,
                        axes=expr.axes,
                        non_equality_tags=expr.non_equality_tags)
        else:
            raise NotImplementedError(type(concat))

    def map_stack(self, expr: Stack,
                  exprs_from_other_calls: Tuple[Array, ...]
                  ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, Stack) for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Stack.
            if any(len(ary.arrays) != len(expr.arrays)  # type: ignore[attr-defined]
                   for ary in exprs_from_other_calls):
                raise ValueError("Cannot concatenate stack expressions"
                                 " with different number of arrays.")

            new_arrays = tuple(
                self.rec(array,
                         tuple(subexpr.arrays[iarray]  # type: ignore[attr-defined]
                               for subexpr in exprs_from_other_calls)
                         )
                for iarray, array in enumerate(expr.arrays))

            return Stack(new_arrays,
                         expr.axis,
                         tags=expr.tags,
                         axes=expr.axes,
                         non_equality_tags=expr.non_equality_tags)
        else:
            raise NotImplementedError(type(concat))

    def map_concatenate(self, expr: Concatenate,
                        exprs_from_other_calls: Tuple[Array, ...]
                        ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, Concatenate)
                       for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Concatenate.
            if any(len(ary.arrays) != len(expr.arrays)  # type: ignore[attr-defined]
                   for ary in exprs_from_other_calls):
                raise ValueError("Cannot concatenate concatenate-expressions"
                                 " with different number of arrays.")

            new_arrays = tuple(
                self.rec(array,
                         tuple(subexpr.arrays[iarray]  # type: ignore[attr-defined]
                               for subexpr in exprs_from_other_calls)
                         )
                for iarray, array in enumerate(expr.arrays)
            )

            return Concatenate(new_arrays,
                               expr.axis,
                               tags=expr.tags,
                               axes=expr.axes,
                               non_equality_tags=expr.non_equality_tags)
        else:
            raise NotImplementedError(type(concat))

    def map_axis_permutation(self, expr: AxisPermutation,
                             exprs_from_other_calls: Tuple[Array, ...]
                             ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, AxisPermutation)
                       for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are AxisPermutation.
            new_array = self.rec(expr.array,
                                 tuple(ary.array  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls))
            return AxisPermutation(new_array,
                                   expr.axis_permutation,
                                   tags=expr.tags,
                                   axes=expr.axes,
                                   non_equality_tags=expr.non_equality_tags)
        else:
            raise NotImplementedError(type(concat))

    def map_reshape(self, expr: Reshape,
                    exprs_from_other_calls: Tuple[Array, ...]
                    ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            new_newshape = _get_concatenated_shape(
                (expr,) + exprs_from_other_calls, concat.axis)

            assert all(isinstance(ary, Reshape) for ary in exprs_from_other_calls)
            # type-ignore-reason: mypy does not respect the assertion that all
            # other exprs are Reshape.
            new_array = self.rec(expr.array,
                                 tuple(ary.array  # type: ignore[attr-defined]
                                       for ary in exprs_from_other_calls))
            return Reshape(new_array,
                           new_newshape,
                           expr.order,
                           tags=expr.tags,
                           axes=expr.axes,
                           non_equality_tags=expr.non_equality_tags)
        else:
            raise NotImplementedError(type(concat))

    def map_function_definition(
            self,
            expr: FunctionDefinition,
            exprs_from_other_calls: Tuple[FunctionDefinition, ...]
            ) -> FunctionDefinition:
        # No clone here because we're cloning in map_call instead
        new_returns = {
            name: self.rec(
                ret,
                tuple(
                    other_expr.returns[name]
                    for other_expr in exprs_from_other_calls))
            for name, ret in expr.returns.items()}
        if all(
                new_ret is ret
                for ret, new_ret in zip(
                    expr.returns.values(),
                    new_returns.values())):
            return expr
        else:
            return attrs.evolve(expr, returns=immutabledict(new_returns))

    def map_call(self, expr: Call, other_callsites: Tuple[Call, ...]) -> Call:
        new_mapper = self.clone_with_new_call_on_stack(expr)
        new_function = new_mapper.rec_function_definition(
            expr.function,
            tuple(other_call.function for other_call in other_callsites))
        new_bindings = {name: (
                            self.rec(
                                bnd, tuple(
                                    callsite.bindings[name]
                                    for callsite in other_callsites))
                            if isinstance(bnd, Array)
                            else bnd)
                        for name, bnd in expr.bindings.items()}
        if (
                new_function is expr.function
                and all(
                    new_bnd is bnd
                    for bnd, new_bnd in zip(
                        expr.bindings.values(),
                        new_bindings.values()))):
            return expr
        else:
            return Call(new_function, immutabledict(new_bindings), tags=expr.tags)

    def map_named_call_result(self,
                              expr: NamedCallResult,
                              exprs_from_other_calls: Tuple[Array, ...]
                              ) -> Array:

        concat = self._get_concatenatability(expr)

        if isinstance(concat, ConcatableIfConstant):
            return expr
        elif isinstance(concat, ConcatableAlongAxis):
            assert all(isinstance(ary, NamedCallResult)
                       for ary in exprs_from_other_calls)
            assert isinstance(expr._container, Call)
            new_call = self.rec(
                expr._container,
                tuple(ary._container  # type: ignore[attr-defined]
                      for ary in exprs_from_other_calls))
            return new_call[expr.name]
        else:
            raise NotImplementedError(type(concat))

    def map_loopy_call_result(self, expr: "LoopyCallResult",
                              exprs_from_other_calls: Tuple[Array, ...],
                              ) -> _InputConcatabilityGetterAcc:
        raise ValueError("Loopy Calls are illegal to concatenate. Maybe"
                         " rewrite the operation as array operations?")


@memoize_on_first_arg
def _get_valid_concatenatability_constraints_simple(
        template_call: Call, *other_calls: Call) -> Tuple[FunctionConcatenability]:
    template_fn = template_call.function
    mapper = _InputConcatabilityGetter()
    output_accs = {
        name: mapper(
            *[cs.function.returns[name] for cs in (template_call,) + other_calls])
        for name in template_fn.returns}

    return _combine_named_result_accs_simple(output_accs)


@memoize_on_first_arg
def _get_valid_concatenatability_constraints_exhaustive(
        fn: FunctionDefinition) -> Generator[
            FunctionConcatenability,
            None,
            None]:
    mapper = _InputConcatabilityGetter()
    output_accs = {name: mapper(output)
                   for name, output in fn.returns.items()}

    yield from _combine_named_result_accs_exhaustive(output_accs)


def _get_ary_to_concatenatabilities(call_sites: Sequence[Call],
                                    ) -> Generator[Mapping[ArrayOnStackT,
                                                       Concatenatability],
                                                   None,
                                                   None]:
    """
    Generates a :class:`Concatenatability` criterion for each array in the
    expression graph of *call_sites*'s function body if they traverse identical
    function bodies.
    """
    fn_concatenatabilities = \
        _get_valid_concatenatability_constraints_simple(*call_sites)

    # select a template call site to start the traversal.
    template_call, *other_calls = call_sites
    template_fn = template_call.function
    fid = next(iter(template_fn.tags_of_type(FunctionIdentifier)))

    concat_idx_to_err_msg = {}

    for iconcat, fn_concatenatability in enumerate(fn_concatenatabilities):
        collector = _ConcatabilityCollector(current_stack=())

        try:
            # verify the constraints on parameters are satisfied
            for name, input_concat in (fn_concatenatability
                                       .input_to_concatenatability
                                       .items()):
                try:
                    if isinstance(input_concat, ConcatableIfConstant):
                        _verify_arrays_same([cs.bindings[name] for cs in call_sites])
                    elif isinstance(input_concat, ConcatableAlongAxis):
                        _verify_arrays_can_be_concated_along_axis(
                            [cs.bindings[name] for cs in call_sites],
                            [],
                            input_concat.axis)
                    else:
                        raise NotImplementedError(type(input_concat))
                except _InvalidConcatenatability as e:
                    raise _InvalidConcatenatability(
                        f"Binding for input {name} is not concatenatable. {str(e)}")

            # verify the constraints on function bodies are satisfied
            for name, output_concat in (fn_concatenatability
                                        .output_to_concatenatability
                                        .items()):
                try:
                    collector(template_call.function.returns[name],
                              output_concat,
                              tuple(other_call.function.returns[name]
                                    for other_call in other_calls))
                except _InvalidConcatenatability as e:
                    raise _InvalidConcatenatability(
                        f"Function output {name} is not concatenatable. {str(e)}")
        except _InvalidConcatenatability as e:
            concat_idx_to_err_msg[iconcat] = str(e)
        else:
            if collector.call_sites_on_hold:
                raise NotImplementedError("Expressions that use part of"
                                          " function's returned values are not"
                                          " yet supported.")

            logger.info(
                f"Found a valid concatenatability for function with ID '{fid}' --\n"
                f"{fn_concatenatability}")

            yield immutabledict(collector.ary_to_concatenatability)

    log_str = (
        f"No more valid concatenatabilities for function with ID '{fid}'. "
        "Unsuitable candidates:\n")
    for iconcat, fn_concatenatability in enumerate(fn_concatenatabilities):
        try:
            err_msg = concat_idx_to_err_msg[iconcat]
        except KeyError:
            continue
        log_str += f"Candidate:\n{fn_concatenatability}\n"
        log_str += f"Error: {concat_idx_to_err_msg[iconcat]}\n\n"
    logger.info(log_str)


def _get_replacement_map_post_concatenating(
        call_sites: Sequence[Call],
        used_call_results: frozenset(NamedCallResult),
        input_concatenator: _InputConcatenator,
        output_slicer: _OutputSlicer) -> Mapping[NamedCallResult, Array]:
    """
    .. note::

        We require *call_sites* to be ordered to determine the concatenation
        order.
    """
    assert call_sites, "Empty `call_sites`."

    ary_to_concatenatabilities = _get_ary_to_concatenatabilities(call_sites)

    template_call_site, *other_call_sites = call_sites
    template_function = template_call_site.function
    fid = next(iter(template_function.tags_of_type(FunctionIdentifier)))

    try:
        ary_to_concatenatability = next(ary_to_concatenatabilities)
    except StopIteration:
        raise ValueError(
            f"No valid concatenatibilities found for function with ID '{fid}'.")
    else:
        if __debug__:
            try:
                next(ary_to_concatenatabilities)
            except StopIteration:
                # unique concatenatibility
                pass
            else:
                from warnings import warn
                # TODO: Take some input from the user to resolve this ambiguity.
                warn(
                    "Multiple concatenation possibilities found for function with "
                    f"ID '{fid}'. This may lead to non-deterministic transformed "
                    "expression graphs.")

    # {{{ actually perform the concatenation

    template_returns = template_function.returns
    template_bindings = template_call_site.bindings

    function_concatenator = _FunctionConcatenator(
        current_stack=(), input_concatenator=input_concatenator,
        ary_to_concatenatability=ary_to_concatenatability)

    if __debug__:
        # FIXME: We may be able to handle this without burdening the user
        # See https://github.com/inducer/pytato/issues/559
        from collections import defaultdict
        param_to_used_calls = defaultdict(set)
        for output_name in template_call_site.keys():
            for csite in call_sites:
                call_result = csite[output_name]
                if call_result in used_call_results:
                    ret = csite.function.returns[output_name]
                    used_params = (
                        {
                            expr.name
                            for expr in InputGatherer()(ret)}
                        & csite.function.parameters)
                    for name in used_params:
                        param_to_used_calls[name] |= {csite}
        for name, used_calls in param_to_used_calls.items():
            if used_calls != set(call_sites):
                from warnings import warn
                warn(
                    f"DAG output does not depend on parameter '{name}' for some "
                    f"calls to function with ID '{fid}'. Concatenation will prevent "
                    "these unused inputs from being removed from the DAG when the "
                    "function is inlined. This may lead to unnecessary computation.")

    # new_returns: concatenated function body
    new_returns: Dict[str, Array] = {}
    for output_name in template_call_site.keys():
        new_returns[output_name] = function_concatenator(
            template_returns[output_name],
            tuple(csite.function.returns[output_name]
                  for csite in other_call_sites))

    # }}}

    # construct new function body
    if any(
            new_returns[output_name] is not template_returns[output_name]
            for output_name in template_returns):
        new_function = FunctionDefinition(
            template_call_site.function.parameters,
            template_call_site.function.return_type,
            immutabledict(new_returns),
            tags=template_call_site.function.tags)
    else:
        new_function = template_call_site.function

    result: Dict[NamedCallResult, Array] = {}

    new_call_bindings: Dict[str, Array] = {}

    # construct new bindings
    for param_name in template_bindings:
        param_placeholder = template_call_site.function.get_placeholder(param_name)
        param_concat = ary_to_concatenatability[((), param_placeholder)]
        if isinstance(param_concat, ConcatableAlongAxis):
            param_bindings = tuple([
                csite.bindings[param_name]
                for csite in call_sites])
            new_binding = input_concatenator(
                param_bindings,
                param_concat.axis)
        elif isinstance(param_concat, ConcatableIfConstant):
            new_binding = template_bindings[param_name]
        else:
            raise NotImplementedError(type(param_concat))
        new_call_bindings[param_name] = new_binding

    # construct new call
    if (
            new_function is not template_call_site.function
            or any(
                new_call_bindings[param_name] is not template_bindings[param_name]
                for param_name in template_bindings)):
        new_call = Call(
            function=new_function,
            bindings=immutabledict(new_call_bindings),
            tags=template_call_site.tags)
    else:
        new_call = template_call_site

    # slice into new_call's outputs to replace the old expressions.
    for output_name, output_ary in (template_call_site
                                    .function
                                    .returns
                                    .items()):
        concat = ary_to_concatenatability[((), output_ary)]
        new_return = new_call[output_name]
        if isinstance(concat, ConcatableIfConstant):
            # FIXME: Does it make sense to not concatenate if some arguments are
            # ConcatableIfConstant and some are ConcatableAlongAxis? Seems like that
            # would cause problems...
            for cs in call_sites:
                result[cs[output_name]] = new_return
        elif isinstance(concat, ConcatableAlongAxis):
            slice_sizes = [
                cs[output_name].shape[concat.axis]
                for cs in call_sites]
            output_slices = output_slicer(new_return, concat.axis, slice_sizes)
            for cs, output_slice in zip(call_sites, output_slices):
                result[cs[output_name]] = output_slice
        else:
            raise NotImplementedError(type(concat))

    return immutabledict(result)


def concatenate_calls(expr: ArrayOrNames,
                      call_site_filter: Callable[[CallSiteLocation], bool],
                      *,
                      inherit_axes: bool = False,
                      warn_if_no_calls: bool = True,
                      err_if_no_calls: bool = False,
                      ignore_tag_types: frozenset(type) | None = None,
                      ) -> ArrayOrNames:
    r"""
    Returns a copy of *expr* after concatenating all call-sites ``C`` such that
    ``call_site_filter(C) is True``.

    :arg call_site_filter: A callable to select which instances of
        :class:`~pytato.function.Call`\ s must be concatenated.
    """
    if ignore_tag_types is None:
        ignore_tag_types: frozenset(type) = frozenset()

    call_site_collector = CallSiteDependencyCollector(stack=())

    all_call_sites = call_site_collector(expr)
    filtered_call_sites = {cs
                           for cs in all_call_sites
                           if call_site_filter(cs)}

    function_ids = {
        next(iter(cs.call.function.tags_of_type(FunctionIdentifier)))
        for cs in filtered_call_sites}

    # Input concatenator needs to be set up outside of the loop in order to prevent
    # creating duplicates; probably not strictly necessary for output slicer
    input_concatenator = _InputConcatenator(inherit_axes=inherit_axes)
    output_slicer = _OutputSlicer(inherit_axes=inherit_axes)

    result = expr

    for fid in function_ids:
        call_site_dep_collector = CallSiteDependencyCollector(stack=())
        call_site_dep_collector(result)

        call_site_to_dep_call_sites = \
            call_site_dep_collector.call_site_to_dep_call_sites

        unbatched_call_sites: Set[CallSiteLocation] = {
            cs for cs in call_site_to_dep_call_sites.keys()
            if call_site_filter(cs) and fid in cs.call.function.tags}

        for cs in unbatched_call_sites:
            for ret in cs.call.function.returns.values():
                nested_calls = collect_nodes_of_type(ret, Call)
                if nested_calls:
                    raise NotImplementedError(
                        "Concatenation of nested calls is not yet supported.")

        call_site_batches: List[FrozenSet[CallSiteLocation]] = []

        replacement_map: Dict[
            Tuple[NamedCallResult, Tuple[Call, ...]],
            Array] = {}

        used_call_results = collect_nodes_of_type(result, NamedCallResult)

        while unbatched_call_sites:
            ready_call_sites = frozenset({
                cs for cs in unbatched_call_sites
                if not call_site_to_dep_call_sites[cs] & unbatched_call_sites})

            from mpi4py import MPI
            rank = MPI.COMM_WORLD.rank

            # if fid.identifier == "_make_fluid_state":
            #     print(f"{rank}: {len(ready_call_sites)=}")

            if not ready_call_sites:
                raise ValueError("Found cycle in call site dependency graph.")

            template_call_site = next(iter(ready_call_sites))
            template_fn = template_call_site.call.function

            from pytato.equality import SimilarityComparer
            similarity_comparer = SimilarityComparer(
                ignore_tag_types=ignore_tag_types)
                # err_on_not_similar=(fid.identifier == "_make_fluid_state"))

            # if fid.identifier == "_make_fluid_state":
            #     for cs in ready_call_sites:
            #         same_outputs = (
            #             frozenset(cs.call.function.returns.keys())
            #             == frozenset(template_fn.returns.keys()))
            #         similar = all(
            #             similarity_comparer(
            #                 cs.call.function.returns[name],
            #                 template_fn.returns[name])
            #             for name in template_fn.returns)
            #         same_stack = (cs.stack == template_call_site.stack)
            #         print(f"{rank}:    {same_outputs=}, {similar=}, {same_stack=}")
            #         # if not similar:
            #         #     for name in template_fn.returns:
            #         #         from pytato.analysis import get_num_nodes
            #         #         nnodes_template = get_num_nodes(template_fn.returns[name])
            #         #         nnodes_other = get_num_nodes(cs.call.function.returns[name])
            #         #         print(f"{rank}:        {name=}, {nnodes_template=}, {nnodes_other=}")

            similar_call_sites = frozenset({
                cs for cs in ready_call_sites
                if (
                    (
                        frozenset(cs.call.function.returns.keys())
                        == frozenset(template_fn.returns.keys()))
                    and all(
                        similarity_comparer(
                            cs.call.function.returns[name],
                            template_fn.returns[name])
                        for name in template_fn.returns)
                    and cs.stack == template_call_site.stack)})

            # if fid.identifier == "_make_fluid_state":
            #     print(f"{rank}: {len(similar_call_sites)=}")

            if not similar_call_sites:
                raise ValueError("Failed to find similar call sites to concatenate.")

            def get_axis0_len(cs):
                first_out_name = next(iter(cs.call.function.returns.keys()))
                axis0_len = cs.call[first_out_name].shape[0]
                assert all(
                    cs.call[name].shape[0] == axis0_len
                    for name in cs.call.function.returns)
                return axis0_len

            batch_call_sites = sorted(similar_call_sites, key=get_axis0_len)

            call_site_batches.append(batch_call_sites)
            unbatched_call_sites -= frozenset(batch_call_sites)

        # FIXME: this doesn't work; need to create/execute batches one at a time,
        # then repeat the steps above to collect the updated call sites after
        # concatenating the previous batch
        for ibatch, call_sites in enumerate(call_site_batches):
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.rank

            template_fn = next(iter(call_sites)).call.function

            # FIXME: Can't currently call get_num_nodes on a function definition
            from pytato.array import make_dict_of_named_arrays
            from pytato.analysis import get_num_nodes
            fn_body = make_dict_of_named_arrays(template_fn.returns)
            nnodes = get_num_nodes(fn_body)

            print(
                f"{rank}: Concatenating function '{fid}' (batch {ibatch+1} of "
                f"{len(call_site_batches)}: {nnodes} nodes, {len(call_sites)} "
                "call sites).")

            if len(call_sites) <= 1:
                if err_if_no_calls:
                    raise ValueError(
                        f"Not enough calls to concatenate function with ID '{fid}'.")
                elif warn_if_no_calls:
                    from warnings import warn
                    warn(
                        f"Not enough calls to concatenate function with ID '{fid}'.",
                        stacklevel=2)
                else:
                    pass
                continue

            old_expr_to_new_expr_map = _get_replacement_map_post_concatenating(
                    [cs.call for cs in call_sites],
                    used_call_results,
                    input_concatenator=input_concatenator,
                    output_slicer=output_slicer)

            stack, = {cs.stack for cs in call_sites}

            replacement_map.update({
                (old_expr, stack): new_expr
                for old_expr, new_expr in old_expr_to_new_expr_map.items()})

        # FIXME: Still getting some duplicated `Concatenate`s, not sure why
        dedup = Deduplicator()
        result = dedup(result)
        replacement_map = {
            old_expr_and_stack: dedup(new_expr)
            for old_expr_and_stack, new_expr in replacement_map.items()}

        result = _NamedCallResultReplacerPostConcatenate(
            replacement_map=replacement_map,
            current_stack=())(result)

    assert isinstance(result, (Array, AbstractResultWithNamedArrays))
    return result

# }}}

# vim:foldmethod=marker
