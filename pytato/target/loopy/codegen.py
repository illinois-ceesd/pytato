from __future__ import annotations


__copyright__ = """Copyright (C) 2020 Matt Wala"""

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

import dataclasses
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

import islpy as isl
from typing_extensions import Never

import loopy as lp
import loopy.symbolic as lp_symbolic
import pymbolic.primitives as prim
from pymbolic import ArithmeticExpression, var

import pytato.reductions as red
import pytato.scalar_expr as scalar_expr
from pytato.array import (
    AbstractResultWithNamedArrays,
    Array,
    DataWrapper,
    DictOfNamedArrays,
    IndexLambda,
    InputArgumentBase,
    NamedArray,
    Placeholder,
    ReductionDescriptor,
    ShapeType,
    SizeParam,
)
from pytato.codegen import (
    SymbolicIndex,
    _generate_name_for_temp,
    is_symbolic_index,
    normalize_outputs,
    preprocess,
)
from pytato.scalar_expr import (
    INT_CLASSES,
    ScalarExpression,
    TypeCast,
)
from pytato.tags import (
    ForceValueArgTag,
    ImplementationStrategy,
    ImplInlined,
    ImplStored,
    Named,
    PrefixNamed,
)
from pytato.target.loopy import ImplSubstitution, LoopyPyOpenCLTarget, LoopyTarget
from pytato.transform import Mapper


if TYPE_CHECKING:
    import pyopencl
    import pytools
    from pymbolic.typing import Expression
    from pytools.tag import Tag

    from pytato.function import Call, NamedCallResult
    from pytato.loopy import LoopyCall
    from pytato.target import BoundProgram


# set in doc/conf.py
if getattr(sys, "_BUILDING_SPHINX_DOCS", False):
    # Avoid import unless building docs to avoid creating a hard
    # dependency on pyopencl, when Loopy can run fine without.
    from pytools.tag import Tag  # noqa: TC001

__doc__ = """
.. autoclass:: PersistentExpressionContext
.. autoclass:: LocalExpressionContext
.. autoclass:: ImplementedResult
.. autoclass:: StoredResult
.. autoclass:: InlinedResult
.. autoclass:: SubstitutionRuleResult
.. autoclass:: CodeGenState
.. autoclass:: CodeGenMapper
.. autoclass:: InlinedExpressionGenMapper

.. autofunction:: domain_for_shape
.. autofunction:: get_loopy_temporary
.. autofunction:: add_store
.. autofunction:: normalize_outputs
.. autofunction:: get_initial_codegen_state

.. class:: ReductionBounds

    A mapping from reduction inames to a tuple ``(lower_bound, upper_bound)``,
    considered half-open.

.. class:: SymbolicIndex

    See :class:`pytato.codegen.SymbolicIndex`.

.. currentmodule:: isl

.. class:: BasicSet

    See :class:`islpy.BasicSet`.
"""


def loopy_substitute(
            expression: Expression,
            variable_assignments: Mapping[str, Expression]
        ) -> Expression:
    from loopy.symbolic import SubstitutionMapper
    from pymbolic.mapper.substitutor import (
        make_subst_func,  # pyright: ignore[reportUnknownVariableType]
    )

    # {{{ early exit for identity substitution

    if all(isinstance(v, prim.Variable) and v.name == k
           for k, v in variable_assignments.items()):
        # Nothing to do here, move on.
        return expression

    # }}}

    return SubstitutionMapper(make_subst_func(variable_assignments))(expression)


# SymbolicIndex and ShapeType are semantically distinct but identical at the
# type level.
ReductionBounds = Mapping[str, tuple[ScalarExpression, ScalarExpression]]


# {{{ LoopyExpressionContexts

@dataclasses.dataclass(init=True, repr=False, eq=False)
class PersistentExpressionContext:
    """
    Mutable state used while generating :mod:`loopy` expressions for a
    :class:`ImplementedResult`. Wraps :class:`CodeGenState` with more
    expression-specific information.

    This data is passed through :class:`InlinedExpressionGenMapper` via arguments,
    and is also used by :meth:`ImplementedResult.to_loopy_expression` to
    retrieve contextual data.

    .. attribute:: state

        The :class:`CodeGenState`.

    .. attribute:: depends_on

        The set of statement IDs that need to be included in
        :attr:`loopy.InstructionBase.depends_on`.

    .. automethod:: update_depends_on

    """
    state: CodeGenState
    _depends_on: frozenset[str] = dataclasses.field(default_factory=frozenset)

    @property
    def depends_on(self) -> frozenset[str]:
        return self._depends_on

    def update_depends_on(self, other: frozenset[str]) -> None:
        self._depends_on = self._depends_on | other


@dataclasses.dataclass(frozen=True)
class LocalExpressionContext:
    """
    Records context being to be conveyed from a parent expression to its
    sub-expressions.

    .. attribute:: local_namespace

        A (read-only) local name mapping used for name lookup when generating
        code.

    .. attribute:: num_indices

        The number of indices of the form ``_0``, ``_1``, allowed in the
        expression.

    .. automethod:: lookup
    """
    num_indices: int
    local_namespace: Mapping[str, ImplementedResult]
    reduction_bounds: ReductionBounds
    var_to_reduction_descr: Mapping[str, ReductionDescriptor]

    def lookup(self, name: str) -> ImplementedResult:
        return self.local_namespace[name]

    def copy(self, *,
             reduction_bounds: ReductionBounds | None = None,
             num_indices: int | None = None,
             local_namespace: Mapping[str, ImplementedResult] | None = None,
             var_to_reduction_descr: Mapping[str, ReductionDescriptor] | None = None,
             ) -> LocalExpressionContext:
        if reduction_bounds is None:
            reduction_bounds = self.reduction_bounds
        if num_indices is None:
            num_indices = self.num_indices
        if local_namespace is None:
            local_namespace = self.local_namespace
        if var_to_reduction_descr is None:
            var_to_reduction_descr = self.var_to_reduction_descr
        return LocalExpressionContext(reduction_bounds=reduction_bounds,
                                      num_indices=num_indices,
                                      local_namespace=local_namespace,
                                      var_to_reduction_descr=var_to_reduction_descr)

# }}}


# {{{ ImplementedResult

class ImplementedResult(ABC):
    """Generated code for a node in the computation graph (i.e., an array
    expression).

    .. automethod:: to_loopy_expression
    """

    @abstractmethod
    def to_loopy_expression(self, indices: SymbolicIndex,
            expr_context: PersistentExpressionContext) -> Expression:
        """Return a :mod:`loopy` expression for this result.

        :param indices: symbolic expressions for the indices of the array
        :param expr_context: the associated expression context. The fields are
            treated as follows:

            - *depends_on* is populated with any dependencies needed for the
              generated expression.
        """

# }}}


# {{{ StoredResult

class StoredResult(ImplementedResult):
    """An array expression generated as a :mod:`loopy` array.

    See also: :class:`pytato.tags.ImplStored`.
    """
    def __init__(self, name: str, num_indices: int, depends_on: frozenset[str]):
        self.name = name
        self.num_indices = num_indices
        self.depends_on = depends_on

    def to_loopy_expression(self, indices: SymbolicIndex,
            expr_context: PersistentExpressionContext) -> Expression:
        assert len(indices) == self.num_indices
        expr_context.update_depends_on(self.depends_on)
        if indices == ():
            return prim.Variable(self.name)
        else:
            return prim.Variable(self.name)[indices]

# }}}


# {{{ InlinedResult

class InlinedResult(ImplementedResult):
    """An array expression generated as a :mod:`loopy` expression containing inlined
    sub-expressions.

    See also: :class:`pytato.tags.ImplInlined`.
    """
    def __init__(self, expr: ScalarExpression,
            num_indices: int,
            depends_on: frozenset[str]):
        self.expr = expr
        self.num_indices = num_indices
        self.depends_on = depends_on

    def to_loopy_expression(self, indices: SymbolicIndex,
            expr_context: PersistentExpressionContext) -> Expression:
        assert len(indices) == self.num_indices
        substitutions = {f"_{d}": i for d, i in enumerate(indices)}
        expr_context.update_depends_on(self.depends_on)
        return loopy_substitute(self.expr, substitutions)

# }}}


# {{{ SubstitutionRuleResult

@dataclasses.dataclass(frozen=True, eq=True)
class SubstitutionRuleResult(ImplementedResult):
    """
    An array expression generated as a
    :class:`loopy.kernel.data.SubstitutionRule`.

    See also: :class:`pytato.target.loopy.ImplSubstitution`.
    """
    subst_name: str
    num_args: int
    depends_on: frozenset[str]

    def to_loopy_expression(self,
                            indices: SymbolicIndex,
                            expr_context: PersistentExpressionContext
                            ) -> Expression:
        assert len(indices) == self.num_args
        expr_context.update_depends_on(self.depends_on)
        return prim.Call(prim.Variable(self.subst_name), indices)
# }}}


# {{{ codegen state

@dataclasses.dataclass(init=True, repr=False, eq=False)
class CodeGenState:
    """A container for data kept by :class:`CodeGenMapper`.

    .. attribute:: _t_unit

        The partial :class:`loopy.TranslationUnit`
        being built.

    .. attribute:: results

        A mapping from :class:`pytato.Array` instances to
        instances of :class:`ImplementedResult`.

    .. attribute:: var_name_gen
    .. attribute:: insn_id_gen

    .. automethod:: update_kernel
    """
    _t_unit: lp.TranslationUnit
    results: dict[Array, ImplementedResult]

    var_name_gen: pytools.UniqueNameGenerator = dataclasses.field(init=False)
    insn_id_gen: pytools.UniqueNameGenerator = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.var_name_gen = self._t_unit.default_entrypoint.get_var_name_generator()
        self.insn_id_gen = (
                self._t_unit.default_entrypoint.get_instruction_id_generator())

    @property
    def t_unit(self) -> lp.TranslationUnit:
        return self._t_unit

    @property
    def kernel(self) -> lp.LoopKernel:
        """
        Returns the entry kernel of the loopy kernel being built.
        """
        return self._t_unit.default_entrypoint

    def update_kernel(self, kernel: lp.LoopKernel) -> None:
        self._t_unit = self._t_unit.with_kernel(kernel)

    def update_t_unit(self, t_unit: lp.TranslationUnit) -> None:
        self._t_unit = t_unit

# }}}


# {{{ codegen mapper

class CodeGenMapper(Mapper[ImplementedResult, Never, [CodeGenState]]):
    """A mapper for generating code for nodes in the computation graph.
    """
    exprgen_mapper: InlinedExpressionGenMapper
    has_loopy_call: bool

    def __init__(self,
                 array_tag_t_to_not_propagate: frozenset[type[Tag]],
                 axis_tag_t_to_not_propagate: frozenset[type[Tag]]) -> None:
        super().__init__()
        self.exprgen_mapper = InlinedExpressionGenMapper(axis_tag_t_to_not_propagate)
        self.array_tag_t_to_not_propagate = array_tag_t_to_not_propagate
        self.axis_tag_t_to_not_propagate = axis_tag_t_to_not_propagate
        self.has_loopy_call = False

    def map_size_param(self, expr: SizeParam,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        arg = lp.ValueArg(expr.name,
                          dtype=expr.dtype,
                          tags=_filter_tags_not_of_type(expr,
                                                        self
                                                        .array_tag_t_to_not_propagate
                                                        ))
        kernel = state.kernel.copy(args=[*state.kernel.args, arg])
        state.update_kernel(kernel)
        assert expr.name is not None
        result = StoredResult(expr.name, expr.ndim, frozenset())
        state.results[expr] = result
        return result

    def map_placeholder(self, expr: Placeholder,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        shape = shape_to_scalar_expression(expr.shape, self, state)

        if expr.tags_of_type(ForceValueArgTag):
            if expr.shape != ():
                raise ValueError("ForceValueArgTag applied to non-scalar")

            arg: lp.ArrayArg | lp.ValueArg = lp.ValueArg(expr.name,
                              dtype=expr.dtype,
                              tags=_filter_tags_not_of_type(expr,
                                                            self
                                                            .array_tag_t_to_not_propagate))
        else:

            arg = lp.GlobalArg(expr.name,
                shape=shape,
                dtype=expr.dtype,
                order="C",
                offset=lp.auto,
                is_input=True,
                is_output=False,
                tags=_filter_tags_not_of_type(expr,
                                              self
                                              .array_tag_t_to_not_propagate))

        kernel = state.kernel.copy(args=[*state.kernel.args, arg])
        state.update_kernel(kernel)
        assert expr.name is not None
        result = StoredResult(expr.name, expr.ndim, frozenset())
        state.results[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        prstnt_ctx = PersistentExpressionContext(state)
        local_ctx = LocalExpressionContext(
            local_namespace={
                name: self.rec(expr.bindings[name], state)
                for name in sorted(expr.bindings)},
            num_indices=expr.ndim,
            reduction_bounds={},
            var_to_reduction_descr=expr.var_to_reduction_descr)
        loopy_expr = self.exprgen_mapper(expr.expr, prstnt_ctx, local_ctx)

        assert not isinstance(loopy_expr, tuple)
        result: ImplementedResult = InlinedResult(loopy_expr,
                                                  expr.ndim,
                                                  prstnt_ctx.depends_on)

        shape_to_scalar_expression(expr.shape, self, state)  # walk over size params

        # {{{ implementation tag

        if expr.tags_of_type(ImplStored):
            name = _generate_name_for_temp(expr, state.var_name_gen)
            result = StoredResult(name, expr.ndim,
                                  frozenset([add_store(name, expr,
                                                       result, state,
                                                       self, True)]))
        elif expr.tags_of_type(ImplInlined):
            # inlined results are automatically handled
            pass
        elif expr.tags_of_type(ImplSubstitution):
            subst_name = _generate_name_for_temp(expr, state.var_name_gen,
                                           default_prefix="_pt_subst")

            add_substitution(subst_name, expr, result, state, self)
            result = SubstitutionRuleResult(subst_name, expr.ndim,
                                            prstnt_ctx.depends_on)
        elif expr.tags_of_type(ImplementationStrategy):
            raise NotImplementedError(
                "Implementation strategy: "
                f"'{next(iter(expr.tags_of_type(ImplementationStrategy)))}'."
            )
        else:
            # default is inlining
            pass
        # }}}

        state.results[expr] = result
        return result

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays,
            state: CodeGenState) -> None:
        for key in sorted(expr.keys()):
            subexpr = expr[key].expr
            name = _generate_name_for_temp(subexpr, state.var_name_gen)
            insn_id = add_store(name, subexpr, self.rec(subexpr, state), state,
                    output_to_temporary=True, cgen_mapper=self)
            state.results[subexpr] = state.results[expr[key]] = (
                    StoredResult(name, subexpr.ndim, frozenset([insn_id])))

    def map_named_array(self, expr: NamedArray,
            state: CodeGenState) -> ImplementedResult:
        if expr in state.results:
            return state.results[expr]

        self.rec(expr._container, state)

        assert expr in state.results
        return state.results[expr]

    def map_loopy_call(self, expr: LoopyCall, state: CodeGenState) -> None:
        self.has_loopy_call = True
        from loopy.kernel.instruction import make_assignment
        from loopy.symbolic import SubArrayRef

        callee_kernel = expr.translation_unit[expr.entrypoint]

        state.update_t_unit(lp.merge([state.t_unit, expr.translation_unit]))

        domains = []

        def _get_sub_array_ref(array: Array, name: str) -> lp_symbolic.SubArrayRef:
            inames = tuple(
                    state.var_name_gen(f"_{name}_dim{d}")
                    for d in range(array.ndim))

            domains.append(domain_for_shape(inames,
                                            shape_to_scalar_expression(array.shape,
                                                                       self, state),
                                            {}))

            inames_as_vars = tuple(var(iname) for iname in inames)
            return SubArrayRef(inames_as_vars,
                               prim.Subscript(var(name), inames_as_vars))

        assignees = []
        params: list[Expression] = []
        depends_on: set[str] = set()
        new_tvs = {}
        new_insn_id = state.insn_id_gen(f"call_{callee_kernel.name}")

        for arg in callee_kernel.args:
            # must traverse in the order of callee's args to generate the correct
            # assignees order
            if isinstance(arg, lp.ArrayArg):
                if arg.is_output:
                    assignee_name = _generate_name_for_temp(
                            expr[arg.name], state.var_name_gen)
                    assignees.append(_get_sub_array_ref(expr[arg.name],
                                                        assignee_name))

                    named_array = expr[arg.name]

                    # stored result for the assignee
                    result = StoredResult(assignee_name, named_array.ndim,
                                          frozenset([new_insn_id]))
                    # record the result for the corresponding loopy array
                    state.results[named_array] = result

                    new_tvs[assignee_name] = get_loopy_temporary(assignee_name,
                                                                 named_array,
                                                                 self, state)
                else:
                    assert arg.is_input
                    pt_arg = expr.bindings[arg.name]
                    assert isinstance(pt_arg, Array)

                    pt_arg_rec = self.rec(pt_arg, state)

                    if isinstance(pt_arg_rec, StoredResult):
                        # found a stored result corresponding to the argument, use it
                        name = pt_arg_rec.name
                        params.append(_get_sub_array_ref(pt_arg, name))
                        depends_on.update(pt_arg_rec.depends_on)
                    else:
                        # did not find a stored result for the sub-expression, store
                        # it and then pass it to the call
                        name = _generate_name_for_temp(pt_arg, state.var_name_gen)
                        store_insn_id = add_store(name, pt_arg,
                                pt_arg_rec,
                                state, output_to_temporary=True,
                                cgen_mapper=self)
                        depends_on.add(store_insn_id)
                        # replace "arg" with the created stored variable
                        state.results[pt_arg] = StoredResult(name, pt_arg.ndim,
                                                          frozenset([store_insn_id]))
                        params.append(_get_sub_array_ref(pt_arg, name))
                        new_tvs[name] = get_loopy_temporary(name, pt_arg,
                                                            self, state)
            else:
                assert isinstance(arg, lp.ValueArg) and arg.is_input
                pt_arg = expr.bindings[arg.name]
                prstnt_ctx = PersistentExpressionContext(state)

                if isinstance(pt_arg, Array):
                    assert pt_arg.ndim == 0
                    pt_arg_rec = self.rec(pt_arg, state)
                    params.append(pt_arg_rec.to_loopy_expression((), prstnt_ctx))
                    depends_on.update(prstnt_ctx.depends_on)
                else:
                    local_ctx = LocalExpressionContext(reduction_bounds={},
                                                       num_indices=0,
                                                       local_namespace={},
                                                       var_to_reduction_descr={})
                    params.append(self.exprgen_mapper(pt_arg,
                                                      prstnt_ctx,
                                                      local_ctx))

        new_insn = make_assignment(
                tuple(assignees),
                var(expr.entrypoint)(*params),
                depends_on=frozenset(depends_on),
                id=new_insn_id)

        # update kernel
        kernel = state.kernel
        tvs = dict(state.kernel.temporary_variables)
        tvs.update(new_tvs)

        kernel = kernel.copy(instructions=[*kernel.instructions, new_insn],
                             temporary_variables=tvs,
                             domains=[*kernel.domains, *domains])

        state.update_kernel(kernel)

    def map_named_call_result(self, expr: NamedCallResult,
                              state: CodeGenState) -> None:
        raise NotImplementedError("LoopyTarget does not support outlined calls"
                                  " (yet). As a fallback, the call"
                                  " could be inlined using"
                                  " pt.tag_all_calls_to_be_inlined.")

    def map_call(self, expr: Call, state: CodeGenState) -> None:
        raise NotImplementedError("LoopyTarget does not support outlined calls"
                                  " (yet). As a fallback, the call"
                                  " could be inlined using"
                                  " pt.tag_all_calls_to_be_inlined.")

# }}}


# {{{ inlined expression gen mapper

ELWISE_INDEX_RE = re.compile(r"_(0|([1-9][0-9]*))")
REDUCTION_INDEX_RE = re.compile(r"_r(0|([1-9][0-9]*))")

# Maps Pytato reduction types to the corresponding Loopy reduction types.
PYTATO_REDUCTION_TO_LOOPY_REDUCTION: Mapping[type[red.ReductionOperation], str] = {
    red.SumReductionOperation: "sum",
    red.ProductReductionOperation: "product",
    red.MaxReductionOperation: "max",
    red.MinReductionOperation: "min",
    red.AllReductionOperation: "all",
    red.AnyReductionOperation: "any",
}


class InlinedExpressionGenMapper(
            scalar_expr.IdentityMapper[
                [PersistentExpressionContext, LocalExpressionContext]]):
    """A mapper for generating :mod:`loopy` expressions with inlined
    sub-expressions.

    The inputs to this mapper are scalar expression as found in
    :class:`pytato.array.IndexLambda`, or expressions that are
    compatible (e.g., shape expressions).

    The outputs of this mapper are scalar expressions suitable for wrapping in
    :class:`InlinedResult`.
    """
    axis_tag_t_to_not_propagate: frozenset[type[Tag]]

    def __init__(self, axis_tag_t_to_not_propagate: frozenset[type[Tag]]) -> None:
        self.axis_tag_t_to_not_propagate = axis_tag_t_to_not_propagate

    def map_subscript(self, expr: prim.Subscript,
                      prstnt_ctx: PersistentExpressionContext,
                      local_ctx: LocalExpressionContext,
                      ) -> ScalarExpression:
        assert isinstance(expr.aggregate, prim.Variable)
        rec_index = self.rec(expr.index, prstnt_ctx, local_ctx)
        assert is_symbolic_index(rec_index)
        res = local_ctx.lookup(expr.aggregate.name).to_loopy_expression(
            rec_index, prstnt_ctx)
        assert prim.is_arithmetic_expression(res)
        return res

    def map_variable(self, expr: prim.Variable,
                     prstnt_ctx: PersistentExpressionContext,
                     local_ctx: LocalExpressionContext,
                     ) -> Expression:

        elw_match = ELWISE_INDEX_RE.fullmatch(expr.name)
        if elw_match:
            # Found an index of the form _0, _1, ...
            index = int(elw_match.group(1))
            if not (0 <= index < local_ctx.num_indices):
                raise ValueError(f"invalid index encountered: _{index}")
            return expr
        elif expr.name in local_ctx.reduction_bounds:
            return expr
        else:
            res = local_ctx.lookup(expr.name).to_loopy_expression((), prstnt_ctx)
            assert prim.is_arithmetic_expression(res)
            return res

    def map_call(self, expr: prim.Call,
                 prstnt_ctx: PersistentExpressionContext,
                 local_ctx: LocalExpressionContext
                 ) -> Expression:
        if isinstance(expr.function, prim.Variable) and (
                expr.function.name.startswith("pytato.c99.")):
            name_in_loopy = expr.function.name[11:]
            pars = self.rec(expr.parameters, prstnt_ctx, local_ctx)
            assert isinstance(pars, tuple)
            return prim.Call(prim.Variable(name_in_loopy), pars)

        if isinstance(expr.function, prim.Variable) and (
                expr.function.name == "pytato.zero"):
            # traversing self.rec for the bindings is not needed.
            return 0

        return super().map_call(expr, prstnt_ctx, local_ctx)

    def map_reduce(self, expr: scalar_expr.Reduce,
                   prstnt_ctx: PersistentExpressionContext,
                   local_ctx: LocalExpressionContext
                   ) -> ScalarExpression:
        from loopy.symbolic import Reduction as LoopyReduction
        state = prstnt_ctx.state

        try:
            loopy_redn = PYTATO_REDUCTION_TO_LOOPY_REDUCTION[type(expr.op)]
        except KeyError as err:
            raise NotImplementedError(expr.op) from err

        unique_names_mapping = {
                old_name: state.var_name_gen(f"_pt_{loopy_redn}" + old_name)
                for old_name in expr.bounds}

        inner_expr = loopy_substitute(expr.inner_expr,
                                      {k: prim.Variable(v)
                                       for k, v in unique_names_mapping.items()})
        new_bounds = {unique_names_mapping[name]: bound_exprs
                      for name, bound_exprs in expr.bounds.items()}

        inner_expr = self.rec(inner_expr, prstnt_ctx,
                              local_ctx.copy(reduction_bounds=new_bounds))

        inner_expr = LoopyReduction(loopy_redn,
                                    tuple(unique_names_mapping.values()),
                                    inner_expr)

        domain = domain_for_shape((), shape=(), reductions={
            redn_iname: (
                self.rec_arith(lbound, prstnt_ctx, local_ctx),
                self.rec_arith(ubound, prstnt_ctx, local_ctx),
                )
            for redn_iname, (lbound, ubound) in new_bounds.items()})
        kernel = state.kernel
        state.update_kernel(kernel.copy(domains=[*kernel.domains, domain]))

        # {{{ pytato tags -> loopy tags

        for name_in_expr, name_in_kernel in sorted(unique_names_mapping.items()):
            for tag in local_ctx.var_to_reduction_descr[name_in_expr].tags:
                if all(not isinstance(tag, tag_t)
                       for tag_t in self.axis_tag_t_to_not_propagate):
                    state.update_kernel(lp.tag_inames(state.kernel,
                                                      {name_in_kernel: tag}))

        # }}}

        return inner_expr

    def map_type_cast(
                self, expr: TypeCast,
                prstnt_ctx: PersistentExpressionContext,
                local_ctx: LocalExpressionContext,
            ) -> ScalarExpression:
        return lp.TypeCast(
                    lp.to_loopy_type(expr.dtype),
                    self.rec(expr.inner_expr, prstnt_ctx, local_ctx))

# }}}


# {{{ utils

def shape_to_scalar_expression(shape: ShapeType,
                               cgen_mapper: CodeGenMapper,
                               state: CodeGenState
                               ) -> tuple[ArithmeticExpression, ...]:
    shape_context = PersistentExpressionContext(state)
    result: list[ArithmeticExpression] = []
    for component in shape:
        if isinstance(component, INT_CLASSES):
            result.append(component)
        else:
            assert isinstance(component, Array)
            expr = cgen_mapper(component, state).to_loopy_expression((), shape_context)
            assert prim.is_arithmetic_expression(expr)
            result.append(expr)

    assert not shape_context.depends_on

    return tuple(result)


def domain_for_shape(dim_names: tuple[str, ...],
         shape: tuple[ScalarExpression, ...],
         reductions: dict[str, tuple[ScalarExpression, ScalarExpression]],
         ) -> isl.BasicSet:
    """Create an :class:`islpy.BasicSet` that expresses an appropriate index domain
    for an array of (potentially symbolic) shape *shape* having reduction
    dimensions *reductions*.

    :param dim_names: A tuple of strings, the names of the axes. These become set
        dimensions in the returned domain.

    :param shape: A tuple of constant or quasi-affine :mod:`pymbolic`
        expressions. The variables in these expressions become parameter
        dimensions in the returned set.  Must have the same length as
        *dim_names*.

    :arg reductions: A map from reduction inames to (lower, upper) bounds
        (as half-open integer ranges). The variables in the bounds become
        parameter dimensions in the returned set.
    """
    assert len(dim_names) == len(shape)

    # Collect parameters.
    param_names_set: set[str] = set()
    for sdep in map(scalar_expr.get_dependencies, shape):
        param_names_set |= sdep

    for bounds in reductions.values():
        for sdep in map(scalar_expr.get_dependencies, bounds):
            # FIXME: Assumes that reduction bounds are not data-dependent.
            param_names_set |= sdep

    set_names = sorted(tuple(dim_names) + tuple(reductions))
    param_names = sorted(param_names_set)

    # Build domain.
    dom = isl.BasicSet.universe(
            isl.Space.create_from_names(isl.DEFAULT_CONTEXT,
            set=set_names,
            params=param_names))

    # Add constraints.
    from loopy.symbolic import aff_from_expr
    affs = isl.affs_from_space(dom.space)

    for iname, dim in zip(dim_names, shape, strict=True):
        dom &= affs[0].le_set(affs[iname])
        dom &= affs[iname].lt_set(aff_from_expr(dom.space, dim))

    for iname, (left, right) in reductions.items():
        dom &= aff_from_expr(dom.space, left).le_set(affs[iname])
        dom &= affs[iname].lt_set(aff_from_expr(dom.space, right))

    doms = dom.get_basic_sets()

    if len(doms) == 0:
        # empty set
        dom = isl.BasicSet.empty(dom.get_space())
    else:
        dom, = doms

    return dom


def _filter_tags_not_of_type(expr: Array,
                             ignore_tag_t: frozenset[type[Tag]]
                             ) -> frozenset[Tag]:
    return frozenset(tag
                     for tag in expr.tags
                     if not isinstance(tag, tuple(ignore_tag_t)))


def add_store(name: str, expr: Array, result: ImplementedResult,
              state: CodeGenState, cgen_mapper: CodeGenMapper,
              output_to_temporary: bool = False) -> str:
    """Add an instruction that stores to a variable in the kernel.

    :param name: name of the output array, which is created
    :param expr: the :class:`~pytato.Array` to store
    :param result: the corresponding :class:`ImplementedResult`
    :param state: code generation state
    :param output_to_temporary: whether to generate an output argument (default)
        or a temporary variable

    :returns: the id of the generated instruction
    """
    # Get expression.
    inames = tuple(
            state.var_name_gen(f"{name}_dim{d}")
            for d in range(expr.ndim))
    indices = tuple(prim.Variable(iname) for iname in inames)
    loopy_expr_context = PersistentExpressionContext(state)
    loopy_expr = result.to_loopy_expression(indices, loopy_expr_context)

    # Make the instruction
    from loopy.kernel.instruction import make_assignment
    assignee = prim.Variable(name)[indices] if indices else prim.Variable(name)
    insn_id = state.insn_id_gen(f"{name}_store")
    insn = make_assignment((assignee,),
            loopy_expr,
            id=insn_id,
            within_inames=frozenset(inames),
            depends_on=loopy_expr_context.depends_on)
    shape = shape_to_scalar_expression(expr.shape, cgen_mapper, state)

    # Get the domain.
    domain = domain_for_shape(inames, shape, {})

    from pytato.utils import are_shape_components_equal
    result_is_empty = any(are_shape_components_equal(s_i, 0) for s_i in expr.shape)
    if result_is_empty:
        # empty array, no need to do computation
        additional_domains = []
        additional_insns = []
    else:
        additional_domains = [domain]
        additional_insns = [insn]

    # Update the kernel.
    kernel = state.kernel

    if output_to_temporary:
        tvar = get_loopy_temporary(name, expr, cgen_mapper, state)
        temporary_variables = dict(kernel.temporary_variables)
        temporary_variables[name] = tvar
        kernel = kernel.copy(temporary_variables=temporary_variables,
                domains=[*kernel.domains, *additional_domains],
                instructions=[*kernel.instructions, *additional_insns])
    else:
        arg = lp.GlobalArg(name,
                shape=shape,
                dtype=expr.dtype,
                order="C",
                is_input=False,
                is_output=True,
                tags=_filter_tags_not_of_type(expr,
                                              cgen_mapper
                                              .array_tag_t_to_not_propagate))
        kernel = kernel.copy(args=[*kernel.args, arg],
                domains=[*kernel.domains, *additional_domains],
                instructions=[*kernel.instructions, *additional_insns])

    # {{{ axes tags -> iname tags

    if not result_is_empty:
        for axis, iname in zip(expr.axes, inames, strict=True):
            for tag in axis.tags:
                if all(not isinstance(tag, tag_t)
                       for tag_t in cgen_mapper.axis_tag_t_to_not_propagate):
                    kernel = lp.tag_inames(kernel, {iname: tag})

    # }}}

    state.update_kernel(kernel)
    return insn_id


def add_substitution(subst_name: str, expr: Array, result: ImplementedResult,
                     state: CodeGenState, cgen_mapper: CodeGenMapper) -> None:
    """Add a :class:`~loopy.kernel.data.SubstitutionRule` to the kernel being built
    in *state*. The substitution rule that will be introduced with take the indices
    of array expression *expr*'s as arguments and return the value for the index.
    """
    # Get expression.
    indices = tuple(prim.Variable(f"_{idim}") for idim in range(expr.ndim))
    loopy_expr_context = PersistentExpressionContext(state)
    loopy_expr = result.to_loopy_expression(indices, loopy_expr_context)

    # Make the substitution rule
    subst_rule = lp.SubstitutionRule(subst_name,
                                     tuple(f"_{idim}" for idim in range(expr.ndim)),
                                     loopy_expr)

    # Update the kernel.
    kernel = state.kernel
    kernel = kernel.copy(
        substitutions={**kernel.substitutions,
                       **{subst_name: subst_rule}})

    state.update_kernel(kernel)


def get_loopy_temporary(name: str, expr: Array, cgen_mapper: CodeGenMapper,
                        state: CodeGenState) -> lp.TemporaryVariable:
    # always allocating to global address space to avoid stack overflow
    address_space = lp.AddressSpace.GLOBAL
    return lp.TemporaryVariable(name,
            shape=shape_to_scalar_expression(expr.shape, cgen_mapper, state),
            dtype=expr.dtype,
            address_space=address_space,
            tags=_filter_tags_not_of_type(expr,
                                          cgen_mapper
                                          .array_tag_t_to_not_propagate))

# }}}


def get_initial_codegen_state(target: LoopyTarget,
        options: lp.Options,
        function_name: str) -> CodeGenState:
    kernel = lp.make_kernel("{:}", [],
            name=function_name,
            target=target.get_loopy_target(),
            options=options,
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION)

    return CodeGenState(_t_unit=kernel, results={})


# {{{ generate_loopy

def generate_loopy(result: Array | AbstractResultWithNamedArrays | dict[str, Array],
                   target: LoopyTarget | None = None,
                   options: lp.Options | None = None,
                   *,
                   function_name: str = "_pt_kernel",
                   cl_device: pyopencl.Device | None = None,
                   array_tag_t_to_not_propagate: frozenset[type[Tag]] = frozenset([
                       ImplStored, Named, PrefixNamed]),
                   axis_tag_t_to_not_propagate: frozenset[type[Tag]] = frozenset(),
                   ) -> BoundProgram:
    r"""Code generation entry point.

    :param result: Outputs of the computation.
    :param target: Code generation target.
    :param options: Code generation options for the kernel.
    :returns: A :class:`pytato.target.BoundProgram` wrapping the generated
        :class:`loopy.TranslationUnit`.

    If *result* is a :class:`dict` or a :class:`pytato.DictOfNamedArrays` and
    *options* is not supplied, then the Loopy option
    :attr:`~loopy.Options.return_dict` will be set to *True*. If it is supplied,
    :attr:`~loopy.Options.return_dict` must already be set to *True*.

    .. note::

        - :mod:`pytato` metadata :math:`\mapsto` :mod:`loopy` metadata semantics:

            - Inames that index over an :class:`~pytato.array.Array`'s axis in the
              allocation instruction are tagged with the corresponding
              :class:`~pytato.array.Axis`'s tags. The caller may choose to not
              propagate axis tags of type *axis_tag_t_to_not_propagate*.
            - :attr:`pytato.Array.tags` of inputs/outputs in *outputs*
              would be copied over to the tags of the corresponding
              :class:`loopy.ArrayArg`. The caller may choose to not
              propagate array tags of type *array_tag_t_to_not_propagate*.
            - Arrays tagged with :class:`pytato.tags.ImplStored` would have their
              tags copied over to the tags of corresponding
              :class:`loopy.TemporaryVariable`. The caller may choose to not
              propagate array tags of type *array_tag_t_to_not_propagate*.

    .. warning::

        Currently only :class:`~pytato.function.Call` nodes that are tagged with
        :class:`pytato.tags.InlineCallTag` can be lowered to :mod:`loopy` IR.
    """

    result_is_dict = isinstance(result, dict | DictOfNamedArrays)
    orig_outputs: AbstractResultWithNamedArrays = normalize_outputs(result)

    if not isinstance(orig_outputs, DictOfNamedArrays):
        raise NotImplementedError(
            f"not implemented for {type(result).__name__}.")

    del result

    if cl_device is not None:
        from warnings import warn
        warn("Passing 'cl_device' is deprecated. This will stop working in 2023.",
                DeprecationWarning, stacklevel=2)

    if target is None:
        target = LoopyPyOpenCLTarget()

    assert isinstance(target, LoopyTarget)

    preproc_result = preprocess(orig_outputs, target)
    outputs = preproc_result.outputs

    # optimization: remove any ImplStored tags on outputs to avoid redundant
    # store-load operations (see https://github.com/inducer/pytato/issues/415)
    # (This must be done after all the calls have been inlined)
    outputs = DictOfNamedArrays(
        {name: (output.without_tags(ImplStored(),
                                    verify_existence=False)
                if not isinstance(output,
                                  InputArgumentBase)
                else output)
         for name, output in outputs._data.items()},
        tags=outputs.tags)

    compute_order = preproc_result.compute_order

    if options is None:
        options = lp.Options(return_dict=result_is_dict)
    if options.return_dict != result_is_dict:
        raise ValueError("options.return_dict is expected to match "
                "whether the returned value is a dictionary")

    state = get_initial_codegen_state(target, options, function_name=function_name)

    from pytato.transform import InputGatherer
    ing = InputGatherer()

    state.var_name_gen.add_names({input_expr.name
            for name in compute_order
            for input_expr in ing(outputs[name].expr)
            if isinstance(input_expr, Placeholder | SizeParam | DataWrapper)
            if input_expr.name is not None})

    state.var_name_gen.add_names(outputs)

    cg_mapper = CodeGenMapper(array_tag_t_to_not_propagate,
                              axis_tag_t_to_not_propagate)

    # Generate code for outputs.
    for name in compute_order:
        expr = outputs[name].expr
        insn_id = add_store(name, expr, cg_mapper(expr, state), state, cg_mapper)
        # replace "expr" with the created stored variable
        state.results[expr] = StoredResult(name, expr.ndim, frozenset([insn_id]))

    # Why call make_reduction_inames_unique?
    # Consider pt.generate_loopy(pt.sum(x) + pt.sum(x)), the generated
    # translation unit would be a single instruction with rhs: `_pt_subst() +
    # _pt_subst()`.  The result of pt.sum(x) is cached => same instance of
    # InlinedResult is emitted for both invocations and we would be required to
    # avoid such reduction iname collisions.
    t_unit = lp.make_reduction_inames_unique(state.t_unit)

    # Disable bounds checking if there is no hand-written LoopyCall in the DAG.
    if not cg_mapper.has_loopy_call:
        t_unit = lp.set_options(t_unit,
                                enforce_array_accesses_within_bounds="no_check")

    return target.bind_program(
            program=t_unit,
            bound_arguments=preproc_result.bound_arguments)

# }}}

# vim:fdm=marker
