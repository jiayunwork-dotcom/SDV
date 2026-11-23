"""ParentChildSum constraint."""

import math

import numpy as np
import pandas as pd

from sdv.cag._errors import ConstraintNotMetError
from sdv.cag.base import BaseConstraint
from sdv.errors import SamplingError


class _InfeasibleParentChildSum(SamplingError):
    """Internal error raised when a sampled group cannot satisfy the constraint."""


def _is_non_bool_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_non_bool_non_negative_int(value):
    return _is_non_bool_int(value) and value >= 0


class ParentChildSum(BaseConstraint):
    """Ensure that a parent total column is the exact sum of a child value column.

    For every parent row, the value of ``parent_total_column`` must equal the sum of the
    ``child_value_column`` values across the child rows whose foreign key points to that
    parent's primary key. Parent rows without any child rows must therefore have a total
    of ``0``. Both amount columns are treated as non-nullable integer amounts
    (for example integer cents).

    The constraint only supports metadata where the parent and child table are connected
    by exactly one direct primary-key / foreign-key relationship, the foreign key is
    non-nullable and neither amount column is a relationship key.

    Args:
        parent_table_name (str):
            Name of the parent table.
        child_table_name (str):
            Name of the child table.
        parent_total_column (str):
            Name of the non-nullable integer total column in the parent table.
        child_value_column (str):
            Name of the non-nullable integer value column in the child table.
        child_min (int, optional):
            Inclusive lower bound for every child value. Must be a non-boolean,
            non-negative integer. Defaults to ``0``.
        child_max (int, optional):
            Inclusive upper bound for every child value. If provided, must be a
            non-boolean integer greater than or equal to ``child_min``. Defaults to
            ``None``.
    """

    _is_single_table = False

    @staticmethod
    def _validate_init_inputs(
        parent_table_name,
        child_table_name,
        parent_total_column,
        child_value_column,
        child_min,
        child_max,
    ):
        string_inputs = {
            'parent_table_name': parent_table_name,
            'child_table_name': child_table_name,
            'parent_total_column': parent_total_column,
            'child_value_column': child_value_column,
        }
        for name, value in string_inputs.items():
            if not isinstance(value, str):
                raise ValueError(f'`{name}` must be a string.')

        if not _is_non_bool_non_negative_int(child_min):
            raise ValueError('`child_min` must be a non-boolean, non-negative integer.')

        if child_max is not None:
            if not _is_non_bool_int(child_max):
                raise ValueError('`child_max` must be an integer or None.')
            if child_max < child_min:
                raise ValueError('`child_max` must be greater than or equal to `child_min`.')

    def __init__(
        self,
        parent_table_name,
        child_table_name,
        parent_total_column,
        child_value_column,
        *,
        child_min=0,
        child_max=None,
    ):
        super().__init__()
        self._validate_init_inputs(
            parent_table_name,
            child_table_name,
            parent_total_column,
            child_value_column,
            child_min,
            child_max,
        )
        self.parent_table_name = parent_table_name
        self.child_table_name = child_table_name
        self.parent_total_column = parent_total_column
        self.child_value_column = child_value_column
        self.child_min = child_min
        self.child_max = child_max
        self.table_name = None

        # Populated during ``_validate_constraint_with_metadata``.
        self._foreign_key = None
        self._parent_primary_key = None

    # ------------------------------------------------------------------
    # Metadata validation
    # ------------------------------------------------------------------
    def _validate_tables_exist(self, metadata):
        for table_name in (self.parent_table_name, self.child_table_name):
            if table_name not in metadata.tables:
                raise ConstraintNotMetError(f"Table '{table_name}' missing from metadata.")

    def _validate_amount_columns(self, metadata):
        pairs = [
            (self.parent_table_name, self.parent_total_column),
            (self.child_table_name, self.child_value_column),
        ]
        for table_name, column_name in pairs:
            columns = metadata.tables[table_name].columns
            if column_name not in columns:
                raise ConstraintNotMetError(
                    f"Table '{table_name}' is missing column '{column_name}'."
                )

    def _validate_amount_column_sdtypes(self, metadata):
        pairs = [
            (self.parent_table_name, self.parent_total_column),
            (self.child_table_name, self.child_value_column),
        ]
        for table_name, column_name in pairs:
            column_metadata = metadata.tables[table_name].columns[column_name]
            sdtype = column_metadata.get('sdtype')
            if sdtype != 'numerical':
                raise ConstraintNotMetError(
                    f"Column '{column_name}' in table '{table_name}' has an incompatible "
                    f"sdtype ('{sdtype}'). The column sdtype must be 'numerical'."
                )

    def _validate_relationship(self, metadata):
        matching_relationships = [
            relationship
            for relationship in metadata.relationships
            if relationship['parent_table_name'] == self.parent_table_name
            and relationship['child_table_name'] == self.child_table_name
        ]
        if len(matching_relationships) == 0:
            raise ConstraintNotMetError(
                f'There is no direct relationship between parent table '
                f"'{self.parent_table_name}' and child table '{self.child_table_name}'."
            )
        if len(matching_relationships) > 1:
            raise ConstraintNotMetError(
                f'Expected exactly one direct relationship between parent table '
                f"'{self.parent_table_name}' and child table '{self.child_table_name}' but "
                f'found {len(matching_relationships)}.'
            )

        relationship = matching_relationships[0]
        self._foreign_key = relationship['child_foreign_key']
        self._parent_primary_key = relationship['parent_primary_key']
        return relationship

    def _validate_acyclic_tree(self, metadata):
        parent_map = metadata._get_parent_map()

        visiting = set()
        visited = set()

        def visit(table_name):
            if table_name in visiting:
                raise ConstraintNotMetError(
                    'ParentChildSum only supports acyclic metadata but a circular dependency '
                    f"was detected while traversing table '{table_name}'."
                )
            if table_name in visited:
                return

            visiting.add(table_name)
            for parent_name in parent_map.get(table_name, set()):
                visit(parent_name)

            visiting.discard(table_name)
            visited.add(table_name)

        visit(self.child_table_name)

    def _validate_foreign_key_not_null(self, metadata):
        """Check the foreign key column is declared as a non-PII id column.

        This metadata version has no explicit ``nullable`` flag, so foreign key
        nullability is actually enforced against the data in
        ``_validate_child_foreign_keys`` at fit time. Here we only make sure the
        column keeps the expected id sdtype.
        """
        foreign_key_metadata = metadata.tables[self.child_table_name].columns[self._foreign_key]
        if foreign_key_metadata.get('sdtype') != 'id':
            raise ConstraintNotMetError(
                f"The foreign key column '{self._foreign_key}' in table "
                f"'{self.child_table_name}' must have sdtype 'id'."
            )

    def _validate_columns_are_not_keys(self, metadata):
        parent_keys = metadata._get_all_keys(self.parent_table_name)
        if self.parent_total_column in parent_keys:
            raise ConstraintNotMetError(
                f"The parent total column '{self.parent_total_column}' in table "
                f"'{self.parent_table_name}' must not be a relationship key."
            )

        child_keys = metadata._get_all_keys(self.child_table_name)
        if self.child_value_column in child_keys:
            raise ConstraintNotMetError(
                f"The child value column '{self.child_value_column}' in table "
                f"'{self.child_table_name}' must not be a relationship key."
            )

    def _validate_constraint_with_metadata(self, metadata):
        self._validate_tables_exist(metadata)
        self._validate_amount_columns(metadata)
        self._validate_relationship(metadata)
        self._validate_acyclic_tree(metadata)
        self._validate_foreign_key_not_null(metadata)
        self._validate_columns_are_not_keys(metadata)
        self._validate_amount_column_sdtypes(metadata)

    # ------------------------------------------------------------------
    # Data validation (fit time accounting)
    # ------------------------------------------------------------------
    @staticmethod
    def _raise_data_error(message):
        raise ConstraintNotMetError(message)

    def _validate_parent_primary_key_unique(self, data):
        parent_data = data[self.parent_table_name]
        primary_key = self._parent_primary_key
        duplicated = parent_data[primary_key].dropna().duplicated(keep=False)
        if duplicated.any():
            duplicate_keys = parent_data.loc[duplicated, primary_key].unique().tolist()
            self._raise_data_error(
                f"Primary key column '{primary_key}' in table '{self.parent_table_name}' "
                f'contains duplicated values: {duplicate_keys[:5]}.'
            )

        if parent_data[primary_key].isna().any():
            self._raise_data_error(
                f"Primary key column '{primary_key}' in table "
                f"'{self.parent_table_name}' contains null values."
            )

    def _validate_child_foreign_keys(self, data):
        parent_data = data[self.parent_table_name]
        child_data = data[self.child_table_name]
        foreign_key = self._foreign_key

        null_fk = child_data[foreign_key].isna()
        if null_fk.any():
            bad_indices = child_data.index[null_fk].tolist()
            self._raise_data_error(
                f"Foreign key column '{foreign_key}' in table '{self.child_table_name}' "
                f'contains null values at row indices: {bad_indices[:5]}.'
            )

        parent_keys = set(parent_data[self._parent_primary_key])
        orphan_mask = ~child_data[foreign_key].isin(parent_keys)
        if orphan_mask.any():
            orphan_keys = child_data.loc[orphan_mask, foreign_key].unique().tolist()
            self._raise_data_error(
                f"Foreign key column '{foreign_key}' in table '{self.child_table_name}' "
                f'references parent keys that do not exist in table '
                f"'{self.parent_table_name}': {orphan_keys[:5]}."
            )

    def _validate_amount_values(self, data):
        specs = [
            (self.parent_table_name, self.parent_total_column),
            (self.child_table_name, self.child_value_column),
        ]
        for table_name, column_name in specs:
            column = data[table_name][column_name]
            null_mask = column.isna()
            if null_mask.any():
                bad_indices = column.index[null_mask].tolist()
                self._raise_data_error(
                    f"Amount column '{column_name}' in table '{table_name}' contains null "
                    f'values at row indices: {bad_indices[:5]}.'
                )

            # Booleans are a subclass of int; reject them explicitly. The helper
            # ``_is_non_bool_int`` returns False for bool and any non-integer value
            # (including numpy floats and decimal values).
            not_integer = [index for index, value in column.items() if not _is_non_bool_int(value)]
            if not_integer:
                bad_values = column.loc[not_integer].unique().tolist()
                if any(isinstance(value, bool) for value in bad_values):
                    self._raise_data_error(
                        f"Amount column '{column_name}' in table '{table_name}' must "
                        'contain integer amounts, not booleans.'
                    )

                self._raise_data_error(
                    f"Amount column '{column_name}' in table '{table_name}' must contain "
                    f'only non-null integer values. Non-integer values found at row indices '
                    f'{not_integer[:5]}: {bad_values[:5]}.'
                )

    def _validate_child_bounds_and_totals(self, data):
        parent_data = data[self.parent_table_name]
        child_data = data[self.child_table_name]
        foreign_key = self._foreign_key
        child_column = self.child_value_column
        parent_column = self.parent_total_column

        out_of_bounds = pd.Series(False, index=child_data.index)
        lower_violation = child_data[child_column] < self.child_min
        out_of_bounds = out_of_bounds | lower_violation
        if self.child_max is not None:
            out_of_bounds = out_of_bounds | (child_data[child_column] > self.child_max)

        if out_of_bounds.any():
            bad_rows = child_data.loc[out_of_bounds, [foreign_key, child_column]]
            self._raise_data_error(
                f"Child values in table '{self.child_table_name}' column '{child_column}' "
                f'must be within the declared bounds '
                f'[{self.child_min}, {self.child_max}]:\n{bad_rows.head(5).to_string()}'
            )

        grouped_sum = child_data.groupby(foreign_key)[child_column].sum()
        parent_totals = parent_data.set_index(self._parent_primary_key)[parent_column]
        aligned_sums = grouped_sum.reindex(parent_totals.index, fill_value=0)

        mismatches = aligned_sums != parent_totals
        if mismatches.any():
            details = pd.DataFrame({
                'parent_key': parent_totals.index[mismatches],
                'declared_total': parent_totals[mismatches].to_numpy(),
                'sum_of_children': aligned_sums[mismatches].to_numpy(),
            })
            self._raise_data_error(
                f"The parent total column '{parent_column}' in table "
                f"'{self.parent_table_name}' does not match the sum of child values from "
                f"table '{self.child_table_name}' column '{child_column}':\n"
                f'{details.head(5).to_string(index=False)}'
            )

    def _validate_constraint_with_data(self, data, metadata):
        self._validate_parent_primary_key_unique(data)
        self._validate_child_foreign_keys(data)
        self._validate_amount_values(data)
        self._validate_child_bounds_and_totals(data)

    # ------------------------------------------------------------------
    # No-op transform: the relationship is enforced while sampling.
    # ------------------------------------------------------------------
    def _get_updated_metadata(self, metadata):
        return metadata

    def _fit(self, data, metadata):
        # Nothing extra to learn: all parameters come from the declaration.
        return

    def _transform(self, data):
        return data

    def _reverse_transform(self, data):
        return data

    def _is_valid(self, data, metadata):
        parent_data = data[self.parent_table_name]
        child_data = data[self.child_table_name]
        is_valid = {
            table: pd.Series(True, index=table_data.index) for table, table_data in data.items()
        }

        foreign_key = self._foreign_key
        grouped_sum = child_data.groupby(foreign_key)[self.child_value_column].sum()
        parent_totals = parent_data.set_index(self._parent_primary_key)[self.parent_total_column]
        aligned_sums = grouped_sum.reindex(parent_totals.index, fill_value=0)
        valid_parent_keys = set(parent_totals.index[aligned_sums == parent_totals])

        parent_valid = parent_data[self._parent_primary_key].isin(valid_parent_keys)
        is_valid[self.parent_table_name] = parent_valid.reset_index(drop=True)
        is_valid[self.child_table_name] = (
            child_data[foreign_key].isin(valid_parent_keys).reset_index(drop=True)
        )
        return is_valid

    # ------------------------------------------------------------------
    # Sampling-time enforcement
    # ------------------------------------------------------------------
    @staticmethod
    def _allocate_integer_values(num_rows, target, lower_bound, upper_bound, weights=None):
        """Split the exact integer ``target`` over ``num_rows`` bounded slots.

        The split follows a greedy largest-remainder (Hamilton) allocation driven by
        ``weights``, clipped to the inclusive ``[lower_bound, upper_bound]`` bounds
        through iterative water-filling: slots that hit a bound freeze and the still
        active slots re-share the residual. This keeps each row close to its sampled
        share of the group total instead of dumping the difference onto a single row.
        Every returned value is an exact Python integer and the values always sum to
        ``target``; a ``_InfeasibleParentChildSum`` is raised when no allocation
        exists.

        Args:
            num_rows (int):
                Number of child rows in the parent group.
            target (int):
                Exact total that the returned values must sum to.
            lower_bound (int):
                Inclusive per-row lower bound.
            upper_bound (int or float):
                Inclusive per-row upper bound (``math.inf`` is allowed).
            weights (numpy.ndarray, optional):
                Non-negative weights of length ``num_rows`` used to preserve the
                within-group shape of the sampled values. If ``None``, equal weights
                are used.
        """
        if num_rows <= 0:
            if target == 0:
                return np.array([], dtype=object)

            raise _InfeasibleParentChildSum('Cannot allocate a non-zero target across zero rows.')

        minimum_total = lower_bound * num_rows
        maximum_total = upper_bound * num_rows
        if target < minimum_total or target > maximum_total:
            raise _InfeasibleParentChildSum(
                f'Target {target} cannot be split into {num_rows} values within '
                f'[{lower_bound}, {upper_bound}].'
            )

        if weights is None:
            weights = np.ones(num_rows, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)
            weights = np.where(weights < 0, 0.0, weights)

        values = [int(lower_bound)] * num_rows
        remaining = target - minimum_total
        active = np.ones(num_rows, dtype=bool)
        capacity = upper_bound - lower_bound

        # Weighted, bounded largest-remainder allocation with water-filling.
        while remaining > 0:
            active_indices = np.flatnonzero(active)
            active_weights = weights[active_indices]
            weight_total = active_weights.sum()
            if weight_total <= 0:
                # Degenerate group (e.g. every sampled value was 0): split evenly.
                active_weights = np.ones(active_indices.size)
                weight_total = active_indices.size

            exact_shares = remaining * active_weights / weight_total
            floors = np.floor(exact_shares).astype(np.int64)
            if capacity != math.inf:
                floors = np.minimum(floors, int(capacity))

            for index, floor_value in zip(active_indices, floors):
                values[int(index)] += int(floor_value)
            remaining -= int(floors.sum())

            if capacity != math.inf:
                saturated = np.array([value >= upper_bound for value in values], dtype=bool)
                newly_frozen = saturated & active
                active = active & ~saturated
                if newly_frozen.any() and remaining > 0 and not active.any():
                    raise _InfeasibleParentChildSum(
                        f'Target {target} cannot be split into {num_rows} values '
                        f'within [{lower_bound}, {upper_bound}].'
                    )

            if remaining == 0:
                break

            if floors.sum() == 0:
                # All active shares are below 1: distribute the leftover units to the
                # slots with the largest fractional remainders, skipping saturated
                # slots when a finite bound applies.
                fractional = exact_shares - np.floor(exact_shares)
                if capacity != math.inf:
                    room = np.array([
                        upper_bound - values[int(index)] > 0 for index in active_indices
                    ])
                    fractional = fractional[room]
                    active_indices = active_indices[room]

                order = np.argsort(-fractional, kind='stable')
                for index in active_indices[order[:remaining]]:
                    values[int(index)] += 1
                remaining = 0

        return np.array(values, dtype=object)

    def _table_depth(self, metadata):
        """Depth of the child table in the metadata tree (roots have depth 0)."""
        parent_map = metadata._get_parent_map()
        memo = {}

        def depth(table_name):
            if table_name in memo:
                return memo[table_name]

            parents = parent_map.get(table_name, set())
            if not parents:
                memo[table_name] = 0
                return 0

            memo[table_name] = 1 + max(depth(parent) for parent in parents)
            return memo[table_name]

        return depth(self.child_table_name)

    def _empty_parent_keys(self, sampled_data):
        """Return the parent primary keys that received no child rows.

        The child row counts are decided by HMA's learned relationship model and
        are never modified by enforcement, so this set is stable for the whole
        top-down pass.
        """
        child_data = sampled_data[self.child_table_name]
        foreign_key = self._foreign_key
        parent_data = sampled_data[self.parent_table_name]
        parents_with_children = set(child_data[foreign_key].unique())
        return {
            key
            for key in parent_data[self._parent_primary_key].to_numpy()
            if key not in parents_with_children
        }

    def enforce(self, sampled_data, pinned_zero_keys=None):
        """Enforce the exact parent/child sum relationship, top-down.

        This is intended to be called by ``enforce_parent_child_sum_constraints``
        in root-to-leaf order. For every parent group:

        * A parent with no child rows has its total forced to ``0`` (the only
          parent amount that is ever rewritten: the HMA ``num_rows=0`` signal is
          authoritative, and with no children there is nothing to sum).
        * Every other parent keeps its model-sampled total as the fixed target.
          Child rows whose own child group is structurally empty (passed in via
          ``pinned_zero_keys``) are pinned to ``0``; the remaining rows receive
          the full target through ``_allocate_integer_values`` using their
          sampled values as weights, so the within-group shape learned by HMA is
          preserved and the residual is spread across the group instead of being
          dumped onto one row.

        Args:
            sampled_data (dict[str, pd.DataFrame]):
                Sampled tables already restored to their integer dtypes.
            pinned_zero_keys (set, optional):
                Primary key values of this child table that must take value
                ``0`` because they have no rows in their own child table.

        Returns:
            set:
                The parent primary keys that received no child rows, for the
                orchestrator to pin one level up.
        """
        if pinned_zero_keys is None:
            pinned_zero_keys = set()

        parent_data = sampled_data[self.parent_table_name]
        child_data = sampled_data[self.child_table_name]
        foreign_key = self._foreign_key
        primary_key = self._parent_primary_key
        child_primary_key = self.metadata.tables[self.child_table_name].primary_key
        child_column = self.child_value_column
        parent_column = self.parent_total_column
        parent_dtype = parent_data[parent_column].dtype
        child_dtype = child_data[child_column].dtype
        upper_bound = self.child_max if self.child_max is not None else math.inf

        empty_parent_keys = self._empty_parent_keys(sampled_data)

        # Structural repair: empty parent groups must total exactly 0.
        if empty_parent_keys:
            empty_positions = parent_data.index[parent_data[primary_key].isin(empty_parent_keys)]
            parent_data.loc[empty_positions, parent_column] = pd.Series(
                0, index=empty_positions, dtype=parent_dtype
            )

        grouped_positions = {
            key: group.index.to_numpy()
            for key, group in child_data.groupby(foreign_key, sort=False)
        }
        parent_lookup = parent_data.set_index(primary_key)[parent_column]

        for parent_key, total in parent_lookup.items():
            total = int(total)
            positions = grouped_positions.get(parent_key)
            if positions is None:
                continue  # Structural repair already forced the total to 0.

            pinned_mask = (
                child_data.loc[positions, child_primary_key].isin(pinned_zero_keys).to_numpy()
            )
            pinned_positions = positions[pinned_mask]
            free_positions = positions[~pinned_mask]

            # Pinned rows (their own child group is empty) must be exactly 0.
            if pinned_positions.size > 0:
                if self.child_min > 0:
                    raise _InfeasibleParentChildSum(
                        f'Child key(s) {list(pinned_zero_keys)[:3]} in table '
                        f"'{self.child_table_name}' must be 0 but child_min="
                        f'{self.child_min}.'
                    )
                child_data.loc[pinned_positions, child_column] = pd.Series(
                    0, index=pinned_positions, dtype=child_dtype
                )

            if free_positions.size == 0:
                if total != 0:
                    # Every child row of this group is itself an empty parent one
                    # level down, so none can carry value. Treat the group as
                    # structurally empty and pin the parent total to 0 instead of
                    # rejecting an otherwise valid draw.
                    parent_position = parent_data.index[parent_data[primary_key] == parent_key]
                    parent_data.loc[parent_position, parent_column] = 0
                    empty_parent_keys.add(parent_key)

                continue

            sampled_values = child_data.loc[free_positions, child_column].to_numpy()
            weights = np.where(sampled_values > 0, sampled_values, 0.0).astype(float)
            new_values = self._allocate_integer_values(
                num_rows=free_positions.size,
                target=total,
                lower_bound=self.child_min,
                upper_bound=upper_bound,
                weights=weights,
            )
            child_data.loc[free_positions, child_column] = pd.Series(
                new_values, index=free_positions, dtype=child_dtype
            )

        return empty_parent_keys


def enforce_parent_child_sum_constraints(constraints, sampled_data, metadata, max_tries):
    """Enforce ``ParentChildSum`` constraints on one sampled dataset, top-down.

    Constraints are visited in root-to-leaf order along the table tree. The row
    counts produced by HMA are never changed. Parent totals keep their
    model-sampled value for every non-empty group; only structurally empty groups
    are pinned to ``0``, and their keys are forwarded to the constraint one level
    above so those child rows are allocated ``0`` instead of taking part in the
    weighted split. This keeps every layer simultaneously consistent without
    overwriting any parent total with the sum of an existing group of children.

    A failed allocation (the sampled target cannot fit the available rows and
    bounds) raises ``SamplingError`` naming the relationship that could not be
    satisfied; the whole sampling attempt is then regenerated.

    Args:
        constraints (list[ParentChildSum]):
            Fitted ParentChildSum constraints.
        sampled_data (dict[str, pd.DataFrame]):
            A complete sampled dataset.
        metadata (sdv.Metadata):
            The (original) metadata.
        max_tries (int):
            The overall attempt budget, used only to make the error message
            actionable.
    """
    ordered = sorted(
        constraints,
        key=lambda constraint: constraint._table_depth(metadata),
    )

    # For each table, the primary keys that are structurally empty parents under
    # the constraint directly below it.
    empty_keys_by_table = {}
    for constraint in ordered:
        empty_keys_by_table[constraint.parent_table_name] = constraint._empty_parent_keys(
            sampled_data
        )

    for constraint in ordered:
        # This constraint's child rows that are empty parents in the next level
        # must be pinned to 0.
        pinned_zero_keys = empty_keys_by_table.get(constraint.child_table_name, set())
        try:
            constraint.enforce(sampled_data, pinned_zero_keys=pinned_zero_keys)
        except _InfeasibleParentChildSum as error:
            raise SamplingError(
                'Unable to satisfy the ParentChildSum constraint between parent '
                f"table '{constraint.parent_table_name}' column "
                f"'{constraint.parent_total_column}' and child table "
                f"'{constraint.child_table_name}' column "
                f"'{constraint.child_value_column}': {error} "
                f'(generation attempt budget max_tries={max_tries}).'
            ) from error

    return sampled_data
