"""Unit tests for the ParentChildSum constraint."""

import math

import numpy as np
import pandas as pd
import pytest

from sdv.cag import ParentChildSum
from sdv.cag._errors import ConstraintNotMetError
from sdv.cag.parent_child_sum import (
    _InfeasibleParentChildSum,
    enforce_parent_child_sum_constraints,
)
from sdv.errors import NotFittedError, SamplingError
from sdv.metadata import Metadata


def _three_table_metadata():
    metadata = Metadata()
    metadata.add_table('customers')
    metadata.add_column('customer_id', 'customers', sdtype='id')
    metadata.set_primary_key('customer_id', 'customers')
    metadata.add_column('total', 'customers', sdtype='numerical', computer_representation='Int64')

    metadata.add_table('orders')
    metadata.add_column('order_id', 'orders', sdtype='id')
    metadata.set_primary_key('order_id', 'orders')
    metadata.add_column('customer_id', 'orders', sdtype='id')
    metadata.add_column('amount', 'orders', sdtype='numerical', computer_representation='Int64')

    metadata.add_table('lines')
    metadata.add_column('line_id', 'lines', sdtype='id')
    metadata.set_primary_key('line_id', 'lines')
    metadata.add_column('order_id', 'lines', sdtype='id')
    metadata.add_column('amount', 'lines', sdtype='numerical', computer_representation='Int64')

    metadata.add_relationship('customers', 'orders', 'customer_id', 'customer_id')
    metadata.add_relationship('orders', 'lines', 'order_id', 'order_id')
    return metadata


def _valid_data():
    customers = pd.DataFrame({
        'customer_id': ['c1', 'c2', 'c3'],
        'total': [10, 30, 0],
    })
    orders = pd.DataFrame({
        'order_id': ['o1', 'o2', 'o3'],
        'customer_id': ['c1', 'c2', 'c2'],
        'amount': [10, 20, 10],
    })
    lines = pd.DataFrame({
        'line_id': ['l1', 'l2', 'l3', 'l4'],
        'order_id': ['o1', 'o2', 'o2', 'o3'],
        'amount': [10, 8, 12, 10],
    })
    return {'customers': customers, 'orders': orders, 'lines': lines}


class TestParentChildSumInit:
    def test_defaults(self):
        constraint = ParentChildSum('customers', 'orders', 'total', 'amount')
        assert constraint.child_min == 0
        assert constraint.child_max is None
        assert constraint._is_single_table is False

    @pytest.mark.parametrize(
        'kwargs, message',
        [
            (dict(parent_table_name=1), 'parent_table_name'),
            (dict(child_table_name=2), 'child_table_name'),
            (dict(parent_total_column=3), 'parent_total_column'),
            (dict(child_value_column=4), 'child_value_column'),
        ],
    )
    def test_string_arguments_validated(self, kwargs, message):
        defaults = dict(
            parent_table_name='customers',
            child_table_name='orders',
            parent_total_column='total',
            child_value_column='amount',
        )
        defaults.update(kwargs)
        with pytest.raises(ValueError, match=message):
            ParentChildSum(**defaults)

    @pytest.mark.parametrize('child_min', [True, False, -1, 1.0, '0'])
    def test_child_min_must_be_non_bool_non_negative_int(self, child_min):
        with pytest.raises(ValueError, match='child_min'):
            ParentChildSum('customers', 'orders', 'total', 'amount', child_min=child_min)

    @pytest.mark.parametrize('child_max', [True, False, 1.5, '5'])
    def test_child_max_must_be_int_or_none(self, child_max):
        with pytest.raises(ValueError, match='child_max'):
            ParentChildSum('customers', 'orders', 'total', 'amount', child_max=child_max)

    def test_child_max_below_child_min_rejected(self):
        with pytest.raises(ValueError, match='child_max'):
            ParentChildSum(
                'customers',
                'orders',
                'total',
                'amount',
                child_min=5,
                child_max=4,
            )

    def test_valid_bounds_accepted(self):
        constraint = ParentChildSum(
            'customers',
            'orders',
            'total',
            'amount',
            child_min=0,
            child_max=100,
        )
        assert constraint.child_min == 0
        assert constraint.child_max == 100


class TestParentChildSumMetadataValidation:
    def test_valid_metadata(self):
        constraint = ParentChildSum('orders', 'lines', 'amount', 'amount')
        constraint.validate(metadata=_three_table_metadata())
        assert constraint._foreign_key == 'order_id'
        assert constraint._parent_primary_key == 'order_id'

    def test_missing_table(self):
        constraint = ParentChildSum('nope', 'lines', 'amount', 'amount')
        with pytest.raises(ConstraintNotMetError, match='missing'):
            constraint.validate(metadata=_three_table_metadata())

    def test_missing_column(self):
        constraint = ParentChildSum('orders', 'lines', 'amount', 'nope')
        with pytest.raises(ConstraintNotMetError, match='missing column'):
            constraint.validate(metadata=_three_table_metadata())

    def test_wrong_sdtype(self):
        # 'order_id' is an id column and also the relationship key, so the
        # relationship-key check fires first; use a dedicated column instead.
        metadata = Metadata()
        metadata.add_table('parents')
        metadata.add_column('pid', 'parents', sdtype='id')
        metadata.set_primary_key('pid', 'parents')
        metadata.add_column('flag', 'parents', sdtype='categorical')
        metadata.add_table('children')
        metadata.add_column('cid', 'children', sdtype='id')
        metadata.set_primary_key('cid', 'children')
        metadata.add_column('pid', 'children', sdtype='id')
        metadata.add_column(
            'value',
            'children',
            sdtype='numerical',
            computer_representation='Int64',
        )
        metadata.add_relationship('parents', 'children', 'pid', 'pid')
        with pytest.raises(ConstraintNotMetError, match='numerical'):
            ParentChildSum('parents', 'children', 'flag', 'value').validate(metadata=metadata)

    def test_no_direct_relationship(self):
        constraint = ParentChildSum('customers', 'lines', 'total', 'amount')
        with pytest.raises(ConstraintNotMetError, match='no direct relationship'):
            constraint.validate(metadata=_three_table_metadata())

    def test_amount_column_is_relationship_key(self):
        metadata = Metadata()
        metadata.add_table('parents')
        metadata.add_column('pid', 'parents', sdtype='id')
        metadata.set_primary_key('pid', 'parents')
        metadata.add_column(
            'value',
            'parents',
            sdtype='numerical',
            computer_representation='Int64',
        )
        metadata.add_table('children')
        metadata.add_column('cid', 'children', sdtype='id')
        metadata.set_primary_key('cid', 'children')
        metadata.add_column('pid', 'children', sdtype='id')
        metadata.add_column(
            'value',
            'children',
            sdtype='numerical',
            computer_representation='Int64',
        )
        metadata.add_relationship('parents', 'children', 'pid', 'pid')

        with pytest.raises(ConstraintNotMetError, match='relationship key'):
            ParentChildSum('parents', 'children', 'pid', 'value').validate(metadata=metadata)
        with pytest.raises(ConstraintNotMetError, match='relationship key'):
            ParentChildSum('parents', 'children', 'value', 'pid').validate(metadata=metadata)

    def test_multiple_relationships_rejected(self):
        metadata = _three_table_metadata()
        metadata.add_column('second_fk', 'lines', sdtype='id')
        metadata.add_relationship('orders', 'lines', 'order_id', 'second_fk')
        constraint = ParentChildSum('orders', 'lines', 'amount', 'amount')
        with pytest.raises(ConstraintNotMetError, match='exactly one'):
            constraint.validate(metadata=metadata)


class TestParentChildSumDataValidation:
    def _fit(self, data, metadata=None, **kwargs):
        constraint = ParentChildSum('orders', 'lines', 'amount', 'amount', **kwargs)
        constraint.fit(data, metadata or _three_table_metadata())
        return constraint

    def test_valid_data_passes(self):
        constraint = self._fit(_valid_data())
        assert constraint._fitted is True

    def test_duplicate_primary_key(self):
        data = _valid_data()
        data['orders'].loc[1, 'order_id'] = data['orders'].loc[0, 'order_id']
        with pytest.raises(ConstraintNotMetError, match='duplicated'):
            self._fit(data)

    def test_null_primary_key(self):
        data = _valid_data()
        data['orders'].loc[0, 'order_id'] = None
        with pytest.raises(ConstraintNotMetError, match='null values'):
            self._fit(data)

    def test_orphan_child(self):
        data = _valid_data()
        data['lines'].loc[0, 'order_id'] = 'missing'
        with pytest.raises(ConstraintNotMetError, match='do not exist'):
            self._fit(data)

    def test_null_foreign_key(self):
        data = _valid_data()
        data['lines'].loc[0, 'order_id'] = None
        with pytest.raises(ConstraintNotMetError, match='null values'):
            self._fit(data)

    def test_null_amount(self):
        data = _valid_data()
        data['lines'].loc[0, 'amount'] = None
        data['lines']['amount'] = data['lines']['amount'].astype('Float64')
        with pytest.raises(ConstraintNotMetError, match='null values'):
            self._fit(data)

    def test_decimal_amount(self):
        data = _valid_data()
        data['lines'].loc[0, 'amount'] = 1.5
        with pytest.raises(ConstraintNotMetError, match='integer'):
            self._fit(data)

    def test_boolean_amount_rejected(self):
        data = _valid_data()
        data['lines']['amount'] = data['lines']['amount'].astype(bool)
        with pytest.raises(ConstraintNotMetError, match='booleans'):
            self._fit(data)

    def test_total_mismatch(self):
        data = _valid_data()
        data['orders'].loc[0, 'amount'] = 999
        with pytest.raises(ConstraintNotMetError, match='does not match the sum'):
            self._fit(data)

    def test_empty_parent_must_be_zero(self):
        data = _valid_data()
        data['customers'].loc[2, 'total'] = 5
        constraint = ParentChildSum('customers', 'orders', 'total', 'amount')
        with pytest.raises(ConstraintNotMetError, match='does not match the sum'):
            constraint.fit(data, _three_table_metadata())

    def test_out_of_bounds_child(self):
        data = _valid_data()
        with pytest.raises(ConstraintNotMetError, match='declared bounds'):
            self._fit(data, child_min=100)

    def test_input_data_unchanged_after_fit(self):
        data = _valid_data()
        before = {table: frame.copy() for table, frame in data.items()}
        self._fit(data)
        for table in before:
            pd.testing.assert_frame_equal(before[table], data[table])

    def test_transform_is_a_noop(self):
        constraint = self._fit(_valid_data())
        transformed = constraint.transform(_valid_data())
        for table in transformed:
            assert list(transformed[table].columns) == list(_valid_data()[table].columns)


class TestAllocator:
    allocate = staticmethod(ParentChildSum._allocate_integer_values)

    @pytest.mark.parametrize(
        'target, n, lo, hi',
        [
            (10, 4, 0, 100),
            (0, 3, 0, 5),
            (17, 5, 1, 10),
            (100, 7, 0, 20),
            (50, 50, 1, 1),
            (1000, 33, 0, math.inf),
            (7, 3, 2, 3),
        ],
    )
    def test_exact_sum_and_bounds(self, target, n, lo, hi):
        values = self.allocate(n, target, lo, hi)
        assert len(values) == n
        assert sum(int(value) for value in values) == target
        assert all(lo <= int(value) <= hi for value in values)
        assert all(type(value) is int for value in values)

    def test_infeasible_targets(self):
        with pytest.raises(_InfeasibleParentChildSum, match='cannot be split'):
            self.allocate(3, 100, 0, 5)
        with pytest.raises(_InfeasibleParentChildSum, match='cannot be split'):
            self.allocate(2, 1, 1, 5)
        with pytest.raises(_InfeasibleParentChildSum, match='zero rows'):
            self.allocate(0, 5, 0, 10)
        assert list(self.allocate(0, 0, 0, 10)) == []

    def test_weighted_split(self):
        values = self.allocate(2, 10, 0, math.inf, weights=np.array([4.0, 1.0]))
        assert [int(value) for value in values] == [8, 2]

    def test_residual_spreads_over_group(self):
        values = self.allocate(4, 8, 0, 2, weights=np.array([100.0, 1.0, 1.0, 1.0]))
        assert sum(int(value) for value in values) == 8
        assert int(values[0]) == 2
        assert all(int(value) == 2 for value in values[1:])

    def test_zero_weights_split_evenly(self):
        values = self.allocate(4, 8, 0, 10, weights=np.zeros(4))
        assert [int(value) for value in values] == [2, 2, 2, 2]


class TestEnforce:
    def _fitted(self, *args, **kwargs):
        constraint = ParentChildSum(*args, **kwargs)
        metadata = _three_table_metadata()
        constraint.validate(metadata=metadata)
        constraint.metadata = metadata
        constraint._fitted = True
        return constraint

    def test_parent_totals_kept_children_reallocated(self):
        constraint = self._fitted('orders', 'lines', 'amount', 'amount')
        sampled = {
            'customers': pd.DataFrame({
                'customer_id': ['c1'],
                'total': [0],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1'],
                'customer_id': ['c1'],
                'amount': [50],
            }),
            'lines': pd.DataFrame({
                'line_id': ['l1', 'l2'],
                'order_id': ['o1', 'o1'],
                'amount': [1, 2],
            }),
        }
        constraint.enforce(sampled)
        assert sampled['orders'].loc[0, 'amount'] == 50
        assert int(sampled['lines']['amount'].sum()) == 50

    def test_empty_group_pinned_to_zero(self):
        constraint = self._fitted('orders', 'lines', 'amount', 'amount')
        sampled = {
            'customers': pd.DataFrame({
                'customer_id': ['c1'],
                'total': [0],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1', 'o2'],
                'customer_id': ['c1', 'c1'],
                'amount': [10, 7],
            }),
            'lines': pd.DataFrame({
                'line_id': ['l1'],
                'order_id': ['o1'],
                'amount': [3],
            }),
        }
        empty = constraint.enforce(sampled)
        assert empty == {'o2'}
        assert sampled['orders'].set_index('order_id').loc['o2', 'amount'] == 0

    def test_all_pinned_group_pin_parent_to_zero(self):
        # The parent has child rows, but every one of them is an empty parent
        # one level down (pinned to zero). The parent total is forced to 0.
        constraint = self._fitted('customers', 'orders', 'total', 'amount')
        sampled = {
            'customers': pd.DataFrame({
                'customer_id': ['c1'],
                'total': [15],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1', 'o2'],
                'customer_id': ['c1', 'c1'],
                'amount': [7, 8],
            }),
            'lines': pd.DataFrame(columns=['line_id', 'order_id', 'amount']),
        }
        empty = constraint.enforce(sampled, pinned_zero_keys={'o1', 'o2'})
        assert 'c1' in empty
        assert sampled['customers'].loc[0, 'total'] == 0
        assert (sampled['orders']['amount'] == 0).all()

    def test_positive_lower_bound_infeasible_for_zero_total(self):
        # child_min=1 with a parent total of 0 cannot be allocated.
        constraint = self._fitted('orders', 'lines', 'amount', 'amount', child_min=1)
        sampled = {
            'customers': pd.DataFrame({
                'customer_id': ['c1'],
                'total': [0],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1'],
                'customer_id': ['c1'],
                'amount': [0],
            }),
            'lines': pd.DataFrame({
                'line_id': ['l1', 'l2'],
                'order_id': ['o1', 'o1'],
                'amount': [0, 0],
            }),
        }
        with pytest.raises(_InfeasibleParentChildSum, match='cannot be split'):
            constraint.enforce(sampled)

    def test_positive_lower_bound_respected(self):
        constraint = self._fitted('orders', 'lines', 'amount', 'amount', child_min=1, child_max=100)
        sampled = {
            'customers': pd.DataFrame({
                'customer_id': ['c1'],
                'total': [0],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1'],
                'customer_id': ['c1'],
                'amount': [10],
            }),
            'lines': pd.DataFrame({
                'line_id': ['l1', 'l2', 'l3'],
                'order_id': ['o1', 'o1', 'o1'],
                'amount': [1, 1, 1],
            }),
        }
        constraint.enforce(sampled)
        values = sampled['lines']['amount']
        assert (values >= 1).all()
        assert int(values.sum()) == 10

    def test_infeasible_group_raises(self):
        constraint = self._fitted('orders', 'lines', 'amount', 'amount', child_max=5)
        sampled = {
            'customers': pd.DataFrame({
                'customer_id': ['c1'],
                'total': [0],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1'],
                'customer_id': ['c1'],
                'amount': [100],
            }),
            'lines': pd.DataFrame({
                'line_id': ['l1', 'l2'],
                'order_id': ['o1', 'o1'],
                'amount': [1, 2],
            }),
        }
        with pytest.raises(_InfeasibleParentChildSum, match='cannot be split'):
            constraint.enforce(sampled)

    def test_two_levels_simultaneously(self):
        c1 = self._fitted('customers', 'orders', 'total', 'amount')
        c2 = self._fitted('orders', 'lines', 'amount', 'amount')
        sampled = {
            'customers': pd.DataFrame({
                'customer_id': ['c1', 'c2'],
                'total': [10, 30],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1', 'o2', 'o3'],
                'customer_id': ['c1', 'c2', 'c2'],
                'amount': [10, 20, 10],
            }),
            'lines': pd.DataFrame({
                'line_id': ['l1', 'l2', 'l3'],
                'order_id': ['o1', 'o2', 'o2'],
                'amount': [3, 7, 9],
            }),
        }
        enforce_parent_child_sum_constraints(
            [c1, c2], sampled, _three_table_metadata(), max_tries=100
        )

        line_sums = sampled['lines'].groupby('order_id')['amount'].sum()
        order_amounts = sampled['orders'].set_index('order_id')['amount']
        assert (line_sums.reindex(order_amounts.index, fill_value=0) == order_amounts).all()

        order_sums = sampled['orders'].groupby('customer_id')['amount'].sum()
        customer_totals = sampled['customers'].set_index('customer_id')['total']
        assert (order_sums.reindex(customer_totals.index, fill_value=0) == customer_totals).all()
        # The empty order is pinned to 0; its customer total stays at 30 and the
        # only non-empty child order carries the full 30.
        orders = sampled['orders'].set_index('order_id')
        assert orders.loc['o3', 'amount'] == 0
        assert orders.loc['o2', 'amount'] == 30
        assert sampled['customers'].set_index('customer_id').loc['c2', 'total'] == 30

        # Integer dtypes are preserved.
        for table, column in [
            ('customers', 'total'),
            ('orders', 'amount'),
            ('lines', 'amount'),
        ]:
            assert np.issubdtype(sampled[table][column].dtype, np.integer)

    def test_infeasible_becomes_sampling_error(self):
        c2 = self._fitted('orders', 'lines', 'amount', 'amount', child_max=5)
        sampled = {
            'customers': pd.DataFrame({
                'customer_id': ['c1'],
                'total': [0],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1'],
                'customer_id': ['c1'],
                'amount': [100],
            }),
            'lines': pd.DataFrame({
                'line_id': ['l1', 'l2'],
                'order_id': ['o1', 'o1'],
                'amount': [1, 2],
            }),
        }
        with pytest.raises(SamplingError, match='max_tries=3'):
            enforce_parent_child_sum_constraints(
                [c2], sampled, _three_table_metadata(), max_tries=3
            )

    def test_is_valid(self):
        constraint = self._fitted('orders', 'lines', 'amount', 'amount')
        good = {
            'customers': pd.DataFrame({
                'customer_id': ['c1'],
                'total': [0],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1', 'o2'],
                'customer_id': ['c1', 'c1'],
                'amount': [5, 7],
            }),
            'lines': pd.DataFrame({
                'line_id': ['l1', 'l2', 'l3'],
                'order_id': ['o1', 'o2', 'o2'],
                'amount': [5, 3, 4],
            }),
        }
        valid = constraint.is_valid(good)
        assert valid['orders'].all()
        assert valid['lines'].all()

        bad = {
            'customers': pd.DataFrame({
                'customer_id': ['c1'],
                'total': [0],
            }),
            'orders': pd.DataFrame({
                'order_id': ['o1'],
                'customer_id': ['c1'],
                'amount': [9],
            }),
            'lines': pd.DataFrame({
                'line_id': ['l1'],
                'order_id': ['o1'],
                'amount': [5],
            }),
        }
        valid = constraint.is_valid(bad)
        assert not valid['orders'].any()
        assert not valid['lines'].any()

    def test_is_valid_requires_fit(self):
        constraint = ParentChildSum('orders', 'lines', 'amount', 'amount')
        with pytest.raises(NotFittedError):
            constraint.is_valid(_valid_data())
