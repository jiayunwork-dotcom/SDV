"""Integration tests for the ParentChildSum constraint with HMASynthesizer."""

import numpy as np
import pandas as pd
import pytest

from sdv.cag import ParentChildSum
from sdv.cag.parent_child_sum import ParentChildSum as PCS
from sdv.errors import SamplingError, SynthesizerInputError
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer


def _build_metadata():
    metadata = Metadata()
    metadata.add_table('customers')
    metadata.add_column('customer_id', 'customers', sdtype='id')
    metadata.set_primary_key('customer_id', 'customers')
    metadata.add_column(
        'total',
        'customers',
        sdtype='numerical',
        computer_representation='Int64',
    )

    metadata.add_table('orders')
    metadata.add_column('order_id', 'orders', sdtype='id')
    metadata.set_primary_key('order_id', 'orders')
    metadata.add_column('customer_id', 'orders', sdtype='id')
    metadata.add_column(
        'amount',
        'orders',
        sdtype='numerical',
        computer_representation='Int64',
    )

    metadata.add_table('lines')
    metadata.add_column('line_id', 'lines', sdtype='id')
    metadata.set_primary_key('line_id', 'lines')
    metadata.add_column('order_id', 'lines', sdtype='id')
    metadata.add_column(
        'amount',
        'lines',
        sdtype='numerical',
        computer_representation='Int64',
    )

    metadata.add_relationship('customers', 'orders', 'customer_id', 'customer_id')
    metadata.add_relationship('orders', 'lines', 'order_id', 'order_id')
    return metadata


def _build_data(seed=7, n_customers=40):
    """Three layers with empty and non-empty groups of varying size and amount."""
    rng = np.random.default_rng(seed)
    customers, orders, lines = [], [], []
    customer_idx = order_idx = line_idx = 0
    for _ in range(n_customers):
        customer_id = f'c{customer_idx}'
        customer_idx += 1
        n_orders = int(rng.choice([0, 1, 2, 3, 4], p=[0.2, 0.25, 0.25, 0.2, 0.1]))
        customer_total = 0
        for _ in range(n_orders):
            order_id = f'o{order_idx}'
            order_idx += 1
            n_lines = int(rng.choice([0, 1, 2, 3, 5], p=[0.25, 0.25, 0.25, 0.15, 0.1]))
            order_total = 0
            for _ in range(n_lines):
                line_id = f'l{line_idx}'
                line_idx += 1
                value = int(rng.integers(1, 50))
                lines.append({
                    'line_id': line_id,
                    'order_id': order_id,
                    'amount': value,
                })
                order_total += value
            orders.append({
                'order_id': order_id,
                'customer_id': customer_id,
                'amount': order_total,
            })
            customer_total += order_total
        customers.append({
            'customer_id': customer_id,
            'total': customer_total,
        })

    return {
        'customers': pd.DataFrame(customers),
        'orders': pd.DataFrame(orders),
        'lines': pd.DataFrame(lines),
    }


def _assert_invariants(sampled, metadata, bounds=None):
    customers = sampled['customers']
    orders = sampled['orders']
    lines = sampled['lines']

    # Primary keys stay unique.
    assert customers['customer_id'].is_unique
    assert orders['order_id'].is_unique
    assert lines['line_id'].is_unique

    # Foreign keys are non-null and point at existing parent rows.
    assert not orders['customer_id'].isna().any()
    assert not lines['order_id'].isna().any()
    assert set(orders['customer_id']).issubset(set(customers['customer_id']))
    assert set(lines['order_id']).issubset(set(orders['order_id']))

    # Amounts stay non-null integers.
    for table, column in [
        ('customers', 'total'),
        ('orders', 'amount'),
        ('lines', 'amount'),
    ]:
        series = sampled[table][column]
        assert not series.isna().any()
        assert np.issubdtype(series.dtype, np.integer)

    if bounds is not None:
        lower, upper = bounds
        assert (lines['amount'] >= lower).all()
        if upper is not None:
            assert (lines['amount'] <= upper).all()

    # Order level: each order total equals the sum of its line items.
    line_sums = lines.groupby('order_id')['amount'].sum()
    order_totals = orders.set_index('order_id')['amount']
    aligned_lines = line_sums.reindex(order_totals.index, fill_value=0)
    pd.testing.assert_series_equal(aligned_lines, order_totals, check_names=False)

    # Customer level: each customer total equals the sum of its orders.
    order_sums = orders.groupby('customer_id')['amount'].sum()
    customer_totals = customers.set_index('customer_id')['total']
    aligned_orders = order_sums.reindex(customer_totals.index, fill_value=0)
    pd.testing.assert_series_equal(aligned_orders, customer_totals, check_names=False)

    # Empty groups are exactly zero.
    empty_orders = set(orders['order_id']) - set(lines['order_id'])
    if empty_orders:
        assert (orders.set_index('order_id').loc[list(empty_orders), 'amount'] == 0).all()
    empty_customers = set(customers['customer_id']) - set(orders['customer_id'])
    if empty_customers:
        assert (customers.set_index('customer_id').loc[list(empty_customers), 'total'] == 0).all()


def _fit_synthesizer(data, metadata, **line_kwargs):
    synthesizer = HMASynthesizer(metadata, verbose=False)
    synthesizer.add_constraints([
        ParentChildSum('customers', 'orders', 'total', 'amount'),
        ParentChildSum('orders', 'lines', 'amount', 'amount', **line_kwargs),
    ])
    synthesizer.fit(data)
    return synthesizer


class TestHMAParentChildSum:
    def test_two_level_totals_hold_across_scales(self):
        data = _build_data()
        metadata = _build_metadata()
        synthesizer = _fit_synthesizer(data, metadata)

        sizes = []
        for scale in (0.5, 1.0, 1.5):
            sampled = synthesizer.sample(scale=scale)
            _assert_invariants(sampled, metadata)
            sizes.append({table: len(frame) for table, frame in sampled.items()})

        # Scaling actually changes the number of generated rows.
        assert sizes[0]['customers'] < sizes[1]['customers'] < sizes[2]['customers']
        assert sizes[0]['lines'] < sizes[1]['lines'] < sizes[2]['lines']

    def test_reset_sampling_is_reproducible(self):
        data = _build_data()
        metadata = _build_metadata()
        synthesizer = _fit_synthesizer(data, metadata)

        synthesizer.reset_sampling()
        first = synthesizer.sample(scale=1.0)
        synthesizer.reset_sampling()
        second = synthesizer.sample(scale=1.0)
        for table in first:
            pd.testing.assert_frame_equal(first[table], second[table])

    def test_save_and_load(self, tmp_path):
        data = _build_data()
        metadata = _build_metadata()
        synthesizer = _fit_synthesizer(data, metadata)
        path = tmp_path / 'synth.pkl'
        synthesizer.save(str(path))

        loaded = HMASynthesizer.load(str(path))
        assert any(isinstance(constraint, PCS) for constraint in loaded.get_constraints())
        sampled = loaded.sample(scale=1.0)
        _assert_invariants(sampled, metadata)

    def test_bounds_are_respected(self):
        data = _build_data()
        metadata = _build_metadata()
        synthesizer = _fit_synthesizer(data, metadata, child_min=0, child_max=60)
        sampled = synthesizer.sample(scale=1.0)
        _assert_invariants(sampled, metadata, bounds=(0, 60))

    def test_generated_groups_vary(self):
        """New groups must show variation in amounts and child row counts."""
        data = _build_data()
        metadata = _build_metadata()
        synthesizer = _fit_synthesizer(data, metadata)
        sampled = synthesizer.sample(scale=2.0)

        line_counts = sampled['lines'].groupby('order_id').size()
        assert line_counts.nunique() > 1
        assert sampled['lines']['amount'].nunique() > 1
        assert sampled['orders']['amount'].nunique() > 1
        assert sampled['customers']['total'].nunique() > 1
        # HMA's relationship cardinality is preserved: not every parent has one
        # child row.
        assert line_counts.max() > 1

    def test_max_tries_validation(self):
        data = _build_data()
        metadata = _build_metadata()
        synthesizer = _fit_synthesizer(data, metadata)
        for invalid in (0, -3, 1.5, True, '10'):
            with pytest.raises(SynthesizerInputError):
                synthesizer.sample(scale=1.0, max_tries=invalid)

    def test_budget_exhaustion_raises_sampling_error(self, monkeypatch):
        data = _build_data()
        metadata = _build_metadata()
        synthesizer = _fit_synthesizer(data, metadata)

        def broken_enforce(self, sampled_data, pinned_zero_keys=None):
            from sdv.cag.parent_child_sum import _InfeasibleParentChildSum

            raise _InfeasibleParentChildSum('forced capacity shortfall')

        monkeypatch.setattr(PCS, 'enforce', broken_enforce)
        with pytest.raises(SamplingError, match='forced capacity shortfall'):
            synthesizer.sample(scale=1.0, max_tries=4)

    def test_half_finished_tables_never_returned(self, monkeypatch):
        data = _build_data()
        metadata = _build_metadata()
        synthesizer = _fit_synthesizer(data, metadata)
        calls = {'count': 0}

        def flaky_enforce(self, sampled_data, pinned_zero_keys=None):
            calls['count'] += 1
            from sdv.cag.parent_child_sum import _InfeasibleParentChildSum

            raise _InfeasibleParentChildSum('always fails')

        monkeypatch.setattr(PCS, 'enforce', flaky_enforce)
        with pytest.raises(SamplingError):
            synthesizer.sample(scale=1.0, max_tries=3)
        assert calls['count'] == 3

    def test_unconstrained_sample_keeps_original_behavior(self):
        data = _build_data()
        metadata = _build_metadata()
        synthesizer = HMASynthesizer(metadata, verbose=False)
        synthesizer.fit(data)

        sampled = synthesizer.sample(0.5)
        assert set(sampled) == set(data)
        # max_tries is accepted but ignored without the constraint.
        sampled = synthesizer.sample(scale=1.0, max_tries=3)
        assert set(sampled) == set(data)

    def test_fit_rejects_violating_data(self):
        data = _build_data()
        metadata = _build_metadata()
        data['customers'].loc[0, 'total'] += 1
        synthesizer = HMASynthesizer(metadata, verbose=False)
        synthesizer.add_constraints([
            ParentChildSum('customers', 'orders', 'total', 'amount'),
        ])
        from sdv.cag._errors import ConstraintNotMetError

        with pytest.raises(ConstraintNotMetError, match='does not match the sum'):
            synthesizer.fit(data)

    def test_input_tables_unchanged_after_fit(self):
        data = _build_data()
        before = {table: frame.copy() for table, frame in data.items()}
        metadata = _build_metadata()
        _fit_synthesizer(data, metadata)
        for table in before:
            pd.testing.assert_frame_equal(before[table], data[table])

    def test_independent_branches_each_enforced(self):
        """One parent table with two child tables constrained independently."""
        metadata = Metadata()
        metadata.add_table('customers')
        metadata.add_column('customer_id', 'customers', sdtype='id')
        metadata.set_primary_key('customer_id', 'customers')
        metadata.add_column(
            'total',
            'customers',
            sdtype='numerical',
            computer_representation='Int64',
        )
        metadata.add_table('orders')
        metadata.add_column('order_id', 'orders', sdtype='id')
        metadata.set_primary_key('order_id', 'orders')
        metadata.add_column('customer_id', 'orders', sdtype='id')
        metadata.add_column(
            'amount',
            'orders',
            sdtype='numerical',
            computer_representation='Int64',
        )
        metadata.add_table('payments')
        metadata.add_column('payment_id', 'payments', sdtype='id')
        metadata.set_primary_key('payment_id', 'payments')
        metadata.add_column('customer_id', 'payments', sdtype='id')
        metadata.add_column(
            'amount',
            'payments',
            sdtype='numerical',
            computer_representation='Int64',
        )
        metadata.add_relationship('customers', 'orders', 'customer_id', 'customer_id')
        metadata.add_relationship('customers', 'payments', 'customer_id', 'customer_id')

        rng = np.random.default_rng(3)
        customers, orders, payments = [], [], []
        for index in range(25):
            customer_id = f'c{index}'
            order_total = int(rng.integers(0, 5)) * 10
            customers.append({'customer_id': customer_id, 'total': order_total})
            for _ in range(int(rng.integers(1, 4))):
                orders.append({
                    'order_id': f'o{len(orders)}',
                    'customer_id': customer_id,
                    'amount': 0,
                })
            for _ in range(int(rng.integers(1, 4))):
                payments.append({
                    'payment_id': f'p{len(payments)}',
                    'customer_id': customer_id,
                    'amount': 0,
                })
        orders_df = pd.DataFrame(orders)
        payments_df = pd.DataFrame(payments)
        # Allocate the customer total across each child table independently.
        for customer_id, group in orders_df.groupby('customer_id'):
            total = next(row['total'] for row in customers if row['customer_id'] == customer_id)
            share = total // 2
            orders_df.loc[group.index, 'amount'] = [share] * (len(group) - 1) + [
                total - share * (len(group) - 1)
            ]
            payment_group = payments_df[payments_df['customer_id'] == customer_id]
            payments_df.loc[payment_group.index, 'amount'] = [share] * (len(payment_group) - 1) + [
                total - share * (len(payment_group) - 1)
            ]

        data = {
            'customers': pd.DataFrame(customers),
            'orders': orders_df,
            'payments': payments_df,
        }
        synthesizer = HMASynthesizer(metadata, verbose=False)
        synthesizer.add_constraints([
            ParentChildSum('customers', 'orders', 'total', 'amount'),
            ParentChildSum('customers', 'payments', 'total', 'amount'),
        ])
        synthesizer.fit(data)
        sampled = synthesizer.sample(scale=1.0)

        order_sums = sampled['orders'].groupby('customer_id')['amount'].sum()
        payment_sums = sampled['payments'].groupby('customer_id')['amount'].sum()
        totals = sampled['customers'].set_index('customer_id')['total']
        pd.testing.assert_series_equal(
            order_sums.reindex(totals.index, fill_value=0),
            totals,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            payment_sums.reindex(totals.index, fill_value=0),
            totals,
            check_names=False,
        )
