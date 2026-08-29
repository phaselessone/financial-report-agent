from __future__ import annotations

import unittest

from src.structured.metric_registry import (
    all_metric_specs,
    extract_metrics,
    match_metric,
    metric_dimension,
    metric_label,
    metric_value_types,
    metrics_compatible,
)
from src.structured.schema import Metric, ValueType


class TestMetricRegistry(unittest.TestCase):
    def test_legacy_metric_label_preserved(self) -> None:
        self.assertEqual(metric_label(Metric.REVENUE), "营业收入")

    def test_new_metric_alias_resolves(self) -> None:
        self.assertEqual(match_metric("公司2025年营业利润为10亿元"), Metric.OPERATING_PROFIT)
        self.assertEqual(match_metric("公司2025年总资产为100亿元"), Metric.TOTAL_ASSETS)

    def test_extract_multiple_metrics(self) -> None:
        self.assertEqual(
            extract_metrics("比较贵州茅台和五粮液2025年营业收入及净利润"),
            (Metric.REVENUE, Metric.NET_PROFIT),
        )

    def test_dimension_lookup(self) -> None:
        self.assertEqual(metric_dimension(Metric.NET_MARGIN), "percentage")
        self.assertEqual(metric_dimension(Metric.CAPEX), "currency")

    def test_metric_compatibility(self) -> None:
        self.assertTrue(metrics_compatible([Metric.REVENUE, Metric.NET_PROFIT, Metric.CAPEX]))
        self.assertTrue(metrics_compatible([Metric.GROSS_MARGIN, Metric.ROE, Metric.NET_MARGIN]))
        self.assertFalse(metrics_compatible([Metric.REVENUE, Metric.GROSS_MARGIN]))

    def test_all_metric_specs_cover_enum(self) -> None:
        self.assertEqual(len(all_metric_specs()), len(Metric))

    def test_registry_binds_each_metric_to_its_own_spec(self) -> None:
        expected = {
            Metric.REVENUE: ("营业收入", "currency"),
            Metric.GROSS_MARGIN: ("毛利率", "percentage"),
            Metric.EPS: ("每股收益", "per_share"),
            Metric.TOTAL_ASSETS: ("总资产", "currency"),
            Metric.FREE_CASH_FLOW: ("自由现金流", "currency"),
        }
        for metric, (label, dimension) in expected.items():
            with self.subTest(metric=metric):
                self.assertEqual(metric_label(metric), label)
                self.assertEqual(metric_dimension(metric), dimension)

    def test_adjusted_is_registered_as_a_supported_value_type(self) -> None:
        self.assertIn(ValueType.ADJUSTED, metric_value_types(Metric.REVENUE))


if __name__ == "__main__":
    unittest.main()
