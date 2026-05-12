import unittest

from src.evaluation.benchmark_assets import industry_from_file_name
from src.utils.text_utils import guess_industry


class IndustryClassificationTests(unittest.TestCase):
    def test_ingest_and_eval_use_same_industry_classification(self) -> None:
        cases = {
            '半导体_2026-04-01_REF001_晶圆代工周报.pdf': 'semiconductor',
            '新能源_2026-04-01_REF002_储能行业观察.pdf': 'new_energy',
            '消费_2026-04-01_REF003_零售行业跟踪.pdf': 'consumer',
            '白酒_2026-04-01_REF004_高端白酒月报.pdf': 'liquor',
        }
        for file_name, expected in cases.items():
            with self.subTest(file_name=file_name):
                self.assertEqual(guess_industry(file_name), expected)
                self.assertEqual(industry_from_file_name(file_name), expected)


if __name__ == '__main__':
    unittest.main()
