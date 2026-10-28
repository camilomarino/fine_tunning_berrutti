import unittest

from reporting.berrutti.audit_optimizer_contract import optimizer_contract


class OptimizerContractTests(unittest.TestCase):
    def group(self):
        return {
            "params": [0, 1],
            "lr": 1e-5,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
        }

    def test_schedule_is_excluded_but_parameter_order_remains(self):
        a = self.group()
        b = {**a, "lr": 2e-6}
        self.assertEqual(optimizer_contract([a]), optimizer_contract([b]))
        self.assertEqual(optimizer_contract([a])[0]["params"], [0, 1])

    def test_changed_betas_epsilon_decay_or_algorithm_rejected(self):
        for key, value in (
            ("betas", (0.8, 0.999)),
            ("eps", 1e-6),
            ("weight_decay", 0.01),
            ("amsgrad", True),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                optimizer_contract([{**self.group(), key: value}])

    def test_duplicate_parameter_membership_rejected(self):
        with self.assertRaises(ValueError):
            optimizer_contract([self.group(), self.group()])


if __name__ == "__main__":
    unittest.main()
