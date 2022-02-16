from random import randint, uniform
import scipy.stats as st
import numpy as np

from SignificanceTesting.significance_tests import NonParametricSignificanceTests, \
    ParametricSignificanceTests
from Utility.data_simulation import simulate_data
from Utility.distribution_validation import test_normality


class ABTest:

    def __init__(self,
                 measurement_type: str,
                 forced_diff: bool = True,
                 pop_proportion: float = 0.40,
                 alpha: float = 0.05,
                 power: float = 0.80,
                 expected_diff_mean=1,
                 expected_diff_prop=0.04,
                 population_size: int = 10000,
                 spread_mod: int = 5,
                 shift_mod: int = 50,
                 ):
        self.measurement_type = measurement_type
        self.forced_diff = forced_diff
        self.control_shift_mod = randint(0, 100)
        if self.forced_diff:
            self.experiment_shift_mod = self.control_shift_mod + 25
        else:
            self.experiment_shift_mod = randint(0, 100)
        self.pop_proportion = pop_proportion
        self.alpha = alpha
        self.power = power
        self.expected_difference_mean = expected_diff_mean
        self.expected_difference_proportion = expected_diff_prop
        self.confidence_interval = 1-self.alpha
        self.population_size = population_size
        self.spread_mod = spread_mod
        self.shift_mod = shift_mod

    def run(self):
        test_metric = 'click-thru-rate' if self.measurement_type.lower() == 'discrete' else 'session duration'
        print(f"\n\nRunning test with data type: {self.measurement_type}\n"
              f"Forced difference applied: {self.forced_diff}\n"
              f"Metric: {test_metric}\n"
              f"Simulated population size: {self.population_size}\n\n"
              f"For this demo, we will measure if there is a significant difference in {test_metric} between our \n"
              f"control and experiment groups, with a control population size "
              f"of {self.population_size} {'and we will force a difference' if self.forced_diff else ''}\n")
        control_data, experiment_data = self._generate_data()
        print("STEP 03: Testing normality...")
        normal = self._test_normality(experiment_data)
        if self.measurement_type == 'continuous' and not normal:
            print(f"Simulated data are not normally distributed. Run demo again")
            return
        if normal:
            p = self._test_significance(control_data, experiment_data, type="parametric")
        else:
            p = self._test_significance(control_data, experiment_data, type="non-parametric")

        print(f"\tINFO: P-value: {p}")
        conclusion = "Failed to reject the null hypothesis. There does not seem to be a difference"
        test_result = False
        if p <= self.alpha:
            conclusion = "Null Hypothesis is rejected. There is a significant difference"
            test_result = True

        print(f"\nTEST RESULT: {conclusion}")
        if self.measurement_type == "continuous":
            control_metric = round(np.mean(control_data), 2)
            experiment_metric = round(np.mean(experiment_data), 2)
        else:
            control_metric = round(control_data[0][0]/(control_data[1][0]+control_data[0][0]), 2)
            experiment_metric = round(experiment_data[0][0]/(experiment_data[1][0]+experiment_data[0][0]), 2)

        return test_result, control_metric, experiment_metric

    def _generate_data(self):
        if self.forced_diff:
            exp_prob = 1 if self.pop_proportion * 1.5 < 1 else round(self.pop_proportion * 1.5, 2)
        else:
            exp_prob = round(uniform(0, 1), 2)
        print("STEP 01: Generating control data...")
        control_data = simulate_data(
            n=self.population_size,
            distribution=self._measurement_type(),
            spread_mod=self.spread_mod,
            shift_mod=self.control_shift_mod,
            prob=self.pop_proportion
        )
        print("STEP 02: Generating experiment data...")
        experiment_data = simulate_data(
            n=self._estimated_sample_size(control_data),
            distribution=self._measurement_type(),
            shift_mod=self.experiment_shift_mod,
            prob=exp_prob
        )
        return control_data, experiment_data

    @staticmethod
    def _test_normality(data):
        if isinstance(data[0], tuple):
            print(f"\tINFO: Data are drawn from binomial distribution. A non-parametric test is recommended")
            return
        return test_normality(data)

    @staticmethod
    def _test_significance(control_data, experiment_data, type: str):
        print("STEP 04: ", end='')
        if type == "parametric":
            print(f"Running {type} test of significance...")
            test = ParametricSignificanceTests(type, control_data, experiment_data)
            p = test.run()
        else:
            print(f"Running {type} test of significance...")
            test = NonParametricSignificanceTests(type, control_data, experiment_data)
            p = test.run()
        return p

    def _measurement_type(self):
        if self.measurement_type == "continuous":
            return "normal"
        elif self.measurement_type in ("binary", "binomial", "boolean", "true/false", "discrete"):
            return "binomial"
        else:
            return "random"

    def _estimated_sample_size(self, control_data):
        z, powerz = self._return_z_score()
        z_score_str = f"\tINFO: We used a Z-score of {z} and power of {self.power} drawn from the standard normal "\
                      f"distribution with alpha level: {self.alpha}"
        print(z_score_str)
        if self.measurement_type == "continuous":
            # diff between 2 means
            n = round((2 * ((z + powerz) ** 2) * np.std(control_data)) / (self.expected_difference_mean ** 2))
            estimate_str = f"\tINFO: Estimated sample size needed for expected difference of {self.expected_difference_mean} "\
                           f"and power level of {self.power} {n}"
            print(estimate_str)
        else:
            # confidence interval of a proportion
            n = round(((z ** 2) * self.pop_proportion * (1 - self.pop_proportion)) / ((self.expected_difference_proportion/2) ** 2))
            estimate_str = f"\tINFO: Estimated sample size needed for population proportion of {self.pop_proportion} "\
                           f"and expected difference of {self.expected_difference_proportion} with confidence interval of" \
                           f" {self.confidence_interval}: {n}"
            print(estimate_str)
        return int(n)

    def _return_z_score(self):
        z = round(st.norm.ppf(1 - (self.alpha / 2)), 2)
        powerz = round(abs(st.norm.ppf(1 - self.power)), 2)

        return z, powerz
