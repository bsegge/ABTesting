import random
from ABTest.ab_test import ABTest


class Demo:

    def __init__(self):
        self.measurement_type = None

    def run(self):
        self._welcome()
        for test in ["continuous", "discrete"]:
            res, cmetric, emetric = self._run_simulation(test)
            self._interpret_test(test, res, cmetric, emetric)

    def _welcome(self):
        print("\n\nWelcome!!!\n\n"
              "The purpose of this module is to demonstrate a couple examples of a/b tests using randomized\n"
              "simulated data. For the purposes of this demo, we will explore both continuous (think session duration)\n"
              "and discrete data (think binomial trials, such as retention <returned / churned>). For each example, we\n" 
              "will generate random data, then we will check for normality of the data (whether it was drawn from a\n" 
              "Gaussian distribution), then we will add some context around the data, and finally we will run a simulated\n"
              "demo to determine both statistical and practical significance.\n\n"
              
              "Note: Practical significance is a bit subjective, and business objectives and real-world context are usually\n"
              "needed for it to be concluded.", end='\n\n'
              )

    def _run_simulation(self, test):
        test = ABTest(
            measurement_type=test,
            forced_diff=False,
            pop_proportion=0.40,
            spread_mod=random.randint(5, 25)
        )
        return test.run()

    def _interpret_test(self, measurement_type: str, result: bool, control_metric: float, experiment_metric: float):
        if measurement_type == "continuous":
            metric = "session duration"
        else:
            metric = "click-thru-rate"
        diff = round(experiment_metric-control_metric, 2)
        if result:
            print(f"\nWe conclude there is a significant difference of {diff} in {metric}. Whether this difference\n"
                  f"is meaningful and we want to productionalize the treatment is up for discussion.")
        else:
            print(f"\n We did not conclude statistical significance, so the difference of: {diff} could be random")
