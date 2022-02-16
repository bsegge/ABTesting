import pandas as pd
import numpy as np
import scipy.stats as stats


class SignificanceTests:

    def __init__(self, test_type: str, control_data: list, experiment_data: list):
        self.test_type = test_type
        self.control_data = control_data
        self.experiment_data = experiment_data


class ParametricSignificanceTests(SignificanceTests):

    def __init__(self, test_type, control_data, experiment_data):
        super().__init__(test_type, control_data, experiment_data)

    def _ttest(self):
        print(f"\tINFO: Control mean: {round(np.mean(self.control_data),2)}\n"
              f"\tINFO: Experiment mean: {round(np.mean(self.experiment_data),2)}")
        stat, p = stats.ttest_ind(self.control_data, self.experiment_data)
        return p

    def _anova(self):
        pass

    def run(self):
        return self._ttest()


class NonParametricSignificanceTests:

    def __init__(self, test_type, control_data, experiment_data):
        self.test_type = test_type
        self.p = None
        self.control_data = control_data
        self.experiment_data = experiment_data

    def _chi2(self):

        control_data = [[i[0], i[1], "control"] for i in self.control_data]
        experiment_data = [[i[0], i[1], "experiment"] for i in self.experiment_data]
        data = control_data+experiment_data

        chi2 = 0
        df = pd.DataFrame(data, columns=['measure', 'outcome', 'group'])
        rows = df['group'].unique()
        cols = df['outcome'].unique()

        crosstab = pd.crosstab(
            df['group'],
            df['outcome'],
            values=df['measure'],
            aggfunc=np.sum,
            margins=True,
            margins_name="Total"
        )
        crosstab = crosstab.astype('int64')

        print(f"Crosstab:\n{crosstab}")

        for col in cols:
            for row in rows:
                observed = int(crosstab[col][row])
                expected = int(crosstab[col]['Total']) * int(crosstab['Total'][row]) / int(crosstab['Total']['Total'])
                x2 = (observed - expected) ** 2 / expected
                chi2 += x2

        p = 1 - stats.norm.cdf(chi2, (len(rows)-1)*(len(cols)-1))
        return p

    def run(self):
        return self._chi2()

