from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson


def interpret_results(func, alpha: float = 0.05):
    def inner(data):
        stat_dict, p_dict = func(data)
        list_of_results = []
        for k, v in p_dict.items():
            if k == 'anderson':
                if v.statistic < v.critical_values[2]:
                    list_of_results.append(True)
                else:
                    list_of_results.append(False)
            else:
                if v > alpha:
                    list_of_results.append(True)
                else:
                    list_of_results.append(False)
        if list_of_results.count(True) >= 2:
            print("\tINFO: Failed to reject h0 with assumption that the data are normal. Data look Gaussian. "
                  "A parametric test is recommended")
            return True
        else:
            print("\tINFO: Reject h0. Data do not look Gaussian. A non-parametric test is recommended")
            return False

    return inner


@interpret_results
def test_normality(data):
    dict_of_p = {'shapiro': shapiro(data)[1], 'dagostino': normaltest(data)[1], 'anderson': anderson(data)}
    dict_of_stats = {'shapiro': shapiro(data)[0], 'dagostino': normaltest(data)[0]}
    return dict_of_stats, dict_of_p
