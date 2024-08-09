# tests/test_explore.py

import pytest
import pandas as pd
import numpy as np
from eda_sanek.explore import Explore
import matplotlib.pyplot as plt
import seaborn as sns

@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        'A': [1, 2, np.nan, 4],
        'B': ['x', 'y', 'z', 'w'],
        'C': [1.5, np.nan, 3.5, 4.5],
        'D': [10, 20, 30, 40]
    }
    return pd.DataFrame(data)

@pytest.fixture
def explore_instance(sample_dataframe):
    """Fixture to provide an Explore instance with the sample DataFrame."""
    return Explore(sample_dataframe)

def test_numerical(explore_instance):
    result = explore_instance.numerical()
    assert list(result.columns) == ['A', 'C', 'D'], "Numerical columns extraction failed."

def test_categorical(explore_instance):
    result = explore_instance.categorical()
    assert list(result.columns) == ['B'], "Categorical columns extraction failed."

def test_missing(explore_instance, mocker):
    mocker.patch('seaborn.heatmap')
    explore_instance.missing()
    sns.heatmap.assert_called_once()

def test_df_joiner(explore_instance, sample_dataframe):
    df1 = sample_dataframe[['A', 'B']]
    df2 = sample_dataframe[['C', 'D']]
    result = explore_instance.df_joiner(df1, df2)
    assert result.shape == (4, 4), "DataFrame concatenation failed."

def test_correlation_map(explore_instance, mocker):
    mocker.patch('matplotlib.pyplot.show')
    explore_instance.correlation_map()
    plt.show.assert_called_once()

def test_plot_hist(explore_instance, mocker):
    mocker.patch('matplotlib.pyplot.show')
    explore_instance.plot_hist('A')
    plt.show.assert_called_once()

def test_summary_statistics(explore_instance):
    result = explore_instance.summary_statistics()
    assert 'A' in result.columns, "Summary statistics calculation failed."

def test_value_counts(explore_instance):
    result = explore_instance.value_counts('B')
    assert result['x'] == 1, "Value counts calculation failed."

def test_pair_plot(explore_instance, mocker):
    mocker.patch('seaborn.pairplot')
    explore_instance.pair_plot()
    sns.pairplot.assert_called_once()

def test_box_plot(explore_instance, mocker):
    mocker.patch('matplotlib.pyplot.show')
    explore_instance.box_plot('A')
    plt.show.assert_called_once()

def test_bar_plot(explore_instance, mocker):
    mocker.patch('matplotlib.pyplot.show')
    explore_instance.bar_plot('B')
    plt.show.assert_called_once()

def test_skewness_kurtosis(explore_instance):
    result = explore_instance.skewness_kurtosis()
    assert 'skewness' in result.columns, "Skewness calculation failed."
    assert 'kurtosis' in result.columns, "Kurtosis calculation failed."

def test_detect_outliers(explore_instance):
    result = explore_instance.detect_outliers('A')
    assert result.sum() == 0, "Outlier detection failed."

def test_unique_values_count(explore_instance):
    result = explore_instance.unique_values_count()
    assert result['A'] == 3, "Unique values count calculation failed."

def test_scatter_plot(explore_instance, mocker):
    mocker.patch('matplotlib.pyplot.show')
    explore_instance.scatter_plot('A', 'C')
    plt.show.assert_called_once()

def test_count_plot(explore_instance, mocker):
    mocker.patch('matplotlib.pyplot.show')
    explore_instance.count_plot('B')
    plt.show.assert_called_once()
