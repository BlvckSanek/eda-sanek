# my_explore_package/explore.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Explore:
    """
    A class for performing exploratory data analysis on a pandas DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    
    Methods
    -------
    numerical(self) -> pd.DataFrame:
        Returns a DataFrame with only numerical columns.
    
    categorical(self) -> pd.DataFrame:
        Returns a DataFrame with only categorical columns.
    
    missing(self):
        Displays a heatmap of missing values.
    
    df_joiner(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        Concatenates two DataFrames along columns.
    
    correlation_map(self):
        Displays a heatmap of the correlation matrix.
    
    plot_hist(self, column: str):
        Plots a histogram of the specified column.
    
    summary_statistics(self) -> pd.DataFrame:
        Returns summary statistics of the DataFrame.
    
    value_counts(self, column: str) -> pd.Series:
        Returns the counts of unique values in the specified categorical column.
    
    pair_plot(self):
        Displays a pair plot of numerical variables.
    
    box_plot(self, column: str):
        Displays a box plot of the specified column.
    
    bar_plot(self, column: str):
        Displays a bar plot of the specified categorical column.
    
    skewness_kurtosis(self) -> pd.DataFrame:
        Returns the skewness and kurtosis of numerical columns.
    
    detect_outliers(self, column: str) -> pd.Series:
        Detects outliers in the specified column using the IQR method.
    
    unique_values_count(self) -> pd.Series:
        Returns the count of unique values for each column.
    
    scatter_plot(self, x: str, y: str):
        Displays a scatter plot between two numerical columns.
    
    count_plot(self, column: str):
        Displays a count plot of the specified categorical column.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Constructs all the necessary attributes for the Explore object.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to analyze.
        """
        self.df = df

    def numerical(self) -> pd.DataFrame:
        """
        Returns a DataFrame with only numerical columns.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only numerical columns.
        """
        return self.df.select_dtypes(include=['int64', 'float64'])

    def categorical(self) -> pd.DataFrame:
        """
        Returns a DataFrame with only categorical columns.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only categorical columns.
        """
        return self.df.select_dtypes(include=['object'])

    def missing(self):
        """
        Displays a heatmap of missing values.

        Returns
        -------
        None
        """
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.show()

    def df_joiner(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenates two DataFrames along columns.

        Parameters
        ----------
        df1 : pd.DataFrame
            The first DataFrame.
        
        df2 : pd.DataFrame
            The second DataFrame.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame.
        """
        return pd.concat([df1, df2], axis=1)

    def correlation_map(self):
        """
        Displays a heatmap of the correlation matrix.

        Returns
        -------
        None
        """
        numerical_df = self.numerical()
        f, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(numerical_df.corr(), annot=True, fmt='.2f', ax=ax)
        plt.show()

    def plot_hist(self, column: str):
        """
        Plots a histogram of the specified column.

        Parameters
        ----------
        column : str
            The column to plot.

        Returns
        -------
        None
        """
        plt.hist(self.df[column])
        plt.show()

    def summary_statistics(self) -> pd.DataFrame:
        """
        Returns summary statistics of the DataFrame.

        Returns
        -------
        pd.DataFrame
            Summary statistics of the DataFrame.
        """
        return self.numerical().describe()

    def value_counts(self, column: str) -> pd.Series:
        """
        Returns the counts of unique values in the specified categorical column.

        Parameters
        ----------
        column : str
            The column to analyze.

        Returns
        -------
        pd.Series
            Counts of unique values.
        """
        return self.df[column].value_counts()

    def pair_plot(self):
        """
        Displays a pair plot of numerical variables.

        Returns
        -------
        None
        """
        numerical_df = self.numerical()
        sns.pairplot(numerical_df)
        plt.show()

    def box_plot(self, column: str):
        """
        Displays a box plot of the specified column.

        Parameters
        ----------
        column : str
            The column to plot.

        Returns
        -------
        None
        """
        sns.boxplot(data=self.df[column].dropna())
        plt.show()

    def bar_plot(self, column: str):
        """
        Displays a bar plot of the specified categorical column.

        Parameters
        ----------
        column : str
            The column to plot.

        Returns
        -------
        None
        """
        sns.barplot(x=self.df[column].value_counts().index, y=self.df[column].value_counts().values)
        plt.show()

    def skewness_kurtosis(self) -> pd.DataFrame:
        """
        Returns the skewness and kurtosis of numerical columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with skewness and kurtosis of numerical columns.
        """
        numerical_df = self.numerical()
        return pd.DataFrame({
            'skewness': numerical_df.skew(),
            'kurtosis': numerical_df.kurt()
        })

    def detect_outliers(self, column: str) -> pd.Series:
        """
        Detects outliers in the specified column using the IQR method.

        Parameters
        ----------
        column : str
            The column to analyze.

        Returns
        -------
        pd.Series
            Boolean series where True indicates an outlier.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.df[column] < (Q1 - 1.5 * IQR)) | (self.df[column] > (Q3 + 1.5 * IQR))
        return outliers

    def unique_values_count(self) -> pd.Series:
        """
        Returns the count of unique values for each column.

        Returns
        -------
        pd.Series
            Series with counts of unique values for each column.
        """
        return self.df.nunique()

    def scatter_plot(self, x: str, y: str):
        """
        Displays a scatter plot between two numerical columns.

        Parameters
        ----------
        x : str
            The column to use for the x-axis.
        
        y : str
            The column to use for the y-axis.

        Returns
        -------
        None
        """
        sns.scatterplot(x=self.df[x], y=self.df[y])
        plt.show()

    def count_plot(self, column: str):
        """
        Displays a count plot of the specified categorical column.

        Parameters
        ----------
        column : str
            The column to plot.

        Returns
        -------
        None
        """
        sns.countplot(data=self.df, x=column)
        plt.show()
