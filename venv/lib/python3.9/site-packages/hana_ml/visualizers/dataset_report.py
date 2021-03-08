"""
This module contains report builders for dataset.

The following class is available:

    * :class:`DatasetReportBuilder`
"""

# pylint: disable=line-too-long
# pylint: disable=too-many-locals
# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
import time
import sys
import warnings
from enum import Enum, unique
import multiprocessing
import threading
import pandas as pd
from IPython.core.display import HTML, display
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from htmlmin.main import minify
from tqdm import tqdm
from hana_ml.algorithms.pal import stats
from hana_ml.dataframe import quotename
from hana_ml.visualizers.model_report import TemplateUtil, PlotUtil
from hana_ml.visualizers.eda import EDAVisualizer
from hana_ml.algorithms.pal.preprocessing import Sampling


class DatasetReportBuilder(object):
    """
    The DatasetReportBuilder instance can analyze the dataset and generate a report in HTML format. \n
    The instance will call the dropna method of DataFrame internally to handle the missing value of dataset.

    The generated report can be embedded in a notebook, including: \n
    - Overview
        - Dataset Info
        - Variable Types
        - High Cardinality %
        - Highly Skewed Variables
    - Sample
        - Top ten rows of dataset
    - Variables
        - Numeric distributions
        - Categorical distributions
        - Variable statistics
    - Data Correlations
    - Data Scatter Matrix


    Examples
    --------

    Create a DatasetReportBuilder instance:

    >>> from hana_ml.visualizers.dataset_report import DatasetReportBuilder
    >>> datasetReportBuilder = DatasetReportBuilder()

    Assume the dataset DataFrame is df and then analyze the dataset:

    >>> datasetReportBuilder.build(df, key="ID")

    Display the dataset report as a notebook iframe.

    >>> datasetReportBuilder.generate_notebook_iframe_report()

     .. image:: dataset_report_example.png

    """

    def __init__(self):
        self.__data_analyzer = None

    def build(self, data, key, scatter_matrix_sampling: Sampling = None):
        """
        Build a report for dataset.

        Parameters
        ----------
        data : DataFrame
            DataFrame to use to build the dataset report.
        key : str
            Name of ID column.
        scatter_matrix_sampling : :class:`~hana_ml.algorithms.pal.preprocessing.Sampling`, optional
            Scatter matrix sampling.
        """
        if key not in data.columns:
            raise Exception("The parameter 'key' value is invalid.")
        self.__data_analyzer = DataAnalyzer(data, key, scatter_matrix_sampling)
        self.__data_analyzer.generate_report_html()

    def generate_html_report(self, filename):
        """
        Save the dataset report as a html file.

        Parameters
        ----------
        filename : str
            Html file name.
        """
        if self.__data_analyzer is None:
            raise Exception('To generate a report, you must call the build method firstly.')

        TemplateUtil.generate_html_file('{}_dataset_report.html'.format(filename), self.__data_analyzer.get_report_html())

    def generate_notebook_iframe_report(self):
        """
        Render the dataset report as a notebook iframe.

        """
        if self.__data_analyzer is None:
            raise Exception('To generate a report, you must call the build method firstly.')

        print('\033[31m{}'.format('In order to review the dataset report better, '
                                  'you need to adjust the size of the left area or hide the left area temporarily!'))
        display(HTML(self.__data_analyzer.get_iframe_report_html()))


@unique
class VariableType(Enum):
    # categorical
    CAT = "CAT"
    # numeric
    NUM = "NUM"
    # date
    DATE = "DATE"


class GenerateSVGStrThread(threading.Thread):
    def __init__(self, name, run_funcs):
        threading.Thread.__init__(self)
        self.name = name
        self.run_funcs = run_funcs
        self.results = []

    def run(self):
        for func in self.run_funcs:
            result = ''
            try:
                result = func()
            except RuntimeError as e:
                result = e
            finally:
                self.results.append(result)

    def get_results(self):
        return self.results


class DataAnalyzer(object):
    def __init__(self, data, key, scatter_matrix_sampling: Sampling = None):
        self.data = data
        self.drop_na_data = self.data.dropna()
        self.scatter_matrix_data = self.data
        if scatter_matrix_sampling:
            self.scatter_matrix_data = scatter_matrix_sampling.fit_transform(data=self.data)
        else:
            if self.data.count() >= 1000:
                scatter_matrix_sampling = Sampling('simple_random_without_replacement', sampling_size=1000)
                self.scatter_matrix_data = scatter_matrix_sampling.fit_transform(data=self.data)
        self.key = key
        self.conn_context = self.data.connection_context

        self.variables = self.data.columns
        self.variables_count = len(self.variables)
        self.variables_dtypes = self.data.dtypes()
        self.variables_describe = self.data.describe()
        self.rows_count = int(self.variables_describe.head(1).collect()['count'])
        self.col_stats = self.variables_describe.collect()
        self.col_stats_names = list(self.col_stats.columns.delete(0))
        self.col_stats_dict = {}
        for i in self.col_stats.index:
            row = self.col_stats.loc[i]
            self.col_stats_dict[row.values[0]] = list(row.values[1:])

        self.warnings_missing = {}
        self.warnings_cardinality = {}

        self.numeric = [i for i in self.variables if self.data.is_numeric(i)]
        self.categorical = [i[0] for i in self.variables_dtypes if (i[1] == 'NVARCHAR') or (i[1] == 'VARCHAR')]
        self.date = [i[0] for i in self.variables_dtypes if (i[1] == 'DATE') or (i[1] == 'TIMESTAMP')]

        self.variable_2_type_dict = {}

        for variable in self.numeric:
            self.variable_2_type_dict[variable] = VariableType.NUM

        for variable in self.categorical:
            self.variable_2_type_dict[variable] = VariableType.CAT

        for variable in self.date:
            self.variable_2_type_dict[variable] = VariableType.DATE

        self.__report_html = None
        self.__iframe_report_html = None

    def get_type(self, variable):
        return self.variable_2_type_dict.get(variable)

    def get_dataset_info(self):
        stats_name = ['Number of rows', 'Number of variables']
        stats_value = [self.rows_count, self.variables_count]

        dataset_dropna_count = self.drop_na_data.count()
        missing = round((self.rows_count - dataset_dropna_count) / self.rows_count * 100, 1)
        stats_name.append('Missing cells(%)')
        stats_value.append(missing)

        memory_size = pd.DataFrame.memory_usage(self.data.collect()).sum()
        record_size = memory_size / self.rows_count
        stats_name.append('Total size in memory(KB)')
        stats_value.append(round(memory_size / 1024, 1))
        stats_name.append('Average row size in memory(B)')
        stats_value.append(round(record_size, 1))

        return stats_name, stats_value

    def get_scatter_matrix(self):
        warnings.simplefilter("ignore")

        fig_size = self.variables_count + 6
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        scatter_matrix(self.scatter_matrix_data.collect(), ax=ax, alpha=0.8)

        def close():
            plt.close()
        fig.close = close

        scatter_matrix_html = PlotUtil.plot_to_str(fig, image_format='pdf')
        return scatter_matrix_html

    def get_warnings_correlation(self):
        warnings_correlation = {}
        warnings_correlation_text = []

        if len(self.numeric) > 1:
            pearsonr_matrix = stats.pearsonr_matrix(data=self.drop_na_data, cols=self.numeric).collect()

            columns = list(pearsonr_matrix['ID'])
            column_size = len(columns)
            pair_warnings_correlation_dict = {}

            for row_index in range(0, column_size):
                key1 = columns[row_index]
                row_data = list(pearsonr_matrix.loc[row_index, :])
                row_data.remove(key1)
                for i in range(0, column_size):
                    key2 = columns[i]
                    value = row_data[i]
                    if key1 != key2:
                        pair_warnings_correlation_dict[str(key1 + '-' + key2)] = value

            for i, col in enumerate(self.numeric):
                for j in range(i+1, len(self.numeric)):
                    dfc = pair_warnings_correlation_dict.get(str(self.numeric[i] + '-' + self.numeric[j]))
                    if (i != j) and (abs(dfc) > 0.3):
                        warnings_correlation[self.numeric[i], self.numeric[j]] = dfc

        text = "There are {} pair(s) of variables that are show significant correlation:".format(len(warnings_correlation))
        warnings_correlation_text.append(text)
        for i in warnings_correlation:
            corr = warnings_correlation.get(i)
            if abs(corr) >= 0.5:
                text = "-  {} and {} are highly correlated, p = {:.2f}".format(i[0], i[1], warnings_correlation.get(i))
                warnings_correlation_text.append(text)
            elif 0.3 <= abs(corr) < 0.5:
                text = "-  {} and {} are moderately correlated, p = {:.2f}".format(i[0], i[1], warnings_correlation.get(i))
                warnings_correlation_text.append(text)
            else:
                pass

        all_li_html = ''
        li_html_template = '''
            <li class="nav-item">
              <a class="nav-link">
                {}
              </a>
            </li>
        '''
        correlation_page_card_footer_html_template = '''
            <div>
              <ul class="nav nav-pills flex-column">{}</ul>
            </div>
        '''
        for text in warnings_correlation_text:
            all_li_html = all_li_html + li_html_template.format(text)
        correlation_page_card_footer_html = correlation_page_card_footer_html_template.format(all_li_html)

        return correlation_page_card_footer_html

    def get_correlation(self):
        fig_size = self.variables_count * 1.5
        fig, ax = plt.subplots(figsize=(fig_size + 6, fig_size + 4))
        eda = EDAVisualizer(ax)
        ax, corr = eda.correlation_plot(data=self.data, key=self.key, corr_cols=self.numeric, label=True)

        def close():
            plt.close()
        fig.close = close

        correlation_html = PlotUtil.plot_to_str(fig, image_format='pdf')
        return correlation_html

    def get_variable_types(self):
        names = ['Numeric', 'Categorical', 'Date']
        values = [len(self.numeric), len(self.categorical), len(self.date)]

        return names, values

    def get_missing_values(self):
        # Missing Values %
        missing_threshold = 10
        for i in self.variables:
            query = 'SELECT SUM(CASE WHEN {0} is NULL THEN 1 ELSE 0 END) AS "nulls" FROM ({1})'
            pct_missing = self.conn_context.sql(query.format(quotename(i), self.data.select_statement))
            pct_missing = pct_missing.collect().values[0][0]
            pct_missing = pct_missing/self.rows_count
            if pct_missing > missing_threshold/100:
                self.warnings_missing[i] = pct_missing
        names = list(self.warnings_missing.keys())
        values = list(self.warnings_missing.values())

        return names, values

    def get_high_cardinality_variables(self):
        warnings_constant = {}
        card_threshold = 100
        for i in self.variables:
            query = 'SELECT COUNT(DISTINCT {0}) AS "unique" FROM ({1})'
            cardinality = self.conn_context.sql(query.format(quotename(i), self.data.select_statement))
            cardinality = cardinality.collect().values[0][0]
            if cardinality > card_threshold:
                self.warnings_cardinality[i] = (cardinality/self.rows_count)*100
            elif cardinality == 1:
                warnings_constant[i] = self.data.collect()[i].unique()

        for i in self.warnings_cardinality:
            if i in self.categorical:
                self.categorical.remove(i)

        names = list(self.warnings_cardinality.keys())
        values = list(self.warnings_cardinality.values())

        return names, values

    def get_highly_skewed_variables(self):
        skew_threshold = 0.5
        numeric = [i for i in self.variables if self.data.is_numeric(i)]

        warnings_skewness = {}
        cont, cat = stats.univariate_analysis(data=self.data, cols=numeric)
        for i in numeric:
            skewness = cont.collect()['STAT_VALUE']
            stat = 'STAT_NAME'
            val = 'skewness'
            var = 'VARIABLE_NAME'
            skewness = skewness.loc[(cont.collect()[stat] == val) & (cont.collect()[var] == i)]
            skewness = skewness.values[0]
            if abs(skewness) > skew_threshold:
                warnings_skewness[i] = skewness

        names = list(warnings_skewness.keys())
        values = list(warnings_skewness.values())

        return names, values

    def get_categorical_variable_distribution_data(self, column):
        pie_data = self.data.agg([('count', column, 'COUNT')], group_by=column).sort(column).collect()
        x_data = list(pie_data[column])
        y_data = list(pie_data['COUNT'])

        return x_data, y_data

    def get_numeric_variable_distribution_data(self, column, bins=20):
        data_ = self.data.dropna(subset=[column])
        query = "SELECT MAX({}) FROM ({})".format(quotename(column), data_.select_statement)
        maxi = self.conn_context.sql(query).collect().values[0][0]
        query = "SELECT MIN({}) FROM ({})".format(quotename(column), data_.select_statement)
        mini = self.conn_context.sql(query).collect().values[0][0]
        diff = maxi-mini
        bin_size = round(diff/bins)
        if bin_size < 1:
            bin_size = 1
        query = "SELECT {0}, FLOOR({0}/{1}) AS BAND,".format(quotename(column), bin_size)
        query += " '[' || FLOOR({0}/{1})*{1} || ', ".format(quotename(column), bin_size)
        query += "' || ((FLOOR({0}/{1})*{1})+{1}) || ')'".format(quotename(column), bin_size)
        query += " AS BANDING FROM ({}) ORDER BY BAND ASC".format(data_.select_statement)
        bin_data = self.conn_context.sql(query)
        bin_data = bin_data.agg([('count', column, 'COUNT'),
                                 ('avg', 'BAND', 'ORDER')], group_by='BANDING')
        bin_data = bin_data.sort('ORDER').collect()
        x_data = list(bin_data['BANDING'])
        y_data = list(bin_data['COUNT'])
        return x_data, y_data

    @staticmethod
    def convert_pandas_to_html(df):
        return df.to_html()\
            .replace('\n', '').replace('  ', '')\
            .replace(' class="dataframe"', 'class="table table-bordered table-hover"')\
            .replace('border="1"', '')\
            .replace(' style="text-align: right;"', '')\
            .replace('<th></th>', '<th style="width: 10px">#</th>')\
            .replace('</thead><tbody>', '')\
            .replace('<thead>', '<tbody>')

    def get_sample_html(self):
        sample_html = ''
        if self.rows_count >= 10:
            sample_html = DataAnalyzer.convert_pandas_to_html(self.data.head(10).collect())
        else:
            sample_html = DataAnalyzer.convert_pandas_to_html(self.data.collect())

        return sample_html

    def generate_report_html(self):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        threads = []
        thread_num = multiprocessing.cpu_count()
        pbar = tqdm(total=4, desc="Generating dataset report...", disable=False, file=sys.stdout, ncols=100, bar_format='{l_bar}%s{bar}%s{r_bar}' % ('\x1b[34m', '\x1b[39m'))

        variable_types = self.get_variable_types()
        high_cardinality_variables = self.get_high_cardinality_variables()
        pbar.update(1)

        # delete ID column
        if self.key:
            if self.key in self.numeric:
                self.numeric.remove(self.key)
            elif self.key in self.categorical:
                self.categorical.remove(self.key)
            elif self.key in self.date:
                self.date.remove(self.key)

        if thread_num == 2:
            thread = GenerateSVGStrThread(
                'Generating scatter matrix, data correlation, sample and highly_skewed_variables...',
                [self.get_scatter_matrix, self.get_correlation, self.get_warnings_correlation,
                 self.get_sample_html, self.get_highly_skewed_variables])
            threads.append(thread)
        elif thread_num >= 3:
            thread1 = GenerateSVGStrThread('Generating scatter matrix and sample...',
                                           [self.get_scatter_matrix, self.get_sample_html])
            thread2 = GenerateSVGStrThread('Generating data correlation and highly_skewed_variables...',
                                           [self.get_correlation, self.get_warnings_correlation, self.get_highly_skewed_variables])
            threads.append(thread1)
            threads.append(thread2)
        for thread in threads:
            thread.start()

        dataset_report_json = {}

        ul_html_template = '''
            <ul class="nav nav-pills flex-column">{}</ul>
        '''
        li_html_template = '''
            <li class="nav-item">
              <a class="nav-link">
                {}
                <span class="float-right">{}</span>
              </a>
            </li>
        '''
        all_li_html = ''
        dataset_info = self.get_dataset_info()
        for i in range(0, len(dataset_info[0])):
            stats_name = dataset_info[0][i]
            stats_value = dataset_info[1][i]
            all_li_html = all_li_html + li_html_template.format(stats_name, stats_value)
        dataset_info_html = ul_html_template.format(all_li_html)

        dataset_report_json['overview_page'] = {
            'charts': []
        }

        dataset_report_json['overview_page']['charts'].append({
            'element_id': 'overview_page_chart_1',
            'x_data': variable_types[0],
            'y_data': variable_types[1],
            'type': 'doughnut',
            'title': '\n'
        })

        dataset_report_json['overview_page']['charts'].append({
            'element_id': 'overview_page_chart_2',
            'x_data': variable_types[0],
            'y_data': variable_types[1],
            'type': 'bar',
            'title': '\n'
        })

        dataset_report_json['overview_page']['charts'].append({
            'element_id': 'overview_page_chart_3',
            'x_data': high_cardinality_variables[0],
            'y_data': high_cardinality_variables[1],
            'type': 'horizontalBar',
            'title': ''
        })

        variables_page_card_tools_menu_html = ''
        variables_page_card_tools_menu_html_template = \
            '<a class="dropdown-item" onclick="switchVariableContent(\'{}\')">{}</a>'
        variables_page_card_body_html = ''
        variables_page_card_body_html_template = '''
            <div class="row" id="{}">
             <div class="col-lg-{}" style="margin:0 auto">
              <div class="chart-responsive">
                <canvas id="{}"></canvas>
              </div>
             </div>
            </div>
        '''
        variables_page_card_footer_html = ''
        variables_page_card_footer_html_template = '''
            <div id="{}">
              <ul class="nav nav-pills flex-column">{}</ul>
            </div>
        '''
        variables_copy = self.variables.copy()
        variables_copy.remove(self.key)
        dataset_report_json['variables_page'] = {
            'variables': variables_copy,
            'child_page_ids': []
        }
        element_id_suffix = 0
        variable_stats_name_dict = {
            'count': 'Number of rows',
            'unique': 'Number of distinct values',
            'nulls': 'Number of nulls',
            'mean': 'Average',
            'std': 'Standard deviation',
            'median': 'Median',
            'min': 'Minimum value',
            'max': 'Maximum value',
            '25_percent_cont': '25% percentile when treated as continuous variable',
            '25_percent_disc': '25% percentile when treated as discrete variable',
            '50_percent_cont': '50% percentile when treated as continuous variable',
            '50_percent_disc': '50% percentile when treated as discrete variable',
            '75_percent_cont': '75% percentile when treated as continuous variable',
            '75_percent_disc': '75% percentile when treated as discrete variable'
        }
        for variable in self.variables:
            if variable == self.key:
                continue
            element_id_suffix = element_id_suffix + 1
            variable_type = self.variable_2_type_dict.get(variable)
            variable_distribution_data = None
            chart_type = 'bar'
            width_percent = 10
            if variable_type == VariableType.NUM:
                variable_distribution_data = self.get_numeric_variable_distribution_data(variable)
                bar_count = len(variable_distribution_data[0])
                if bar_count < 5:
                    width_percent = 4
                elif 5 <= bar_count < 10:
                    width_percent = 6
                elif 10 <= bar_count < 15:
                    width_percent = 8
            elif variable_type == VariableType.CAT:
                variable_distribution_data = self.get_categorical_variable_distribution_data(variable)
                chart_type = 'doughnut'
                width_percent = 6
            else:
                variable_distribution_data = [[], []]
                chart_type = 'doughnut'
                width_percent = 2

            element_id = 'variables_page_chart_{}'.format(element_id_suffix)
            dataset_report_json['variables_page'][variable] = {
                'element_id': element_id,
                'x_data': variable_distribution_data[0],
                'y_data': variable_distribution_data[1],
                'type': chart_type,
                'title': 'Distribution of {}'.format(variable)
            }
            child_page_id = 'variables_page_{}'.format(element_id_suffix)
            dataset_report_json['variables_page']['child_page_ids'].append(child_page_id)

            variables_page_card_tools_menu_html = \
                variables_page_card_tools_menu_html + \
                variables_page_card_tools_menu_html_template.format(child_page_id, variable)

            variables_page_card_body_html = \
                variables_page_card_body_html + \
                variables_page_card_body_html_template.format(child_page_id, width_percent, element_id)

            variable_stats = self.col_stats_dict[variable]
            all_li_html = ''
            for i in range(0, len(self.col_stats_names)):
                stats_value = variable_stats[i]
                stats_name = self.col_stats_names[i]
                all_li_html = all_li_html + li_html_template.format(variable_stats_name_dict[stats_name], stats_value)
            variables_page_card_footer_html = \
                variables_page_card_footer_html+\
                variables_page_card_footer_html_template.format('{}_footer'.format(child_page_id), all_li_html)
        pbar.update(1)

        for thread in threads:
            thread.join()

        scatter_matrix_html = None
        correlation_html = None
        warnings_correlation_html = None
        sample_html = None
        highly_skewed_variables = None
        if thread_num == 1:
            scatter_matrix_html = self.get_scatter_matrix()
            correlation_html = self.get_correlation()
            warnings_correlation_html = self.get_warnings_correlation()
            sample_html = self.get_sample_html()
            highly_skewed_variables = self.get_highly_skewed_variables()
        elif thread_num == 2:
            scatter_matrix_html = threads[0].get_results()[0]
            correlation_html = threads[0].get_results()[1]
            warnings_correlation_html = threads[0].get_results()[2]
            sample_html = threads[0].get_results()[3]
            highly_skewed_variables = threads[0].get_results()[4]
        elif thread_num >= 3:
            scatter_matrix_html = threads[0].get_results()[0]
            sample_html = threads[0].get_results()[1]
            correlation_html = threads[1].get_results()[0]
            warnings_correlation_html = threads[1].get_results()[1]
            highly_skewed_variables = threads[1].get_results()[2]

        # check RuntimeError
        if isinstance(scatter_matrix_html, RuntimeError):
            raise scatter_matrix_html
        if isinstance(correlation_html, RuntimeError):
            raise correlation_html
        if isinstance(warnings_correlation_html, RuntimeError):
            raise warnings_correlation_html

        dataset_report_json['overview_page']['charts'].append({
            'element_id': 'overview_page_chart_4',
            'x_data': highly_skewed_variables[0],
            'y_data': highly_skewed_variables[1],
            'type': 'horizontalBar',
            'title': ''
        })
        pbar.update(1)

        template = TemplateUtil.get_template('dataset_report.html')
        self.__report_html = template.render(
            dataset_name=self.data.name,
            start_time=start_time,
            dataset_info=dataset_info_html,
            sample=sample_html,
            scatter_matrix_content=scatter_matrix_html,
            correlation_page_card_body=correlation_html,
            correlation_page_card_footer=warnings_correlation_html,
            variables_page_card_tools=variables_page_card_tools_menu_html,
            variables_page_card_body=variables_page_card_body_html,
            variables_page_card_footer=variables_page_card_footer_html,
            dataset_report_json=dataset_report_json)
        self.__report_html = minify(self.__report_html,
                                    remove_all_empty_space=True,
                                    remove_comments=True,
                                    remove_optional_attribute_quotes=False)
        pbar.update(1)
        pbar.close()

    def get_report_html(self):
        return self.__report_html

    def get_iframe_report_html(self):
        if self.__iframe_report_html is None:
            self.__iframe_report_html = TemplateUtil.get_notebook_iframe(self.__report_html)
        return self.__iframe_report_html
