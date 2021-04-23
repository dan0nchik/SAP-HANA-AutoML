from hana_automl.utils.perfomance import Benchmark
from hana_automl.utils.connection import connection_context
import pandas as pd


def main():
    b = Benchmark(connection_context)
    b.run('../data/MiniBooNE.csv', task='binary_cls', label='signal', categorical=['signal'], grad_boost=True)
    b.plot_results()


if __name__ == "__main__":
    main()
