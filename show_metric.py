"""Прочитать метрики после обучения и вывести графики."""


from pathlib import Path

import matplotlib.pyplot as plt


def main(**kwargs):
    metrics_dir = kwargs['metrics_dir']
    metrics_files = metrics_dir.glob('*.txt')
    for i, metric_file in enumerate(metrics_files):
        with open(metric_file, 'r') as f:
            vals = list(map(lambda str_val: float(str_val.strip()),
                            f.readlines()))
            plt.figure(i)
            plt.plot(vals)
            plt.title(metric_file.name.split('.')[0])
    plt.show()


if __name__ == '__main__':
    metrics_dir = (Path(__file__).parent / 'Yolov7/work_dir/train_1/metrics')
    main(metrics_dir=metrics_dir)
