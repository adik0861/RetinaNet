import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path
import re
import io

loss = Path.cwd().parent.joinpath('savefiles', 'checkpoints', 'loss.log').read_text()
loss = re.sub(r'[\]\[]', '', loss)
df = pd.read_csv(io.StringIO(loss), names=['epoch', 'iteration', 'cls_loss', 'box_loss', 'run_loss'])


def avg_loss(period):
    _df = df.groupby(df.index // period).mean()
    x = np.array(list(_df.index))
    y_cls = np.array(_df['cls_loss'].to_list())
    y_box = np.array(_df['box_loss'].to_list())
    plt.plot(x, y_cls, y_box)
    plt.show()

