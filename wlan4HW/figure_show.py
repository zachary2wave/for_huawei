import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%%
def show(channelset):
    NODES = {'N7_AP_196': (13, 12),
             'N7_AP_197': (19, 12),
             'N7_AP_212': (1, 2.5),
             'N7_AP_298': (4.5, 9.5),
             'N7_AP_299': (10.5, 10),
             'N7_AP_300': (16.5, 10),
             'N7_AP_301': (1.5, 9.5),
             'N7_AP_302': (7.5, 8.5),
             'N7_AP_303': (13.5, 8),
             'N7_AP_304': (19.5, 8.5),
             'N7_AP_305': (4.5, 7),
             'N7_AP_306': (10.5, 5.5),
             'N7_AP_307': (16.5, 4),
             'N7_AP_308': (1, 4.5),
             'N7_AP_309': (13.5, 5.5),
             'N7_AP_310': (19, 5.5),
             'N7_AP_311': (17.5, 6),
             'N7_AP_313': (19, 2),
             'N7_AP_334': (23.5, 12),
             'N7_AP_335': (22, 10),
             'N7_AP_336': (28, 10),
             'N7_AP_337': (34, 10),
             'N7_AP_338': (39.5, 10),
             'N7_AP_339': (23.5, 9.5),
             'N7_AP_340': (31, 9.5),
             'N7_AP_341': (35, 9),
             'N7_AP_342': (21, 6),
             'N7_AP_343': (27.5, 7),
             'N7_AP_344': (39.5, 7),
             'N7_AP_345': (24.5, 5.5),
             'N7_AP_346': (36, 6.5),
             'N7_AP_348': (39.5, 2),
             'N7_AP_359': (21, 6),
             'N7_AP_3': (32.5, 7.5),
             'N7_AP_9': (21, 2),
             'N7_AP_20': (17, 2),
             'N7_AP_57': (16, 6),
             'N7_AP_64': (8, 5),
             'N7_AP_79': (17, 8),
             'N7_AP_84': (32, 5),
             }

    aplist = ['N7_AP_197', 'N7_AP_212', 'N7_AP_298', 'N7_AP_299', 'N7_AP_300', 'N7_AP_301', 'N7_AP_302', 'N7_AP_303',
              'N7_AP_304', 'N7_AP_305', 'N7_AP_306', 'N7_AP_308', 'N7_AP_309', 'N7_AP_310', 'N7_AP_313', 'N7_AP_334',
              'N7_AP_335', 'N7_AP_336', 'N7_AP_337', 'N7_AP_338', 'N7_AP_339', 'N7_AP_340', 'N7_AP_341', 'N7_AP_342',
              'N7_AP_343', 'N7_AP_344', 'N7_AP_345', 'N7_AP_346', 'N7_AP_348', 'N7_AP_359', 'N7_AP_3', 'N7_AP_9',
              'N7_AP_20', 'N7_AP_57', 'N7_AP_64', 'N7_AP_79', 'N7_AP_84']
    index_new = [0, 1, 9, 10, 11, 12, 13, 14,
                 15, 16, 17, 19, 20, 21, 23, 37,
                 38,39, 40, 41, 42, 43, 44, 45,
                 46, 47, 48, 49, 50, 60, 61, 62,
                 63, 64,65, 66, 67]
    x_list = []
    y_list = []
    ap_list = []

    for ap in aplist:
        try:
            x_list.append(list(NODES[ap])[0])
            y_list.append(list(NODES[ap])[1])
            ap_list.append(ap)
        except:
            continue

    zb_xy = pd.DataFrame(index=ap_list, columns=["x", "y"])

    zb_xy["x"] = x_list
    zb_xy["y"] = y_list



    zb_xy.index = index_new

    group = channelset
    plt.figure('Draw1')
    marker = ["1", "o", "v", "<", "2", "h", "H", "s", "p", "*", "x", "+", "d", "D"]
    for j, gro in zip(range(13), group):
        px = []
        py = []
        c = np.random.rand(1)
        for ap in gro:
            px.append(zb_xy.loc[ap, "x"])
            py.append(zb_xy.loc[ap, "y"])
        plt.scatter(px, py, c=c, s=100)


