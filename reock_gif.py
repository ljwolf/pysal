import matplotlib.pyplot as plt
import pysal as ps
from pysal.contrib import compact as com
import shutil as sh
import os 
import numpy as np

bads = [2,17,20,45]

chains = ps.open(ps.examples.get_path('columbus.shp')).read()

for i, chain in enumerate(chains):
    print('constructing chain {}'.format(i))
    pset = [point for part in chain.parts for point in part[:-1]]
    parray = np.array(pset)

    removed = []
    k = 0
    working = i not in bads
    if working:
        wlabel = 'working'
    else:
        wlabel = 'not_working'
    POINTS = [p for p in pset]

    pset = [pset[j] for j in com.mbc.ConvexHull(pset).vertices]

    min_x, min_y = parray.min(axis=0)
    max_x, max_y = parray.max(axis=0)

    mplaxes = [.95*min_x, 1.05*max_x, .95*min_y, 1.05*max_y]
    prefix = 'figures/{w}/{ch}'.format(w=wlabel, ch=i)

    fig = plt.figure(figsize=(10,10))
    ax1=fig.add_subplot(1,1,1)
    plt.plot([p[0] for p in POINTS], [p[1] for p in POINTS], 'k', linewidth=1)
    plt.plot([p[0] for p in pset], [p[1] for p in pset], 'm--', linewidth=1.5)
    plt.axis(mplaxes)
    plt.title("Shape: {ch}".format(ch=i))
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    plt.savefig(prefix + '/{}.png'.format(k))
    plt.clf() 
    done = False
    while not done:
        angles = [com.mbc._angle(*com.mbc._neighb(p,pset)) for p in pset]
        circles = [com.mbc._circle(*com.mbc._neighb(p,pset)) for p in pset]
        radii = [c[0] for c in circles]
        lexord = com.mbc.np.lexsort((radii, angles))
        lexmax = lexord[-1]

        if not angles[lexmax] <= com.mbc.PI/2.:
            removed.append((lexmax, pset.pop(lexmax)))
        else:
            done = True
        k+=1
        fig = plt.figure(figsize=(10,10))
        ax1=fig.add_subplot(1,1,1)
        plt.plot([p[0] for p in POINTS], [p[1] for p in POINTS], 'k', linewidth=1)
        plt.plot([p[0] for p in pset], [p[1] for p in pset], 'm--', linewidth=1.5)
        plt.plot(removed[-1][-1][0], removed[-1][-1][1], 'mo')
        for j, (_, (px, py)) in enumerate(removed[:-1]):
            plt.plot(px,py, 'ro', alpha=1./(len(removed)-j))
        solution = plt.Circle(circles[lexmax][-1], radius=circles[lexmax][0], 
                      fill=False, ec='c')
        ax1.add_patch(solution)
        plt.axis(mplaxes)
        plt.savefig(prefix + '/{}.png'.format(k))
        plt.close()
    os.system('convert -delay 150 -loop 0 {p}/*.png {p}/{i}.gif'.format(p=prefix, i=i))
