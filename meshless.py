#!/usr/bin/env python

# visualize meshless voronoi-like cells
# python by Ben Wibking <wibking.1@astronomy.ohio-state.edu>
# math is due to Gaburov & Nitadori 2010
# [this is the hydro method implemented in GIZMO (Hopkins 2014)]

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Visualize meshless voronoi-like cells.')

parser.add_argument('npoints', type=int, help='number of mesh-generating points')
parser.add_argument('seed', type=int, help='seed for random number generation')
parser.add_argument('sigma', type=float, help='width of kernel function')
parser.add_argument('subtitle', help='subtitle for the plot')

args = parser.parse_args()

def generate_2d_points(npoints,seed):
    np.random.seed([seed])
    return np.random.uniform(low=0.,high=1.,size=(npoints,2))

import math
invsqrt_twopi = 1./math.sqrt(2. * math.pi)

def weight_function(dx, dy, sigma):
    r_sq = dx*dx + dy*dy
    return invsqrt_twopi/sigma*np.exp(-r_sq/(sigma*sigma))

def compute_meshless_density(x, y, points, mypoint, sigma):
    # first compute inverse number density at (x,y)
    w_inverse = 0.
    for j in xrange(points.shape[0]):
        w_inverse += weight_function(x - points[j,0], y - points[j,1], sigma)

    # now compute the weighted number density due to point 'mypoint'
    return weight_function(x - mypoint[0], y - mypoint[1], sigma)/w_inverse

def plot_meshless(args):
    npoints = args.npoints
    points = generate_2d_points(npoints,args.seed)

    x = np.arange(0.,1.,.01)
    y = np.arange(0.,1.,.01)
    xx,yy = np.meshgrid(x,y)

    import prettyplotlib as ppl
    fig,ax = ppl.subplots()

    for mypoint in points:
        dens = compute_meshless_density(xx, yy, points, mypoint, args.sigma)
        plt.contourf(xx,yy,dens,alpha=1./npoints)

    ppl.scatter(ax,points[:,0],points[:,1],color='white')
    ax.autoscale(tight=True)
    plt.title(args.subtitle)
    plt.show()

plot_meshless(args)
