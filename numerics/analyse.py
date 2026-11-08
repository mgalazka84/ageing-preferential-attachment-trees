"""Compute the mass-equation roots and produce the figures and tables from the CSV tree summaries.
Run with Python, NumPy, SciPy and Matplotlib. Paths are relative to this file.
"""
from functools import lru_cache
from math import exp, expm1, log
from pathlib import Path
import csv
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import digamma, hyp2f1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "figures"
FIGURES.mkdir(exist_ok=True)
CHECK_ALPHAS = (0., .1, .25, .5, .75, .9, .95, .99)
CUT = log(2)
YMAX = 72.


@lru_cache(maxsize=250000)
def integrated_weight(alpha, y):
    """I_alpha(exp(-y)), using a series with argument at most 1/2 on either side."""
    if alpha == 0.:
        return y
    epsilon = 1. - alpha
    if y <= CUT:
        z = -expm1(-y)
        return z**epsilon / epsilon * hyp2f1(1., epsilon, epsilon+1., z)
    x = exp(-y)
    term = alpha * x
    correction = term
    for j in range(2, 128):
        term *= x * (alpha+j-1)/j
        correction += term/j
        if term/j < 2.e-17:
            break
    return y - digamma(epsilon) - np.euler_gamma - correction


def integrate_y(fun, lo=0., hi=YMAX):
    edges = [lo] + ([CUT] if lo < CUT < hi else []) + [hi]
    return sum(quad(fun, a, b, epsabs=3.e-13, epsrel=3.e-13, limit=300)[0]
               for a,b in zip(edges, edges[1:]))


@lru_cache(maxsize=None)
def exponent(alpha):
    if alpha == 0.:
        return .5
    epsilon = 1-alpha
    const = -digamma(epsilon)-np.euler_gamma
    def residual(b):
        mass = integrate_y(lambda y: exp(-y+b*integrated_weight(alpha,y)))
        tail = exp(b*const-(1-b)*YMAX)/(1-b)
        return mass + tail - 2.
    return brentq(residual, 0., min(.51, .71*epsilon), xtol=5.e-15)


def independent_mass(alpha, b):
    """Solve dQ/dy=(1-exp(-y))^-alpha and dM/dy=exp(-y+bQ).
    The change z=y^(1-alpha) regularises the first equation at zero.
    """
    epsilon = 1-alpha
    def initial(z, state):
        y = z**(1/epsilon)
        ratio = y/(-expm1(-y)) if y else 1.
        return [ratio**alpha/epsilon, y**alpha*exp(-y+b*state[0])/epsilon]
    first = solve_ivp(initial, [0., 1.], [0., 0.], method='DOP853',
                       rtol=2.e-12, atol=2.e-13)
    def remaining(y, state):
        return [(-expm1(-y))**(-alpha), exp(-y+b*state[0])]
    last = solve_ivp(remaining, [1., YMAX], first.y[:,-1], method='DOP853',
                     rtol=2.e-12, atol=2.e-13)
    if not first.success or not last.success:
        raise RuntimeError("independent quadrature failed")
    q,mass=last.y[:,-1]
    return mass + exp(-YMAX+b*q)/(1-b)


def degree_pmf(alpha,k):
    b=exponent(alpha)
    def density(y):
        hazard=b*integrated_weight(alpha,y)
        return exp(-y-hazard)*(-expm1(-hazard))**(k-1)
    return integrate_y(density)


def cohort_mean(alpha,a,bound):
    b=exponent(alpha)
    low=-log(bound)
    high=-log(a) if a>0 else YMAX
    mass=integrate_y(lambda y: exp(-y+b*integrated_weight(alpha,y)), low, high)
    if a==0:
        const=-digamma(1-alpha)-np.euler_gamma
        mass+=exp(b*const-(1-b)*YMAX)/(1-b)
    return mass/(bound-a)


def export_beta():
    with (RESULTS/'beta.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['alpha','beta','beta_over_epsilon','inverse_beta','gamma','independent_mass_residual'])
        for a in CHECK_ALPHAS:
            b=exponent(a)
            residual=independent_mass(a,b)-2.
            assert abs(residual)<2.e-9, (a,residual)
            writer.writerow([a,b,b/(1-a),1/b,1+1/b,residual])
    for k in range(1,17):
        assert abs(degree_pmf(0.,k)-4./(k*(k+1)*(k+2)))<5.e-13


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#d5d8da',linewidth=.45)
    ax.tick_params(labelsize=8,width=.6,length=3)
    ax.set_axisbelow(True)


def figures():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8.,
                         'axes.labelsize':8.3,'legend.fontsize':7.1,
                         'axes.linewidth':.65,'lines.linewidth':1.35,
                         'pdf.fonttype':42,'ps.fonttype':42})
    green='#17626c'; orange='#a24e26'; dark='#262831'
    x=np.linspace(0.,.99,71)
    bs=np.array([exponent(float(a)) for a in x])
    fig,axes=plt.subplots(1,2,figsize=(5.20,2.40))
    for ax in axes: style(ax); ax.set_xlabel(r'Ageing exponent $\alpha$'); ax.set_xlim(0,1)
    left,right=axes
    left.plot(x,bs,color=green,label=r'$\beta_\alpha$')
    z=np.linspace(0,.44,30);left.plot(z,.5-(1-log(2))*z,'--',color=dark,label='Tangent at zero')
    z=np.linspace(.62,1,30);left.plot(z,log(2)*(1-z),':',color=orange,label='Near-critical tangent')
    z=np.linspace(.7,1,30);left.plot(z,.806465994*(1-z),'-.',color='#6c7083',label='Dorogovtsev–Mendes')
    left.set_ylabel(r'Degree growth exponent $\beta_\alpha$'); left.set_ylim(0,.54)
    left.legend(frameon=False,loc='lower left',handlelength=1.6)
    right.plot(x,bs/(1-x),color=green)
    right.axhline(log(2),color=dark,linestyle='--')
    right.axhline(.8065,color=orange,linestyle=':')
    right.text(.04,.710,r'$\log 2$',color=dark)
    right.text(.04,.815,'Dorogovtsev–Mendes 0.8065',color=orange,fontsize=7.4)
    right.set_ylim(.48,.853)
    right.set_ylabel(r'$\beta_\alpha/(1-\alpha)$')
    left.set_title('(a) Growth exponent',loc='left',fontsize=8.8)
    right.set_title('(b) Critical coefficient',loc='left',fontsize=8.8)
    fig.subplots_adjust(left=.115,right=.977,bottom=.20,top=.87,wspace=.42)
    fig.savefig(FIGURES/'exponents.tmp', format='pdf');plt.close(fig)
    (FIGURES/'exponents.tmp').replace(FIGURES/'exponents.pdf')

    paths=[RESULTS / f'tree_{code:02d}.csv' for code in (0, 25, 50, 75, 90)]
    trees={}; summary=[]
    for path in paths:
        data=np.genfromtxt(path,names=True,delimiter=',')
        a=float(data['alpha'][0]);trees[a]=data
        for n in np.unique(data['n']):
            d=data[data['n']==n]
            normal=d['T_scaled']
            root=d['root_degree']/n**exponent(a)
            summary.append((a,int(n),len(d),np.mean(normal),np.std(normal,ddof=1)/np.sqrt(len(d)),1/exponent(a),
                            np.median(root),*np.quantile(root,[.25,.75]),np.mean(d['degree1']),degree_pmf(a,1)))
    with (RESULTS/'tree_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['alpha','n','runs','T_mean','T_standard_error','T_limit','root_scaled_median','root_scaled_q25','root_scaled_q75','leaf_mean','leaf_limit'])
        w.writerows(summary)

    fig,axes=plt.subplots(1,2,figsize=(5.20,2.56))
    left,right=axes
    for ax in axes:style(ax)
    for a,col,marker in [(0.,dark,'o'),(.5,green,'s'),(.9,orange,'^')]:
        vals=trees[a][trees[a]['n']==max(trees[a]['n'])]
        xmid=(np.arange(20)+.5)/20
        theory=np.array([cohort_mean(a,j/20,(j+1)/20) for j in range(20)])
        emp=np.array([np.mean(vals['cohort'+str(j+1)]) for j in range(20)])
        left.plot(xmid,theory,color=col,linewidth=1.0)
        left.plot(xmid,emp,marker=marker,ls='none',ms=3.0,mfc='white',mec=col,mew=.80,label=rf'$\alpha={a:g}$')
        ks=np.arange(1,13)
        pi=np.array([degree_pmf(a,int(k)) for k in ks])
        obs=np.array([np.mean(vals['degree'+str(k)]) for k in ks])
        right.plot(ks,pi,color=col,linewidth=1.)
        right.plot(ks,obs,marker=marker,ms=3.,mfc='white',mec=col,mew=.8,ls='none')
    left.set(xlabel='Relative birth time',ylabel='Mean degree in cohort',xlim=(0,1))
    left.legend(loc='upper right',frameon=False,handlelength=.9)
    left.set_title('(a) Equal-width cohorts',loc='left',fontsize=8.8)
    right.set(xlabel='Degree $k$',ylabel='Fraction of vertices',yscale='log',xlim=(.75,12.25))
    right.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    right.set_title('(b) Degree frequencies',loc='left',fontsize=8.8)
    fig.subplots_adjust(left=.105,right=.978,bottom=.19,top=.87,wspace=.5)
    fig.savefig(FIGURES/'cohorts-degrees.tmp', format='pdf');plt.close(fig)
    (FIGURES/'cohorts-degrees.tmp').replace(FIGURES/'cohorts-degrees.pdf')

if __name__=='__main__':
    from observables import export, export_table_rows
    export_beta()
    figures()
    export(RESULTS, FIGURES, exponent, integrated_weight, integrate_y, degree_pmf)
    export_table_rows(RESULTS, HERE.parent/"tables")
