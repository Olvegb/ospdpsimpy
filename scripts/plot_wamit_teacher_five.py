#!/usr/bin/env python3
"""Use only GunnerusL35_20201112.mat. Produce 0..180 deg WAMIT results every 10 degrees.

Exactly five PDF deliverables named after teacher's requested hydrodynamic
categories. The original Gunnerus 100-frequency .mat is the *only* data source.
All headings 0,10,...,180 degrees and wave-period T [s] are included.
No interpolation, FMU building, or physical simulation.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data_examples' / 'GunnerusL35_20201112.mat'
DEFAULT_OUTPUT = ROOT / 'src' / 'ospfmusim' / 'hull' / 'plots' / 'wamit_interpolation'
v=loadmat(SOURCE,simplify_cells=True)['vessel']
w=np.asarray(v['motionRAO']['w'],dtype=float).ravel()
wf=np.asarray(v['forceRAO']['w'],dtype=float).ravel()
wd=np.asarray(v['driftfrc']['w'],dtype=float).ravel()
wa=np.asarray(v['freqs'],dtype=float).ravel()
heads=np.rad2deg(np.asarray(v['headings'],dtype=float).ravel())
assert len(w)==100 and len(wa)==102 and np.array_equal(wa[1:-1],w)
assert np.allclose(w,wf) and np.allclose(w,wd)
assert np.allclose(heads,np.arange(0,360,10))
assert np.allclose(v['main']['CG'],[-1.71,0,0.7])
assert np.all(np.diff(w)>0) and w[0]>0
T=2*np.pi/w
order=np.argsort(T)
T=T[order]
DOFS=('Surge','Sway','Heave','Roll','Pitch','Yaw')
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':0.22,
                     'font.family':'DejaVu Sans','figure.facecolor':'white',
                     'savefig.facecolor':'white'})

def get_response(kind, field, i, heading_index):
    raw=np.asarray(v[kind][field][i],dtype=float)
    if raw.ndim==3:
        raw=raw[:,:,0]
    assert raw.shape==(100,36),(kind,field,i,raw.shape)
    y=raw[order,heading_index]
    assert np.all(np.isfinite(y))
    return y

def fig_rao(heading_index,kind):
    is_motion=kind=='motionRAO'
    fig,ax=plt.subplots(6,2,figsize=(16.5,11.7))
    beta=int(round(float(heads[heading_index])))
    fig.suptitle(
        f'Gunnerus | {beta:03d}° wave heading | '
        +('MOTION RAOs' if is_motion else 'FIRST-ORDER WAVE EXCITATION')
        + ' | 100 calculated WAMIT periods',
        fontsize=17,y=.992,weight='bold')
    for j,name in enumerate(DOFS):
        amplitude=get_response(kind,'amp',j,heading_index)
        phase=get_response(kind,'phase',j,heading_index)
        phase=((phase+np.pi)%(2*np.pi))-np.pi
        if is_motion:
            amplitude=amplitude*(180/np.pi if j>=3 else 1.0)
            unit='deg/m' if j>=3 else 'm/m'
        else:
            amplitude=amplitude/1000.
            unit='kNm/m' if j>=3 else 'kN/m'
        x,y=ax[j,0],ax[j,1]
        x.plot(T,amplitude,color='#12689c',lw=1.5)
        y.plot(T,phase,color='#ba6928',lw=1.25)
        x.set_title(f'{j+1}. {name}: amplitude',fontsize=10,weight='bold')
        y.set_title(f'{j+1}. {name}: phase',fontsize=10,weight='bold')
        x.set_ylabel(unit)
        y.set_ylabel('rad')
        for a in (x,y):
            a.set_xlim(2,30)
            a.set_xticks([5,10,15,20,25,30])
            a.tick_params(labelsize=7)
        if j==5:
            x.set_xlabel('Wave period T [s]')
            y.set_xlabel('Wave period T [s]')
    fig.subplots_adjust(top=.945,bottom=.067,left=.07,right=.975,hspace=.66,wspace=.19)
    return fig

def fig_drift(heading_index):
    beta=int(round(float(heads[heading_index])))
    fig,axes=plt.subplots(3,1,figsize=(11.7,8.3))
    fig.suptitle(f'Gunnerus | {beta:03d}° wave heading | SECOND-ORDER MEAN WAVE DRIFT\n'
                 'Diagonal second-order coefficients (not complete two-frequency QTF)',
                 fontsize=14,weight='bold',y=.994)
    labels=('Mean surge drift force','Mean sway drift force','Mean yaw drift moment')
    units=('N/m²','N/m²','Nm/m²')
    for j,ax in enumerate(axes):
        raw=np.asarray(v['driftfrc']['amp'][j],dtype=float)
        if raw.ndim==3:
            raw=raw[:,:,0]
        assert raw.shape==(100,36)
        ax.plot(T,raw[order,heading_index],color='#12689c',lw=1.6)
        ax.set_title(labels[j],fontsize=11,weight='bold')
        ax.set_ylabel(units[j])
        ax.set_xlim(2,30)
        ax.set_xticks([5,10,15,20,25,30])
    axes[-1].set_xlabel('Wave period T [s]')
    fig.subplots_adjust(top=.86,bottom=.09,left=.10,right=.97,hspace=.45)
    return fig

def fig_matrix(matrix_name,mode):
    mat=np.asarray(v[matrix_name],dtype=float)[:,:,1:-1][:,:,order]
    assert mat.shape==(6,6,100)
    title='Added mass A' if matrix_name=='A' else 'Potential radiation damping B'
    if mode=='diagonal':
        fig,ax=plt.subplots(3,2,figsize=(11.7,8.3))
        fig.suptitle(f'Gunnerus | {title} | diagonal coefficients | calculated 100 frequencies',
                     fontsize=15,weight='bold',y=.995)
        for j,a in enumerate(ax.flat):
            a.plot(T,mat[j,j],lw=1.5)
            a.set(title=f'{DOFS[j]} ({matrix_name}[{j+1},{j+1}])',
                  xlabel='Wave period T [s]',ylabel=matrix_name)
            a.set_xlim(2,30)
        fig.tight_layout(rect=[0,.01,1,.94])
    else:
        fig,axes=plt.subplots(6,6,figsize=(16.5,11.7),sharex=True)
        fig.suptitle(f'Gunnerus | {title} | full 6x6 matrix | calculated 100 frequencies',
                     fontsize=15,weight='bold',y=.99)
        for a in range(6):
            for b in range(6):
                ax=axes[a,b]
                ax.plot(T,mat[a,b],lw=1.,color='#12689c')
                ax.set_title(f'{matrix_name}[{a+1},{b+1}]',fontsize=8)
                ax.set_xlim(2,30)
                ax.tick_params(labelsize=6)
                if a==5:ax.set_xlabel('T [s]',fontsize=8)
        fig.subplots_adjust(top=.943,bottom=.075,left=.06,right=.975,hspace=.6,wspace=.43)
    return fig

def fig_roll_damping():
    fig,ax=plt.subplots(figsize=(11.7,8.3))
    b=np.asarray(v['B'],dtype=float)[3,3,1:-1][order]
    bv=np.asarray(v['Bv'],dtype=float)[3,3,1:-1][order]
    assert np.all(np.isfinite(b)) and np.all(np.isfinite(bv))
    ax.plot(T,b,lw=2,label='B[4,4] WAMIT potential/radiation')
    ax.plot(T,bv,lw=2,label='Bv[4,4] additional viscous (MSS)')
    ax.plot(T,b+bv,lw=2,ls='--',label='Total B+Bv')
    ax.set(xlabel='Wave period T [s]',ylabel='Roll damping [N m s/rad]',
           title='Gunnerus | Potential and additional viscous roll damping')
    ax.legend(loc='best')
    fig.tight_layout()
    return fig


def fig_matrix_row(matrix_name, row):
    mat=np.asarray(v[matrix_name],dtype=float)[:,:,1:-1][:,:,order]
    assert mat.shape == (6,6,100)
    title='Added mass A' if matrix_name=='A' else 'Potential radiation damping B'
    fig, axes=plt.subplots(3,2,figsize=(11.7,8.3))
    fig.suptitle(f'Gunnerus | {title} | {DOFS[row]} matrix row {row+1}',
                 fontsize=14,weight='bold',y=.99)
    for column, ax in enumerate(axes.flat):
        ax.plot(T,mat[row,column,:],lw=1.4,color='#12689c')
        ax.set_title(f'{matrix_name}[{row+1},{column+1}]: {DOFS[row]} / {DOFS[column]}',
                     fontsize=9)
        ax.set_xlabel('Wave period T [s]')
        ax.set_ylabel(f'{matrix_name}[{row+1},{column+1}] [SI]')
        ax.set_xlim(2,30)
        ax.set_xticks([5,10,15,20,25,30])
    fig.tight_layout(rect=[.02,.035,.99,.925])
    return fig


PDF_NAMES = [
    '01_Motion_RAOs.pdf',
    '02_Added_Mass.pdf',
    '03_Potential_Damping.pdf',
    '04_First_Order_Wave_Load_Coefficients.pdf',
    '05_Second_Order_Wave_Load_Coefficients.pdf'
]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    out=args.output.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    paths=[out/f for f in PDF_NAMES]
    # One single scrollable document per teacher-requested category.
    with PdfPages(paths[0]) as pdf:
        for heading in range(19):
            fig=fig_rao(heading,'motionRAO')
            pdf.savefig(fig)
            plt.close(fig)
    for path,key in [(paths[1],'A'),(paths[2],'B')]:
        with PdfPages(path) as pdf:
            fig=fig_matrix(key,'all')
            pdf.savefig(fig)
            plt.close(fig)
            for row in range(6):
                fig=fig_matrix_row(key,row)
                pdf.savefig(fig)
                plt.close(fig)
    with PdfPages(paths[3]) as pdf:
        for heading in range(19):
            fig=fig_rao(heading,'forceRAO')
            pdf.savefig(fig)
            plt.close(fig)
    with PdfPages(paths[4]) as pdf:
        for heading in range(19):
            fig=fig_drift(heading)
            pdf.savefig(fig)
            plt.close(fig)
    info={
        'source_mat':str(SOURCE),
        'source_sha256':sha256(SOURCE.read_bytes()).hexdigest(),
        'calculated_frequencies':int(len(w)),
        'matrix_frequencies_including_synthetic_limits':int(len(wa)),
        'wave_period_range_s':[float(T[0]),float(T[-1])],
        'wave_headings_deg':list(range(0,181,10)),
        'interpolation_performed':False,
        'synthetic_matrix_limits_plotted':False,
        'PDF_pages':dict(zip(PDF_NAMES,[19,7,7,19,19])),
        'second_order_scope':'WAMIT .8 mean wave drift only, not full QTF'
    }
    (out/'plot_validation.json').write_text(json.dumps(info,indent=2)+'\n')
    print('DONE - 5 required plot PDFs from GunnerusL35_20201112.mat only')
    for path in paths:
        print(' ',path.name,'size',path.stat().st_size,'bytes')
    print('Real WAMIT angular frequencies:',f'{w[0]:.9f}',f'{w[-1]:.9f}','rad/s')
    print('Wave periods:',f'{T[0]:.2f}',f'{T[-1]:.2f}','seconds')


if __name__=='__main__':
    main()
