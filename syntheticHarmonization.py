import numpy as np
import os
from scipy.linalg import sqrtm as sqrtm
from scipy.linalg import logm as logm
from scipy.linalg import expm as expm
from numpy.linalg import inv as inv
import subprocess
import pandas as pd
import matplotlib.pyplot as plt


def loadFile(fil):
    fi=open(fil,'r')
    r=fi.readlines()
    fi.close()
    for i in range(0,len(r)):
        while r[i].endswith('\n') or r[i].endswith('\r') and len(r[i])>1:
             r[i]=r[i][:-1]
        if r[i]=='\r' or r[i]=='\n':
            r[i]=''
    return r
def saveFile(r,fil):
    fi=open(fil,'w')
    for i in range(0,len(r)):
        fi.write(r[i]+'\n')
    fi.close()


def visualization(a,grs,oup,ouq): 
 u,s,vh=np.linalg.svd(a,full_matrices=False,compute_uv=True)
 ord=np.argsort(-s)
 s=s[ord[:2]]
 u=u[:,ord[:2]]
 pos=u.dot(np.diag(s))
 
 xmmi=np.min(pos[:,0])
 xmma=np.max(pos[:,0])
 xmin=(xmmi+xmma)/2.0+(xmmi-xmma)/2.0*1.1
 xmax=(xmmi+xmma)/2.0+(xmma-xmmi)/2.0*1.1
 ymmi=np.min(pos[:,1])
 ymma=np.max(pos[:,1])
 ymin=(ymmi+ymma)/2.0+(ymmi-ymma)/2.0*1.1
 ymax=(ymmi+ymma)/2.0+(ymma-ymmi)/2.0*1.1
 
 
 ppos=pos[:len(grs),:]
 plt.clf()
 colors={0:'k.',1:'b.',2:'m.',3:'g.',4:'c.',5:'r.'}
 un=np.unique(grs)
 for g in un:
  plt.plot(ppos[grs==g,0],ppos[grs==g,1],colors[g])
 plt.xlim([ xmin,xmax ])  
 plt.ylim([ ymin,ymax ])
 plt.savefig(oup,bbox_inches='tight')
  

 ppos=pos[len(grs):,:]
 plt.clf()
 colors={0:'k.',1:'b.',2:'m.',3:'g.',4:'c.',5:'r.'}
 un=np.unique(grs)
 for g in un:
  plt.plot(ppos[grs==g,0],ppos[grs==g,1],colors[g])
 
 
 plt.xlim([ xmin,xmax ])  
 plt.ylim([ ymin,ymax ])
 plt.savefig(ouq,bbox_inches='tight')


def core(dimension_):

  nrep=10   #number of random datasets

  ###################################################
  # STEP 1 : random data set generation
  ###################################################
  print('STEP 1 : random data sets generation')
  if not os.path.isdir('synthetic'):
    os.mkdir('synthetic')
  for rep in range(1,1+nrep):   # generate 10 random data sets
    print('    generating data set '+str(rep))
    dim=int(dimension_)     # number of time series ni each synthetic scan
    nto=int(0.8*dim)
    nt=1000                 # number of time points (time series will contain a thousand temporal measures)
    nns=[200,100,50,50,75]  # group sizes: one large group with 200 scans, two medium groups of slightly different sizes(75 and 100 scans), and two small groups of 50 scans (total 475 scans) 
    ng=5                    # number of groups, so 5 groups  

    effect=0.75             # effect size versus noise amplitude
    noise=1.0

    # generation of the group effects
    ts=[]
    for g in range(0,ng):
      ct=np.random.normal(0.0,1.0,(dim,nto))
      ct=np.cov(ct)
      ts.append(ct)
    se=np.random.normal(0.0,1.0,(dim,nto))*effect*0.25
    se=np.cov(se)
    ae=np.random.normal(0.0,1.0,(dim,nto))*effect
    ae=np.cov(ae)

    gscale=[]
    for g in range(0,ng):
      a=np.random.uniform(-1.0,1.0)
      b=np.random.uniform(1.0,5.0)
      if a<0.0:
        gscale.append(1.0/b)
      else:
        gscale.append(b)
  
    if not os.path.isdir('synthetic/data_'+str(rep)):
      os.mkdir('synthetic/data_'+str(rep))

    # generation of the time series and demographics information 
    ready=['file,sex,age,group']
    cpt=0
    for g in range(0,ng):
      for i in range(0,nns[g]):
        age=np.random.uniform(0.0,1.0)     # random "age" between 0 and 1
        sex=np.random.uniform(0.0,1.0)     # random binary sex value
        if sex<=0.5:
          sex=0
        else:
          sex=1
        ready.append( os.getcwd().replace('\\','/')+'/synthetic/data_'+str(rep)+'/data_'+str(rep)+'_'+str(cpt)+'.npy,'+str(sex)+','+str(age)+','+str(g) )
    
        t=ts[g]+(sex*se+age*ae)*gscale[g]*gscale[g]
        tt=np.random.normal(0.0,1.0,(dim,nt))
        tt=sqrtm(t).dot(sqrtm(inv( np.cov(tt) ))).dot(tt)  +np.random.normal(0.0,1.0,(dim,nt))*noise

        np.save('synthetic/data_'+str(rep)+'/data_'+str(rep)+'_'+str(cpt)+'.npy',tt)  # saving the individual time series
        cpt=cpt+1

    saveFile(ready,'synthetic/data_'+str(rep)+'_ready.csv')    # demographics and group information

  ###################################################
  # STEP 2 : random data set harmonization
  ###################################################
  print('STEP 2 : random data set harmonization')
  for rep in range(1,1+nrep):   # for each data set: 3 harmonizations

    print(' '.join(['  ','python','harmonization.py','-i','synthetic/data_'+str(rep)+'_ready.csv','-o','synthetic/harmonized_mean_'+str(rep),'-f','bw','-m','mean','-b','id']))
    subprocess.run(['python','harmonization.py','-i','synthetic/data_'+str(rep)+'_ready.csv','-o','synthetic/harmonized_mean_'+str(rep),'-f','bw','-m','mean','-b','id'])
    
    print(' '.join(['  ','python','harmonization.py','-i','synthetic/data_'+str(rep)+'_ready.csv','-o','synthetic/harmonized_meanScale_'+str(rep),'-f','bw','-m','meanScale','-b','id']))
    subprocess.run(['python','harmonization.py','-i','synthetic/data_'+str(rep)+'_ready.csv','-o','synthetic/harmonized_meanScale_'+str(rep),'-f','bw','-m','meanScale','-b','id'])

    print(' '.join(['  ','python','harmonization.py','-i','synthetic/data_'+str(rep)+'_ready.csv','-o','synthetic/harmonized_'+str(rep),'-f','bw','-m','combat','-b','id']))
    subprocess.run(['python','harmonization.py','-i','synthetic/data_'+str(rep)+'_ready.csv','-o','synthetic/harmonized_combat_'+str(rep),'-f','bw','-m','combat','-b','id'])

    print('')
    
  ###################################################
  # STEP 3 : harmonization results visualization
  ###################################################
  print('STEP 3 : harmonization results visualization (PCA)')
  for rep in range(1,1+nrep):
    aa=np.load( 'synthetic/harmonized_combat_'+str(rep)+'/logvalues_before_harmonization.npy')
    bb=np.load( 'synthetic/harmonized_combat_'+str(rep)+'/logvalues_after_harmonization.npy')
    cc=np.concatenate((aa,bb),axis=0)
    groups=np.array(pd.read_csv('synthetic/data_'+str(rep)+'_ready.csv')['group'])
    visualization(cc,groups,'synthetic/harmonized_combat_'+str(rep)+'/log_connectomes_before_harmonization.jpg','synthetic/harmonized_combat_'+str(rep)+'/log_connectomes_after_harmonization.jpg')
    
    aa=np.load( 'synthetic/harmonized_mean_'+str(rep)+'/logvalues_before_harmonization.npy')
    bb=np.load( 'synthetic/harmonized_mean_'+str(rep)+'/logvalues_after_harmonization.npy')
    cc=np.concatenate((aa,bb),axis=0)
    groups=np.array(pd.read_csv('synthetic/data_'+str(rep)+'_ready.csv')['group'])
    visualization(cc,groups,'synthetic/harmonized_mean_'+str(rep)+'/log_connectomes_before_harmonization.jpg','synthetic/harmonized_mean_'+str(rep)+'/log_connectomes_after_harmonization.jpg')

    aa=np.load( 'synthetic/harmonized_meanScale_'+str(rep)+'/logvalues_before_harmonization.npy')
    bb=np.load( 'synthetic/harmonized_meanScale_'+str(rep)+'/logvalues_after_harmonization.npy')
    cc=np.concatenate((aa,bb),axis=0)
    groups=np.array(pd.read_csv('synthetic/data_'+str(rep)+'_ready.csv')['group'])
    visualization(cc,groups,'synthetic/harmonized_meanScale_'+str(rep)+'/log_connectomes_before_harmonization.jpg','synthetic/harmonized_meanScale_'+str(rep)+'/log_connectomes_after_harmonization.jpg')



################################################################################
if __name__ == "__main__":
  from argparse import ArgumentParser, RawTextHelpFormatter
  parser = ArgumentParser(description="Generates 10 random synthetic data sets, applies 3 harmonization methods, and generates a visualization of the results",formatter_class=RawTextHelpFormatter)
  parser.add_argument("-d", "--dimension",help="Dimension of the matrices to generate", required=True)
  
  args = parser.parse_args()
  core(args.dimension)
