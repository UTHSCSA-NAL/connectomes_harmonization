import pandas as pd
import numpy as np
import os
from neuroCombat import neuroCombat
from scipy.linalg import sqrtm as sqrtm
from scipy.linalg import logm as logm
from scipy.linalg import expm as expm
from numpy.linalg import inv as inv


def normalization(si):
    on=np.ones((1,si.shape[1]))
    eps=0.000000001
    si=si-np.mean(si,axis=1,keepdims=True).dot(on)
    si=np.divide(si,np.maximum(np.sqrt(np.sum(si*si,axis=1,keepdims=True)).dot(on),eps*on)   )
    return si
def oas(c,n):
    p=c.shape[0]
    trcc=np.sum(c*c)
    trc=np.trace(c)
    la=((1.0-2.0/p)*trcc+trc*trc)/(n+1.0-2.0/p)/(trcc-trc*trc/p)
    la=min(la,1.0)
    return (1.0-la)*c+la*np.trace(c)/p*np.eye(p) ,la
def ts2corr(ts):
    ts=normalization(ts)
    c=ts.dot(np.transpose(ts))
    c,la=oas(c,ts.shape[1])
    return c

def bw_logm(m,c):
    return sqrtm(m.dot(c))+sqrtm(c.dot(m))-2.0*m
def bw_expm(m,v):
    si,u=np.linalg.eig(m)
    si=np.reshape(si,(1,m.shape[1]))
    on=np.ones((m.shape[0],1))
    sig=on.dot(si)+np.transpose( on.dot(si) )
    x=u.dot(np.divide( np.transpose(u).dot(v).dot(u) ,sig)).dot(np.transpose(u))
    t=x+np.eye(m.shape[0])
    return t.dot(m).dot(t)
def bw_barycenter(cs):
    niter=100
    m=cs[0]
    for i in range(1,len(cs)):
        m=m+cs[i]
    m=m/float(len(cs))
    for ii in range(0,niter):
        mm=sqrtm(m)
        n=sqrtm( mm.dot(cs[0]).dot(mm) )
        for i in range(1,len(cs)):
            n=n+sqrtm( mm.dot(cs[i]).dot(mm) )
        m=n/float(len(cs))
    return     
    
def ai_logm(m,c):
    mm=sqrtm(m)
    imm=inv(mm)
    return mm.dot( logm( imm.dot(c).dot(imm) ) ).dot(mm)
def ai_expm(m,v):
    mm=sqrtm(m)
    imm=inv(mm)
    return mm.dot( expm( imm.dot(v).dot(imm) ) ).dot(mm)
def ai_barycenter(cs):
    niter=100
    m=cs[0]
    for i in range(1,len(cs)):
        m=m+cs[i]
    m=m/float(len(cs))
    for ii in range(0,niter):
        mm=inv(sqrtm(m))
        n=logm( mm.dot(cs[0]).dot(mm) )
        for i in range(1,len(cs)):
            n=n+logm( mm.dot(cs[i]).dot(mm) )
        mm=sqrtm(m)  
        m=mm.dot( expm(n/float(len(cs))) ).dot(mm)
    return m    

def cut(m):
    return m[np.triu_indices(m.shape[0],k=0)]
def rebuild(v,n):
    m=np.zeros((n,n))
    m[np.triu_indices(n,k=0)]=v
    return m+np.transpose(m)-np.diag(np.diag(m))

def runNeurocombat(data,dem,gr):
  # Specifying the batch (scanner variable) as well as a biological covariate to preserve:
  covs={'batch':gr}
  for i in range(0,dem.shape[1]):
    covs['dem'+str(i)]=dem[:,i]
  covs=pd.DataFrame(covs)
  #Harmonization step:
  ret=neuroCombat(dat=np.transpose(data),covars=covs,batch_col='batch')
  datb=np.transpose(ret["data"])
  rr={}
  rr['zscaling_std']=np.squeeze(np.sqrt(ret['estimates']['var.pooled']))
  rr['zscaling_mean']=np.mean(ret['estimates']['stand.mean'],axis=1)
  rr['combat_mean']=ret['estimates']['gamma.star']
  rr['combat_std']=np.sqrt(ret['estimates']['delta.star'])
  return datb,rr

def ztransform(a):
  eps=0.99999
  return 0.5*np.log(np.divide( 1.0+eps*a,1.0-eps*a ))

################################################################################
def core(inp,oup,fra,met,bar):
 inp=pd.read_csv(inp)
 
 # ALL MATRICES AND COVARIATES TO PRESERVE
 al=[] 
 cols=list(inp.columns)
 cols.remove('file')
 cols.remove('group')
 ncols=len(cols)
 covar=np.zeros((inp.shape[0],ncols))
 groups=np.array(inp['group'])
 if inp.at[0,'file'].endswith('.npy'):
  ca=ts2corr(np.load(inp.at[0,'file']))
  n=ca.shape[0]
  c=cut(ca)
 else:
  ca=ts2corr(np.loadtxt(inp.at[0,'file']))
  n=ca.shape[0]
  c=cut(ca)
 val=np.zeros((inp.shape[0],c.shape[0]))
 
 for i in range(0,inp.shape[0]):
  for ico in range(0,len(cols)):
   covar[i,ico]=inp.at[i,cols[ico]]
  fil=inp.at[i,'file']
  if fil.endswith('.npy'):
   al.append(ts2corr(np.load(fil)))
  else:
   al.append(ts2corr(np.loadtxt(fil)))
 
 # BARYCENTER
 b=np.zeros((n,n))
 if bar=='id':
  b=np.eye(n)
 elif bar=='mean': # standard mean
  nal=len(al)
  for i in range(0,nal):
   b=b+al[i]/float(nal)
 elif bar=='frechet':
  if fra=='ai': 
   b=ai_barycenter(al)
  elif fra=='bw':
   b=bw_barycenter(al)
  elif fra=='no':  # standard mean
   nal=len(al)
   for i in range(0,nal):
    b=b+al[i]/float(nal)  
  elif fra=='fi':  # mean after fisher z-transform
   nal=len(al)
   for i in range(0,nal):
    b=b+ztransform(al[i])/float(nal)  

 # HARMONIZATION PREPARATION
 if   fra=='no':
  for i in range(0,val.shape[0]):
   val[i,:]=cut(al[i])
 elif fra=='fi':
  for i in range(0,val.shape[0]):  
   val[i,:]=cut(ztransform(al[i]))
 elif fra=='ai':
  for i in range(0,val.shape[0]):
   val[i,:]=cut(ai_logm(b,al[i]))   
 elif fra=='bw':
  for i in range(0,val.shape[0]):  
   val[i,:]=cut(bw_logm(b,al[i]))
 
 # HARMONIZATION 
 vam=np.zeros(val.shape)
 if   met=='mean':
    me=np.mean(val,axis=0,keepdims=True)
    grs=np.unique(groups)
    for gr in grs:
      meg=np.mean(val[groups==gr,:],axis=0,keepdims=True)
      vam[groups==gr,:]=val[groups==gr,:]+np.ones((len(groups[groups==gr]),1)).dot(me-meg)
      
 elif met=='meanScale':
    me=np.mean(val,axis=0,keepdims=True)
    grs=np.unique(groups)
    ngr=len(grs)
    scales=np.zeros((ngr))
    szs=np.zeros((ngr))
    for j in range(0,ngr):
      meg=np.mean(val[groups==grs[j],:],axis=0,keepdims=True)
      tmp=val[groups==grs[j],:]-np.ones((len(groups[groups==grs[j]]),1)).dot(meg)
      tmp=tmp*tmp
      scales[j]=np.sqrt(np.mean(tmp))
      szs[j]=len(groups[groups==grs[j]])
    scale=np.sum(scales*szs)/np.sum(szs)
    for j in range(0,ngr):
      meg=np.mean(val[groups==grs[j],:],axis=0,keepdims=True)
      meh=np.ones((len(groups[groups==grs[j]]),1)).dot(meg)
      vam[groups==grs[j],:]=(val[groups==grs[j],:]-meh)*scale/scales[j] +np.ones((len(groups[groups==grs[j]]),1)).dot(me)
 
 elif met=='combat':
  vam,model=runNeurocombat(val,covar,groups)

 
 
 # OUTPUT MATRIX RECONSTRUCTION AND SAVE
 if not os.path.isdir(oup):
  os.mkdir(oup)
 for i in range(0,inp.shape[0]):
  fi=inp.at[i,'file']
  fi=fi[fi.rfind('/')+1:]
  if fi.endswith('.npy'):
   np.save(oup+'/connectome_'+fi,rebuild(vam[i,:],n))
  else:
   np.savetxt(oup+'/connectome_'+fi,rebuild(vam[i,:],n),fmt='%.8.f')
   
 np.save(oup+'/logvalues_before_harmonization.npy',val)
 np.save(oup+'/logvalues_after_harmonization.npy' ,vam)
   
 
 
################################################################################
if __name__ == "__main__":
 from argparse import ArgumentParser, RawTextHelpFormatter
 parser = ArgumentParser(description="Harmonization of a set of fMRI time series",formatter_class=RawTextHelpFormatter)
 parser.add_argument("-i", "--input",help="Input CSV file containing at least one column 'file' storing absolute path written with / , pointing to files either stored as .npy matrices or text files saved using numpy.savetxt; one column 'group' indicating in what group the scan belongs; and at least one other column storing the floating point values of a covariate that has a linear effect on the connetomes that needs to be preserved during the harmonization, such as age or a binary value encoding sex (any number of covariates can be provided, as long as at least one covariate is present).", required=True)
 parser.add_argument("-o", "--output",help="Output folder where the harmonized connectomes will be stored")
 parser.add_argument("-f", "--framework",help="Riemannian framwork. Should be in the following list: no/fi/ai/bw (no transformation, Fisher z-transform, Affine-Invariant metric, Bures-Wasserstein metric)")
 parser.add_argument("-m", "--method",help="Harmonization method. Should be in the following list: mean/meanScale/combat (correcting for connectomes mean, meand and scale, using ComBat)")
 parser.add_argument("-b", "--barycenter",help="Barycenter matrix considered as a reference. Should be in the following list: id/mean/frechet  (identity matrix, mean connectome without Fisher z-transform, or Frechet geometric mean. When using the Fisher z-transform as a framework, the Frechet mean is replaced by the mean after Fisher z-transform. When the 'no transformation' framework is selected, the Frechet mean is replaced by the standard mean).")
  
 args = parser.parse_args()
 if not args.framework in ['no','fi','ai','bw']:
  print('Wrong harmonization framework: '+args.framework)
  print('   should be in no/fi/ai/bw')
  exit()
 if not args.method in ['mean','meanScale','combat']:
  print('Wrong harmonization method: '+args.method)
  print('   should be in mean/meanScale/combat')
  exit()
 if not args.barycenter in ['id','mean','frechet']:
  print('Wrong barycenter: '+args.barycenter)
  print('   should be in id/mean/frechet')
  exit()
  
 core(args.input,args.output,args.framework,args.method,args.barycenter)
  
