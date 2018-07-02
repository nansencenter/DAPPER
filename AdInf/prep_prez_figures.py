
##

from common import *
plt.style.use('AdInf/paper2.mplstyle')

def xtract_prop(list_of_strings,propID,cast=float,fillval=np.nan):
  "E.g. xtract_prop(R1.labels,'infl',float)"
  props = []
  for s in list_of_strings:
    x = re.search(r'.*'+propID+':(.+?)(\s|$)',s)
    props += [cast(x.group(1))] if x else [fillval]
  return array(props)

def optimz(RT,field):
  mu   = RT.mean_field(field)[0]
  best = np.nanmin(mu,0)
  return best

def cselect(RT,cond,field):
  RT = deepcopy(RT).split(cond)
  return RT.mean_field(field)[0][0]

##

OD = OrderedDict # short alias
DFLT = {'ls':'-','lw':1.5,'marker':'o','ms':4}
STYLES = [
    ('EnKF.*FULL'        , 'EnKF-$N$ full model'   , OD(c=blend_rgb('k' ,0.7),marker='s',markerfacecolor='none')),
    ('EnKF.*N:80.*FULL'  , ''                      , OD(marker='+')),
    ('EnKF.*FULL'        , -1                      , OD()),
    ('EnKF_pre.* infl:\?', 'ETKF tuned'            , OD(c='k',markerfacecolor='none')),
    ('EAKF_A07'          , 'EAKF adaptive'         , OD(c='C2',marker='s')),
    ('ETKF_Xplct'        , 'ETKF adaptive'         , OD(c='C1',marker='v')),
    ('EnKF_N_Xplct'      , 'EnKF-$N$ hybrid'       , OD(c='xkcd:azure')),
    ('var~f:\S*'         , ''                      , OD()),
    ('nu_f:\S*'          , ''                      , OD()),
    (' Cond:0'           , ''                      , OD()),
    ('poorly tuned'      , 'ETKF excessive'        , OD(c='k',marker='x')),
    ]

def style(label):
  "Apply matching styles in their order"
  D = DFLT.copy()
  D['label'] = label
  matches = [i for i,row in enumerate(STYLES) if re.search(row[0],label)]
  for i in matches:
    row = STYLES[i]
    D.update(row[2])
    if row[1] is -1:
      del D['label']
    elif row[1] is not None:
      D['label'] = re.sub(row[0],row[1],D['label'])
  return D

######

# Score transformation.
# Overwrite (below) to use
Skill = lambda x: x 

Singls  = OD()        # each label is here will be plotted 
Groups  = OD()        # for each group: only min (per xticks) is plotted
AxProps = OD()        # labels, etc


##############################
# LorenzUV
##############################

##############################
fname = 'LUV_vs_F_dtObs_015'
AxProps['xlabel'] = 'Forcing ($F\,$) — both for truth and DA'

R = ResultsTable('data/remote/AdInf/bench_LUV/F_run4[5-9]') # Full with new adapt
Singls['FULL']  = R.split('FULL').split('nu:1')
Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
Groups['EnKF_pre infl:?'].load('data/remote/AdInf/bench_LUV/F_run62')
R = ResultsTable('data/remote/AdInf/bench_LUV/F_run57') # V3 adapt, 64 rep, from munin
R.rm('EAKF_A07 damp:1')
R.rm('ETKF_Xplct infl:1.015')
R.rm('EnKF_N_Xplct.*nu_f:10000$')
R.load          ('data/remote/AdInf/bench_LUV/F_run82') # add _N, Cond=False
R.rm('EnKF_N_Xplct Cond:0.*nu_f:1000$')
R.rm('EnKF_N$')
Singls['R'] = R

AxProps['xlim'] = (6,27)
AxProps['xticks'] = arange(5,31)[::5]
AxProps['ylim'] = (0,1.0)


##############################
fname  = 'LUV_vs_c_dtObs=015'
AxProps['xlabel'] = 'Speed-scale ratio ($c\,$) — both for truth and DA'

R = ResultsTable('data/remote/AdInf/bench_LUV/c_run41') # All, V3 adapt, 32 rep
Singls['FULL']  = R.split('FULL').split('nu:1')
Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
R.mv(r'detp:1 ',r'')
R.rm('(Clim|Var|CHEAT)')
R.mv(r'detp:1 ',r'')
R.load('data/remote/AdInf/bench_LUV/c_run44') # add c=14,15 (adaptive only)
R.load('data/remote/AdInf/bench_LUV/c_run46') # add (_X nu_f=1e4) (_N_X nu_f=1e3) (_N_mod) (_N)
R.mv(r'^(\w+)(?!.*N:20)',r'\1 N:20')
R.mv(r' N:20',r'')
R.rm('EAKF_A07 damp:1')
R.rm('ETKF_Xplct infl:1.015')
R.rm('EnKF_N_mod')
R.rm('ETKF_Xplct.*nu_f:10000$')
R.rm('EnKF_N_Xplct.*nu_f:1000$')
R.rm('EnKF_N_Xplct.*nu_f:10000$')
R.rm('EnKF_N$')
Singls['R'] = R

Singls['R2'] = ResultsTable('data/remote/AdInf/bench_LUV/c_run49') # add Cond=0
Singls['R2'].mv('(.*)',r'\1 Cond:0')
Singls['R2'].rm('EnKF_N_Xplct nu_f:1000 ')


AxProps['xlim'] = (0.9,13)
AxProps['ylim'] = (0.3,0.45)
AxProps['xscale'] = 'log'


##############################
# Lorenz95
##############################
fname = 'L95_vs_F_N=20_dtObs=015'
AxProps['xlabel'] = 'Forcing ($F\,$) — DA assumes $F=8$'

R = ResultsTable('data/remote/AdInf/bench_L95/FTr_run1[45]') # Old?
Singls['FULL'] = R.split('FULL')
Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
R = ResultsTable('data/remote/AdInf/bench_L95/FTr_run19') # V3 adapt, 64 rep
R.rm('EAKF_A07 damp:1')
R.rm('ETKF_Xplct infl:1.015')
R.load          ('data/remote/AdInf/bench_L95/FTr_run24') # add _N_X nu_f=1e3
R.load          ('data/remote/AdInf/bench_L95/FTr_run26') # add _X nu_f=1e4
R.load          ('data/remote/AdInf/bench_L95/FTr_run27') # add _N_mod
R.mv(' liv~g:.','')
R.mv(' sto~u:.','')
R.rm('EnKF_N_mod')
R.rm('ETKF_Xplct.*nu_f:10000$')
R.rm('EnKF_N_Xplct.*nu_f:1000$')
R.rm('EnKF_N_Xplct.*nu_f:10000$')
Singls['R'] = R

Singls['R2'] = ResultsTable('data/remote/AdInf/bench_L95/FTr_run30') # add Cond=0
Singls['R2'].mv('(.*)',r'\1 Cond:0')
Singls['R2'].rm('EnKF_N_Xplct nu_f:1000 ')

AxProps['xticks'] = arange(7,9+0.5,0.5)
AxProps['ylim'] = (0.32,0.7)
AxProps['xlim'] = (7,9)


##############################
# Lorenz63
##############################

##############################
fname = 'L63_N=3_vs_Q'
AxProps['xlabel'] = 'Magnitude ($\mathbf{Q}_{ii}/\Delta\,$) of noise on truth'

#R.rm('(?!.*detp:0)')

R = ResultsTable('data/remote/AdInf/bench_L63/Q_run4')
Singls['FULL'] = R.split('FULL').split('nu:1').split('N:80')
R = ResultsTable('data/remote/AdInf/bench_L63/Q_run33') # Full, 32 repeat
Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
Singls['FULL N3'] = R.split('FULL').split('N:3')
R = ResultsTable('data/remote/AdInf/bench_L63/Q_run34') # V3 adapt, 64 rep
R.rm('EAKF_A07 damp:1')
R.rm('ETKF_Xplct infl:1.015')
R.load          ('data/remote/AdInf/bench_L63/Q_run37') # add (_X nu_f=1e4) (_N_X nu_f=1e3)
R.rm('ETKF_Xplct.*nu_f:10000$')
R.rm('EnKF_N_Xplct.*nu_f:1000$')
R.rm('EnKF_N_Xplct.*nu_f:10000$')
Singls['R'] = R

Singls['R2'] = ResultsTable('data/remote/AdInf/bench_L63/Q_run42') # add Cond=0
Singls['R2'].mv('(.*)',r'\1 Cond:0')
Singls['R2'].rm('EnKF_N_Xplct nu_f:1000 ')
Singls['R2'].mv(' N:3',r'')
Singls['R2'].mv(' liv~g:.','')
Singls['R2'].mv(' sto~u:.','')

AxProps['xscale'] = 'log'
AxProps['xscale'] = 'log'
AxProps['xlim'] = (1e-4,1e+1)
AxProps['ylim'] = (0.4,0.9)


##############################
# Plot
##############################


fig, ax = plt.subplots()

h = dict()

#---- RMSE

for lbl, RT in Groups.items():
  if RT:
    h[lbl], = ax.plot(RT.xticks,Skill(optimz(RT,'rmse_a')),**style(lbl))

# Sub-optimally tuned 
G = Groups['EnKF_pre infl:?']
infls = xtract_prop(G.labels,'infl')[np.nanargmin(G.mean_field('rmse_a')[0],0)]
infls_all = xtract_prop(G.labels,'infl')
OFS = 0.10 # inflation tuning offset
rmses = G.mean_field('rmse_a')[0]
rmvs  = G.mean_field('rmv_a' )[0]
iiS   = arange(len(G.xticks))
iiV   = [np.argmin(np.abs(infls[iS]+OFS-infls_all)) for iS in iiS]
h['poorly tuned'], = ax.plot(G.xticks, rmses[iiV,iiS], **style('poorly tuned'))

for RT in Singls.values():
  if RT and len(RT.labels):
    for iC,(row,name) in enumerate(zip(RT.mean_field('rmse_a')[0],RT.labels)): 
      h[name], = ax.plot(RT.xticks,Skill(row),**style(name))

#---- RMV
def vstyle(dct):
  dct['lw'] = 1
  dct['ls'] = '--'
  dct['marker'] = None
  try:    del dct['label']
  except: pass
  return dct

for lbl, G in Groups.items():
  if G:
    rmv  = G.mean_field('rmv_a')[0]
    best = np.nanargmin(G.mean_field('rmse_a')[0],0)
    rmv  = rmv[best,arange(rmv.shape[1])]
    h['RMV_'+lbl], = ax.plot(G.xticks,Skill(rmv),**vstyle(style(lbl)))

h['RMV_'+'poorly tuned'], = ax.plot(G.xticks, rmvs[iiV,iiS], **vstyle(style('poorly tuned')))

for RT in Singls:
  if 'FULL' in RT: continue
  else: RT = Singls[RT]
  if RT and len(RT.labels):
    for iC,(row,name) in enumerate(zip(RT.mean_field('rmv_a')[0],RT.labels)): 
      h['RMV_'+name], = ax.plot(RT.xticks,Skill(row),**vstyle(style(name)))


#---- AxProps
for k,v in AxProps.items(): ax.set(**{k:v})
ax.grid(color='k',alpha=0.6,lw=0.4,axis='both')
ax.legend()
ax.set_ylabel('RMSE')

##

savefig_n.index = 1
def tog(h,save=True,*a,**b):
  toggle_viz(h,prompt=True,*a,**b)
  if save: savefig_n('data/AdInf/figs/prez/'+fname+'_')

# Hide all elements
toggle_viz(h.values(),prompt=False) # legend=0 also nice

#tog([h[k] for k in h if re.search('EnKF_N N:20 .* FULL',k)])
#tog([h[k] for k in h if re.search('EnKF_N N:80 .* FULL',k)])
tog(h['EnKF_pre infl:?'])
tog(h['poorly tuned'])
tog(h['EAKF_A07 var~f:0.01'])
tog(h['ETKF_Xplct nu_f:1000'])
tog([h[k] for k in h if k.startswith('EnKF_N_Xplct')])
tog([h[k] for k in h if k.startswith('RMV')])

##
