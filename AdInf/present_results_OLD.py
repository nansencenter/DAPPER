# Script that loads batches of related data,
# makes seletions and grouping (database ops) on them,
# and presents them.
# The script is a sandbox, with quick and ephemeral changes, 
# and naming conventions that are not really thought-through,
# => it's not really legible.
# In order to understand anything,
# go through it line-by-line, printing the intermediate results,
# (e.g. >>> Singls['FULL'])

######

from common import *

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

def nonBase_EnKF_N(s):
  return s.startswith('EnKF_N') and not re.search('(FULL|CHEAT)',s)

# Score transformation.
# Overwrite (below) to use
Skill = lambda x: x 

OD      = OrderedDict # short alias
Singls  = OD()        # each label is here will be plotted 
Groups  = OD()        # for each group: only min (per xticks) is plotted
AxProps = OD()        # labels, etc
OnOff   = None


DFLT = {'ls':'-','marker':'o','ms':3}
STYLES = [
    ('EnKF.*FULL'       , None        , OD(c=blend_rgb('k' ,1.0))),
    ('EnKF.*CHEAT'      , None        , OD(c=blend_rgb('k' ,0.7))),
    ('Climatology'      , None        , OD(c=blend_rgb('k' ,0.5))),
    ('Var3D'            , '3D-Var'    , OD(c=blend_rgb('k' ,0.5))),
    ('EnKF .* infl:\?'  , None        , OD(c=blend_rgb('b' ,1.0))),
    ('EnKF_N .*infl:\?' , None        , OD(c=blend_rgb('b' ,1.0))),
    ('ETKF_M11'         , 'M11'       , OD(c=blend_rgb('C1',1.0))),
    ('EAKF_A07'         , 'A07'       , OD(c=blend_rgb('C2',1.0))),
    ('Xplct'            , None        , OD(c=blend_rgb('C3',1.0))),
    ('InvCS'            , None        , OD(c=blend_rgb('C4',1.0))),
    ('N_Xplct'          , None        , OD(c=blend_rgb('C5',1.0))),
    ('N_InvCS'          , None        , OD(c=blend_rgb('C8',1.0))),
    (r'(\w+?):\?'       , r'\1:OPT'   , OD()), # replace 'xxx:?' by 'xxx:OPT'
    ('detp:0'           , None        , OD(ls='--')),
    ('detp:2'           , None        , OD(ls=':')),
    ('nu:2'             , None        , OD(ls=':')),
    (r'( detp:1)'       , ''          , OD()),
    (r'( CLIP:0.9)'     , ''          , OD()),
    ]

def style(label):
  "Apply matching styles in their order"
  D = DFLT.copy()
  D['label'] = label
  matches = [i for i,row in enumerate(STYLES) if re.search(row[0],label)]
  for i in matches:
    row = STYLES[i]
    D.update(row[2])
    if i and row[1] is not None:
      D['label'] = re.sub(row[0],row[1],D['label'])
  return D




##############################
# Lorenz95
##############################

##############################
# AxProps['xlabel'] = 'Magnitude (var/Q) of noise on truth'
# 
# # R = ResultsTable('data/remote/AdInf/bench_L95/Q_run[3-7]')
# R = ResultsTable('data/remote/AdInf/bench_L95/Q_run10')
# 
# Singls['FULL'] = R.split('FULL').split('(N:20.*nu:2|N:80.*nu:1)')
# Singls['Base'] = R.rm('(Clim|Var)')
# 
# Groups['EnKF        infl:?'] = R.rm('^EnKF ' )
# Groups['EnKF_pre    infl:?'] = R.split('^EnKF_pre ' )
# Groups['EnKF_N nu:2 infl:?'] = R.rm('EnKF_N .*nu:2')
# Groups['EnKF_N      infl:?'] = R.rm('^EnKF_N ' )
# 
# # N30 = R.split('N:30')
# # Groups['M11 var~f:?']        = R.split('M11')
# # Groups['A07 var~f:?']        = R.split('A07')
# 
# # Groups['Xplicit nu_f:?']     = R.split('ETKF_Xplicit')
# # Singls['Adapt']              = R
# 
# AxProps['xlim'] = (1e-6,1e-1)
# AxProps['ylim'] = (0.17,0.35)
# AxProps['xscale'] = 'log'
# #AxProps['yscale'] = 'log'
# OnOff = array([1,1,1,0,0,0,1,0,1,0,1,0,0]).astype(bool)

##############################
# AxProps['xlabel'] = 'Forcing (F) truth'
# R = ResultsTable('data/remote/AdInf/bench_L95/FTr_run1')
# 
# Singls['FULL'] = R.split('FULL').split('N:80.*nu:1')
# Singls['Base'] = R.rm('(Clim|Var)')
# 
# Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
# 
# R.rm('(M11|A07).*infl:1 ')
# 
# x0 = Singls['FULL'].split('N:80 nu:1').mean_field('rmse_a')[0][0]
# x1 = optimz(Groups['EnKF_pre infl:?'],'rmse_a')
# Skill = lambda x: x-x1
# 
# AxProps['xlim'] = (7,9)
# OnOff = array([1,1,1,0,1,0,0,0,0,1,1,0,1,0]).astype(bool)


##############################
# AxProps['xlabel'] = 'Forcing (F)'
#  
# R = ResultsTable('data/remote/AdInf/bench_L95/F_run[23]')
# 
# Singls['FULL'] = R.split('FULL').split('(N:20.*nu:2|N:80.*nu:1)')
# Singls['Base'] = R.split('(Clim|Var)')
# 
# Groups['EnKF        infl:?'] = R.rm('^EnKF ' )
# Groups['EnKF_pre    infl:?'] = R.split('^EnKF_pre ' )
# Groups['EnKF_N nu:2 infl:?'] = R.rm('EnKF_N .*nu:2')
# Groups['EnKF_N      infl:?'] = R.rm('^EnKF_N ' )
# 
# AxProps['xlim'] = (1,40)
# AxProps['ylim'] = (0.1,10)
# AxProps['xscale'] = 'log'
# AxProps['yscale'] = 'log'
# AxProps['xticks'] = 1 + arange(40)[::3]
# AxProps['xticklabels'] = 1 + arange(40)[::3]
# AxProps['yticks'] = 0.1 * (1 + arange(30))
# AxProps['yticklabels'] = 0.1 * (1 + arange(30))

##############################
# AxProps['xlabel'] = 'Ensemble size (N)'
 

##############################
# LorenzUV Wilks
##############################
# AxProps['xlabel'] = 'Speed ratio (c)'
# 
# # Full run
# Base = ResultsTable('data/remote/AdInf/bench_LUV/c_run4')
# # Adds coordinate c=0.01. Unfortunately, uses slightly different infl values.
# Base.load('data/remote/AdInf/bench_LUV/c_run5')
# # Also adds c=0.01, but used explicitly NO parameterization for c=0.01.
# #Base.load('data/remote/AdInf/bench_LUV/c_run7') 
# Groups['EnKF   infl:? detp:1'] = Base.split('^EnKF ')
# Groups['EnKF_N infl:? detp:1'] = Base.split(nonBase_EnKF_N)
# 
# # Full run. Uses dt=0.005 also for truncated model (i.e. DA).
# #Base = ResultsTable('data/remote/AdInf/bench_LUV/c_run9')  # 8 repetitions
# #Base = ResultsTable('data/remote/AdInf/bench_LUV/c_run8') # only 1 repetition, with truth at c=50 in a tiny limit cycle
# 
# Singls['Base'] = Base


##############################
# LorenzUV Lorenz
##############################

##############################
# AxProps['xlabel'] = 'Forcing (F)'
# 
# #Base = ResultsTable('data/remote/AdInf/bench_LUV/F_run[1-4]')
# Base = ResultsTable('data/remote/AdInf/bench_LUV/F_run[5-8]')
# Singls['FULL']  = Base.split('N:80 nu:2 FULL')
# Singls['CHEAT'] = Base.split('N:40 nu:2 CHEAT')
# Others = ResultsTable('data/remote/AdInf/bench_LUV/F_run17')
# N15 = Others.split('N:15')
# #N20 = Others.split('N:20')
# 
# Groups['EnKF   infl:? detp:1'] = Others.split('^EnKF ' )
# Groups['EnKF_N infl:? detp:1'] = Others.split('^EnKF_N')
# _  = Groups['EnKF   infl:? detp:1'].split('detp:0') # Groups['EnKF   infl:? detp:0']
# _  = Groups['EnKF_N infl:? detp:1'].split('detp:0') # Groups['EnKF_N infl:? detp:0']
#  
# # Singls['M11']       = Others.split('M11')
# # Groups['M11 v_b:?'] = deepcopy(Singls['M11'])
# # Singls['M11'].rm('v_b:(0.001|0.37)',INV=1)
# #  
# # Singls['A07']       = Others.split('A07')
# # Groups['A07 v_b:?'] = deepcopy(Singls['A07'])
# # Singls['A07'].rm('v_b:(0.07|0.87)',INV=1)
# 
# #Singls['Adapt'] = ResultsTable('data/remote/AdInf/bench_LUV/F_run19')
# #Singls['Adapt'].rm('N:15')
# 
# Singls['M11'] = ResultsTable('data/remote/AdInf/bench_LUV/F_run20')
# Singls['M11'].rm('N:20',INV=True)
# Singls['M11'].rm('var~o:1',INV=False)
# 
# AxProps['xlim'] = (3,40)
# AxProps['ylim'] = (0,0.8)

##############################
# AxProps['xlabel'] = 'Forcing (F)'
# R = ResultsTable('data/remote/AdInf/bench_LUV/F_run25')
#  
# Singls['FULL'] = R.split('FULL').split('N:80.*nu:1')
# Singls['Base'] = R.rm('(Clim|Var)')
# 
# Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
# 
# R.rm('(M11|A07).*infl:1 ')

##############################
# AxProps['xlabel'] = 'Coupling strength (h)'
# 
# # Base = ResultsTable('data/remote/AdInf/bench_LUV/h_run[5-8]')
# # Singls['Base'] = Base
# # 
# # Groups['EnKF   infl:? detp:1'] = Base.split('^EnKF ')
# # Groups['EnKF_N infl:? detp:1'] = Base.split(nonBase_EnKF_N)
# # Groups['EnKF   infl:? detp:0'] = Groups['EnKF   infl:? detp:1'].split('detp:0')
# # Groups['EnKF_N infl:? detp:0'] = Groups['EnKF_N infl:? detp:1'].split('detp:0')
# # 
# # # Singls['M11'] = ResultsTable('data/remote/AdInf/bench_LUV/h_run11')
# # # Singls['A07'] = Singls['M11'].split('A07')
# # # Groups['M11 v_b:?'] = deepcopy(Singls['M11'])
# # # Groups['A07 v_b:?'] = deepcopy(Singls['A07'])
# # # Singls['A07'].rm('v_b:0.(56|083)',INV=1)
# # # Singls['M11'].rm('v_b:0.00(2|48)',INV=1)
# # 
# # Singls['Adapt'] = ResultsTable('data/remote/AdInf/bench_LUV/h_run13')
# # Singls['Adapt'].rm('N:15')
# 
# R = ResultsTable('data/remote/AdInf/bench_LUV/h_run15')
# Singls['Base']  = R.rm('(Clim|Var)')
# Singls['FULL']  = R.split('FULL').split('(N:20|N:80.*nu:1)')
# Singls['CHEAT'] = R.split('CHEAT').split('N:40')
# R.rm('N:15')
# R.rm('detp:0')
# 
# Groups['EnKF        infl:?'] = R.split('^EnKF ' )
# Groups['EnKF_N      infl:?'] = R.rm('^EnKF_N ' )
# 
# AxProps['xlim'] = (0,6)
# AxProps['ylim'] = (-0.01,0.4)
#OnOff = array([1,0,1,0,1,1,0,0,0,1,0,1,0,1,0,0]).astype(bool)

##############################
# AxProps['xlabel'] = 'Speed ratio (c)'
# 
# # 4x2 repetition, T=200, 40 data points.
# Base = ResultsTable('data/remote/AdInf/bench_LUV/c_run1[2-5]')
# Base.rm(r'EnKF_N (?!.*N:40 nu:2).*FULL') # rm EnKF_N FULL except (...)
# Base.rm([1,2,6]) # rm inds 1,2,4
# Singls['Base'] = Base
# 
# # TODO Replace source of labels below (all the way to AxProps) by c_run30
# Groups['EnKF   infl:? detp:1'] = Base.split('^EnKF ')
# Groups['EnKF_N infl:? detp:1'] = Base.split(nonBase_EnKF_N)
# 
# # Adds detpO0
# Groups['EnKF   infl:? detp:0'] = ResultsTable('data/remote/AdInf/bench_LUV/c_run1[6-9]')
# Groups['EnKF_N infl:? detp:0'] = Groups['EnKF   infl:? detp:0'].split('EnKF_N')
# # Adds detpO 2,4 -- quite unstable
# #Groups['EnKF   infl:? detp:2'] = ResultsTable('data/remote/AdInf/bench_LUV/c_run2[4-7]')
# #Groups['EnKF   infl:? detp:4'] = ResultsTable('data/remote/AdInf/bench_LUV/c_run2[0-3]')
# 
# #Singls['M11'] = ResultsTable('data/remote/AdInf/bench_LUV/c_run29')
# #Singls['A07'] = Singls['M11'].split('A07')
# #Groups['M11 v_b:?'] = deepcopy(Singls['M11'])
# #Groups['A07 v_b:?'] = deepcopy(Singls['A07'])
# #Singls['A07'].rm('v_b:(1|0.14)',INV=1)
# #Singls['M11'].rm('v_b:0.0(02|4)',INV=1)
# 
# Singls['Adapt'] = ResultsTable('data/remote/AdInf/bench_LUV/c_run31')
# Singls['Adapt'].rm('N:15')
# 
# AxProps['xlim'] = (0,80)
# AxProps['ylim'] = (-0.01,0.4)


##############################
# AxProps['xlabel'] = 'Obs var (R)'
# 
# Base = ResultsTable('data/remote/AdInf/bench_LUV/R_run1')
# 
# Singls['FULL']  = Base.split('FULL')
# Singls['CHEAT'] = Base.split('CHEAT')
# Singls['CHEAT'].rm('N:40 nu:2',INV=1)
# Singls['FULL' ].rm('(N:20 nu:2|N:40 nu:1)',INV=1)
# 
# Others, Singls['Base'] = Base.split2('(EnKF|ETKF|EAKF)')
# N15 = Others.split('N:15')
# 
# Groups['EnKF   infl:? detp:1'] = Others.split('^EnKF ' )
# Groups['EnKF_N infl:? detp:1'] = Others.split('^EnKF_N')
# _  = Groups['EnKF   infl:? detp:1'].split('detp:0') # Groups['EnKF   infl:? detp:0']
# _  = Groups['EnKF_N infl:? detp:1'].split('detp:0') # Groups['EnKF_N infl:? detp:0']
#  
# Singls['M11']       = Others.split('M11')
# Groups['M11 v_b:?'] = deepcopy(Singls['M11'])
# Singls['M11'].rm([0],INV=1)
#  
# Singls['A07']       = Others.split('A07')
# Groups['A07 v_b:?'] = deepcopy(Singls['A07'])
# Singls['A07'].rm('v_b:0.15',INV=1)
# 
# 
# #x0    = Singls['FULL'].split('N:40 nu:1').mean_field('rmse_a')[0][0]
# #x1    = optimz(Groups['EnKF   infl:? detp:1'],'rmse_a')
# #Skill = lambda x: (x-x0)/(x1-x0)
# 
# AxProps['xlim'] = (0.01,40)
# AxProps['ylim'] = (0.01,2)
# AxProps['xscale'] = 'log'
# #AxProps['yscale'] = 'log'


##############################
# Lorenz63
##############################

##############################
AxProps['xlabel'] = 'Magnitude (var/Q) of noise on truth'
R = ResultsTable('data/remote/AdInf/bench_L63/Q_run2')

Singls['FULL'] = R.split('FULL').split('N:80.*nu:1')
Singls['Base'] = R.rm('(Clim|Var)')

Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )

R.rm('(M11|A07).*infl:1 ')

AxProps['xscale'] = 'log'
OnOff = array([1,1,0,0,1,1,0,1,0,1,1,0,1,0]).astype(bool)

##############################
# AxProps['xlabel'] = 'Prandtl (sigma)'
# R = ResultsTable('data/remote/AdInf/bench_L63/FDA_run2')
#  
# Singls['FULL'] = R.split('FULL').split('N:80.*nu:1')
# Singls['Base'] = R.rm('(Clim|Var)')
# 
# Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
# 
# R.rm('(M11|A07).*infl:1 ')


##############################
# Plot
##############################

# for k,v in Groups.items():
#   print_c(k)
#   print(v)

# RT = Singls['A07']
# for iC,(row,name) in enumerate(zip(RT.mean_field('rmse_a')[0],RT.labels)): 
#   ax.plot(RT.xticks,Skill(row),**style(name))
# check = toggle_lines()

#G = Groups['EnKF   infl:? detp:1']
#infls = xtract_prop(G.labels,'infl')[np.nanargmin(G.mean_field('rmse_a')[0],0)]

# To scale xaxis so that points are equi-distant,
# first remove xticks from plotting calls,
# then apply the following:
#xax = ax.get_xaxis()
#ticks = [int(x) for x in np.linspace(0,len(xticks)-1,8)]
#xax.set_ticks(ticks)
#xax.set_ticklabels('{:.3g}'.format(x) for x in xticks[ticks])

fig, ax = plt.subplots()

for RT in Singls.values():
  if RT and len(RT.labels):
    for iC,(row,name) in enumerate(zip(RT.mean_field('rmse_a')[0],RT.labels)): 
      ax.plot(RT.xticks,Skill(row),**style(name))

for lbl, RT in Groups.items():
  if RT:
    ax.plot(RT.xticks,Skill(optimz(RT,'rmse_a')),**style(lbl))

R2 = deepcopy(R)
#R2.rm(arange(10,17+1),INV=True)
#R2.rm('_N_')
for iC,(row,name) in enumerate(zip(R2.mean_field('rmse_a')[0],R2.labels)): 
  ax.plot(R2.xticks,Skill(row),**style(name))

for k,v in AxProps.items(): ax.set(**{k:v})
check = toggle_lines(state=OnOff)

##



