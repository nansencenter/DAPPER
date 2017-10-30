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
Singls  = OD()        # each label is here will be shown 
Groups  = OD()        # for each group: only min (per abscissa) is shown
AxProps = OD()        # labels, etc


DFLT = {'ls':'-','marker':'o','ms':3}
STYLES = [
    ('EnKF.*FULL'       , None        , OD(c=blend_rgb('C1',1.0))),
    ('EnKF.*CHEAT'      , None        , OD(c=blend_rgb('g' ,0.5))),
    ('Climatology'      , None        , OD(c=blend_rgb('k' ,1.0))),
    ('Var3D'            , '3D-Var'    , OD(c=blend_rgb('k' ,0.7))),
    ('EnKF .* infl:\?'  , None        , OD(c=blend_rgb('b' ,0.9))),
    ('EnKF_N .*infl:\?' , None        , OD(c=blend_rgb('b' ,0.5))),
    ('ETKF_M11_v1'      , 'M11'       , OD(c=blend_rgb('C4',0.5))),
    ('EAKF_A07_v1'      , 'A07'       , OD(c=blend_rgb('C9',0.5))),
    ('M11.*v_b:\?'      , None        , OD(c=blend_rgb('C4',0.9))),
    ('A07.*v_b:\?'      , None        , OD(c=blend_rgb('C9',0.9))),
    (r'(\w+?):\?'       , r'\1:OPT'   , OD()), # replace 'xxx:?' by 'xxx:OPT'
    ('detp:0'           , None        , OD(ls='--')),
    ('detp:2'           , None        , OD(ls=':')),
    ]

def style(label):
  "Apply matching styles in their order"
  D = DFLT.copy()
  D['label'] = label
  matches = [i for i,row in enumerate(STYLES) if re.search(row[0],label)]
  for i in matches:
    row = STYLES[i]
    D.update(row[2])
    if i and row[1]:
      D['label'] = re.sub(row[0],row[1],D['label'])
  return D


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
# Singls['M11']       = Others.split('M11')
# Groups['M11 v_b:?'] = deepcopy(Singls['M11'])
# Singls['M11'].rm('v_b:(0.001|0.37)',INV=1)
#  
# Singls['A07']       = Others.split('A07')
# Groups['A07 v_b:?'] = deepcopy(Singls['A07'])
# Singls['A07'].rm('v_b:(0.07|0.87)',INV=1)
# 
# AxProps['xlim'] = (3,40)
# AxProps['ylim'] = (0,0.7)

##############################
# AxProps['xlabel'] = 'Coupling strength (h)'
# 
# Base = ResultsTable('data/remote/AdInf/bench_LUV/h_run[5-8]')
# Singls['Base'] = Base
# 
# # TODO Replace source of labels below (all the way to AxProps) by h_run12
# Groups['EnKF   infl:? detp:1'] = Base.split('^EnKF ')
# Groups['EnKF_N infl:? detp:1'] = Base.split(nonBase_EnKF_N)
# Groups['EnKF   infl:? detp:0'] = Groups['EnKF   infl:? detp:1'].split('detp:0')
# Groups['EnKF_N infl:? detp:0'] = Groups['EnKF_N infl:? detp:1'].split('detp:0')
# 
# Singls['M11'] = ResultsTable('data/remote/AdInf/bench_LUV/h_run11')
# Singls['A07'] = Singls['M11'].split('A07')
# Groups['M11 v_b:?'] = deepcopy(Singls['M11'])
# Groups['A07 v_b:?'] = deepcopy(Singls['A07'])
# Singls['A07'].rm('v_b:0.(56|083)',INV=1)
# Singls['M11'].rm('v_b:0.00(2|48)',INV=1)
# 
# AxProps['xlim'] = (0,6)
# AxProps['ylim'] = (-0.01,0.4)

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
# Singls['M11'] = ResultsTable('data/remote/AdInf/bench_LUV/c_run29')
# Singls['A07'] = Singls['M11'].split('A07')
# Groups['M11 v_b:?'] = deepcopy(Singls['M11'])
# Groups['A07 v_b:?'] = deepcopy(Singls['A07'])
# Singls['A07'].rm('v_b:(1|0.14)',INV=1)
# Singls['M11'].rm('v_b:0.0(02|4)',INV=1)
# 
# AxProps['xlim'] = (0,80)
# AxProps['ylim'] = (-0.01,0.5)


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
# Plot
##############################

# for k,v in Groups.items():
#   print_c(k)
#   print(v)

fig, ax = plt.subplots()

# RT = Singls['A07']
# for iC,(row,name) in enumerate(zip(RT.mean_field('rmse_a')[0],RT.labels)): 
#   ax.plot(RT.abscissa,Skill(row),**style(name))
# check = toggle_lines()

for RT in Singls.values():
  if len(RT.labels):
    for iC,(row,name) in enumerate(zip(RT.mean_field('rmse_a')[0],RT.labels)): 
      ax.plot(RT.abscissa,Skill(row),**style(name))

for lbl, RT in Groups.items():
  ax.plot(RT.abscissa,Skill(optimz(RT,'rmse_a')),**style(lbl))

#G = Groups['EnKF   infl:? detp:1']
#infls = xtract_prop(G.labels,'infl')[np.nanargmin(G.mean_field('rmse_a')[0],0)]

#ax.set_xscale('log');
ax.set(**AxProps)
check = toggle_lines()

# To scale xaxis so that points are equi-distant,
# first remove abscissa from plotting calls,
# then apply the following:
#xax = ax.get_xaxis()
#ticks = [int(x) for x in np.linspace(0,len(abscissa)-1,8)]
#xax.set_ticks(ticks)
#xax.set_ticklabels('{:.3g}'.format(x) for x in abscissa[ticks])




