# To run script:
# >>> check = []
# >>> %run -i AdInf/present_results_NEW.py
#
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

def cselect(RT,cond,field):
  RT = deepcopy(RT).split(cond)
  return RT.mean_field(field)[0][0]

######

# Score transformation.
# Overwrite (below) to use
Skill = lambda x: x 

OD      = OrderedDict # short alias
Singls  = OD()        # each label is here will be plotted 
Groups  = OD()        # for each group: only min (per xticks) is plotted
AxProps = OD()        # labels, etc
OnOff   = None

##

DFLT = {'ls':'-','marker':'o','ms':3}
STYLES = [
    ('EnKF_N'           , None        , OD(c=blend_rgb('C4',1.0))),
    ('EnKF.*FULL'       , None        , OD(c=blend_rgb('k' ,1.0))),
    ('EnKF.*CHEAT'      , None        , OD(c=blend_rgb('k' ,0.7))),
    ('Climatology'      , None        , OD(c=blend_rgb('k' ,0.5))),
    ('Var3D'            , '3D-Var'    , OD(c=blend_rgb('k' ,0.5))),
    ('EnKF .* infl:\?'  , None        , OD(c=blend_rgb('b' ,1.0))),
    ('EnKF_N .*infl:\?' , None        , OD(c=blend_rgb('b' ,1.0))),
    ('EnKF_N_mod'       , None        , OD(c=blend_rgb('C6',1.0))),
    ('ETKF_M11'         , 'M11'       , OD(c=blend_rgb('C1',1.0))),
    ('M11(?!.*var~o)'   , None        , OD(c=blend_rgb('C1',0.7),ls='--')),
    ('EAKF_A07'         , 'A07'       , OD(c=blend_rgb('C2',1.0))),
    ('Xplct'            , None        , OD(c=blend_rgb('C3',1.0))),
    ('Xplct.*nu_f'      , None        , OD(c=blend_rgb('C3',0.7),ls='--')),
    ('Xplct.*L:'        , None        , OD(c=blend_rgb('C3',0.7),ls=':')),
    ('InvCS'            , None        , OD(c=blend_rgb('C4',1.0))),
    ('N_Xplct'          , None        , OD(c=blend_rgb('C5',1.0))),
    ('N_Xplct.*Cond:0'  , None        , OD(c=blend_rgb('C9',1.0),ls='-')),
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
# LorenzUV
##############################

##############################
# AxProps['title'] = 'LUV N=20 dtObs=0.05'
# AxProps['xlabel'] = 'Forcing (F) [both truth and DA]'
# 
# R = ResultsTable('data/remote/AdInf/bench_LUV/F_run26') # All, but old adaptive
# Singls['CHEAT'] = R.split('CHEAT').split('N:80.*nu:1')
# Singls['FULL'] = R.split('FULL').split('N:80.*nu:1')
# Singls['Base'] = R.rm('(Clim|Var)')
# Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
# 
# R = ResultsTable('data/remote/AdInf/bench_LUV/F_runX')
# Singls['R'] = R
# 
# AxProps['xlim'] = (3,30)
# AxProps['ylim'] = (0,0.8)


##############################
# AxProps['title']  = 'LUV N=20 dtObs=0.15 N=20'
# AxProps['xlabel'] = 'Forcing (F) [both truth and DA]'
# 
# R = ResultsTable('data/remote/AdInf/bench_LUV/F_run4[5-9]') # Full with new adapt
# Singls['CHEAT'] = R.split('CHEAT').split('N:80')
# Singls['FULL']  = R.split('FULL')
# Singls['Base']  = R.split('(Clim|Var(?!.*detp:0))')
# Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
# R = ResultsTable('data/remote/AdInf/bench_LUV/F_run57') # V3 adapt, 64 rep, from munin
# R.load          ('data/remote/AdInf/bench_LUV/F_run60') # add (_X nu_f=1e4) (_N_X nu_f=1e3) (_N_mod)
# R.load          ('data/remote/AdInf/bench_LUV/F_run82') # add _N, Cond=False
# Singls['R'] = R
# Groups['EnKF_pre infl:?'].load('data/remote/AdInf/bench_LUV/F_run62')
# 
# AxProps['xlim'] = (3,30)
# AxProps['ylim'] = (0,1.2)
# 
# OnOff = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]

##############################
# AxProps['title']  = 'LUV dtObs=0.15 N=20'
# AxProps['xlabel'] = 'Speed ratio (c)'
# 
# R = ResultsTable('data/remote/AdInf/bench_LUV/c_run41') # All, V3 adapt, 32 rep
# Singls['CHEAT'] = R.split('CHEAT').split('nu:1')
# Singls['FULL']  = R.split('FULL').split('nu:1')
# Singls['Base']  = R.split('(Clim|Var)')
# Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
# R.mv(r'detp:1 ',r'')
# R.load('data/remote/AdInf/bench_LUV/c_run44') # add c=14,15 (adaptive only)
# R.load          ('data/remote/AdInf/bench_LUV/c_run46') # add (_X nu_f=1e4) (_N_X nu_f=1e3) (_N_mod) (_N)
# R.mv(r'^(\w+)(?!.*N:20)',r'\1 N:20')
# Singls['R'] = R
# Singls['R2'] = ResultsTable('data/remote/AdInf/bench_LUV/c_run49') # add Cond=0
# Singls['R2'].mv('(.*)',r'\1 Cond:0')
# 
# AxProps['xlim'] = (0.9,22)
# AxProps['ylim'] = (0.1,0.45)
# AxProps['xscale'] = 'log'
# 
# OnOff = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

# ##############################
# AxProps['title'] = 'LUV'
# AxProps['xlabel'] = 'Coupling strength (h)'
# 
# R = ResultsTable('data/remote/AdInf/bench_LUV/h_run1[6-9]')
# Singls['R'] = R
# 
# OnOff = [1, 0, 1, 0, 0, 1, 0, 0, 0]

# ########
# AxProps['title']  = 'LUV N=20 wilks05 dtObs=0.05'
# R = ResultsTable('data/remote/AdInf/bench_wilks05/F_run2') # dtObs=0.05
# Singls['R'] = R
# AxProps['xlabel'] = 'Forcing (F) [both truth and DA]'
# ########
# AxProps['title']  = 'LUV N=20 wilks05 dtObs=0.15'
# R = ResultsTable('data/remote/AdInf/bench_wilks05/F_run3')
# Singls['R'] = R
# AxProps['xlabel'] = 'Forcing (F) [both truth and DA]'
# ########
# AxProps['title']  = 'LUV N=8 wilks05 dtObs=0.15'
# R = ResultsTable('data/remote/AdInf/bench_wilks05/F_run8')
# Singls['R'] = R
# AxProps['xlabel'] = 'Forcing (F) [both truth and DA]'
# ########
# AxProps['title']  = 'LUV N=8 wilks05 dtObs=0.05'
# R = ResultsTable('data/remote/AdInf/bench_wilks05/F_run9')
# Singls['R'] = R
# AxProps['xlabel'] = 'Forcing (F) [both truth and DA]'





##############################
# Lorenz95
##############################

##############################
# AxProps['title'] = 'L95 dtObs=0.15 N=20'
# AxProps['xlabel'] = 'Magnitude (Q) of noise on truth'
# 
# R = ResultsTable('data/remote/AdInf/bench_L95/Q_run2[01]') # All, V3 adapt, 2x12 rep
# R.load          ('data/remote/AdInf/bench_L95/Q_run22') # add (_N_X nu_f=1e3) (_X nu_f=1e4) (_N_mod)
# Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )          
# Singls['R'] = R
# 
# AxProps['xscale'] = 'log'
# AxProps['ylim'] = (0.3,0.7)
# 
# OnOff = [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]

##############################
# AxProps['title'] = 'L95 N=20 dtObs=0.15'
# AxProps['xlabel'] = 'Forcing (F) truth'
# 
# R = ResultsTable('data/remote/AdInf/bench_L95/FTr_run1[45]') # Old?
# Singls['FULL'] = R.split('FULL')
# Singls['Base'] = R.split('(Clim|Var)')
# Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
# R = ResultsTable('data/remote/AdInf/bench_L95/FTr_run19') # V3 adapt, 64 rep
# R.load          ('data/remote/AdInf/bench_L95/FTr_run24') # add _N_X nu_f=1e3
# R.load          ('data/remote/AdInf/bench_L95/FTr_run26') # add _X nu_f=1e4
# R.load          ('data/remote/AdInf/bench_L95/FTr_run27') # add _N_mod
# Singls['R'] = R
# Singls['R2'] = ResultsTable('data/remote/AdInf/bench_L95/FTr_run30') # add Cond=0
# Singls['R2'].mv('(.*)',r'\1 Cond:0')
# 
# AxProps['ylim'] = (0.3,0.7)
# OnOff = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]

##############################
# Lorenz63
##############################

##############################
N = 3
AxProps['title'] = 'L63 N={:d}'.format(N)
AxProps['xlabel'] = 'Magnitude (Q) of noise on truth'

R = ResultsTable('data/remote/AdInf/bench_L63/Q_run4')
Singls['FULL'] = R.split('FULL').split('nu:1')
Singls['Base'] = R.split('(Clim|Var)')

if N==5:
  Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
  R = ResultsTable('data/remote/AdInf/bench_L63/Q_run20')
  R = ResultsTable('data/remote/AdInf/bench_L63/Q_run35') # V3 adapt, 64 rep
  R.load          ('data/remote/AdInf/bench_L63/Q_run38') # add (_X nu_f=1e4) (_N_X nu_f=1e3)
  R.load          ('data/remote/AdInf/bench_L63/Q_run40') # add (_N_mod)
  OnOff = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1]
elif N==3:
  #R = ResultsTable('data/remote/AdInf/bench_L63/Q_run21')
  R = ResultsTable('data/remote/AdInf/bench_L63/Q_run33') # Full, 32 repeat
  Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
  Groups['EnKF_pre infl:?'].load('data/remote/AdInf/bench_L63/Q_run4[5-9]') # 5x6 repeats
  Singls['FULL N3'] = R.split('FULL').split('N:3')
  R = ResultsTable('data/remote/AdInf/bench_L63/Q_run34') # V3 adapt, 64 rep
  R.load          ('data/remote/AdInf/bench_L63/Q_run37') # add (_X nu_f=1e4) (_N_X nu_f=1e3)
  R.load          ('data/remote/AdInf/bench_L63/Q_run39') # add (_N_mod)
  Singls['R2'] = ResultsTable('data/remote/AdInf/bench_L63/Q_run42') # add Cond=0
  Singls['R2'].mv('(.*)',r'\1 Cond:0')
  OnOff = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
Singls['R'] = R

AxProps['xscale'] = 'log'
AxProps['ylim'] = (0.35,0.8)


##############################
# N = 5
# AxProps['title'] = 'L63 N={:d}'.format(N)
# AxProps['xlabel'] = 'Prandtl (sigma) Truth'
# 
# R = ResultsTable('data/remote/AdInf/bench_L63/FTr_run5') # Full, old adapt
# Singls['FULL'] = R.split('FULL').split('nu:1')
# Singls['Base'] = R.split('(Clim|Var)')
# 
# if N==5:
#   Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
#   R = ResultsTable('data/remote/AdInf/bench_L63/FTr_run7') # Adaptive
#   R = ResultsTable('data/remote/AdInf/bench_L63/FTr_run20') # V3 adapt, 64 rep
#   R.load          ('data/remote/AdInf/bench_L63/FTr_run21') # add (_X nu_f=1e4) (_N_X nu_f=1e3)
#   R.load          ('data/remote/AdInf/bench_L63/FTr_run24') # add (_N_mod)
#   OnOff = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1]
# elif N==3:
#   R = ResultsTable('data/remote/AdInf/bench_L63/FTr_run6')
#   Groups['EnKF_pre infl:?'] = R.split('^EnKF_pre ' )
#   R = ResultsTable('data/remote/AdInf/bench_L63/FTr_run19') # V3 adapt, 64 rep
#   R.load          ('data/remote/AdInf/bench_L63/FTr_run22') # add (_X nu_f=1e4) (_N_X nu_f=1e3)
#   R.load          ('data/remote/AdInf/bench_L63/FTr_run23') # add (_N_mod)
#   OnOff = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1]
# Singls['R'] = R
# 
# AxProps['xlim'] = (6,18)
# AxProps['ylim'] = (0.35,1.4)


########

for RT in Singls.values():
  if RT and len(RT.labels):
    RT.mv(' liv~g:.','')
    RT.mv(' sto~u:.','')

##############################
# Plot
##############################

# for k,v in Groups.items():
#   print_c(k)
#   print(v)

#G = Groups['EnKF   infl:? detp:1']
#infls = xtract_prop(G.labels,'infl')[np.nanargmin(G.mean_field('rmse_a')[0],0)]

# To scale xaxis so that points are equi-distant,
# first remove xticks from plotting calls,
# then apply the following:
#xax = ax.get_xaxis()
#ticks = [int(x) for x in np.linspace(0,len(xticks)-1,8)]
#xax.set_ticks(ticks)
#xax.set_ticklabels('{:.3g}'.format(x) for x in xticks[ticks])

##

AxProps['ylabel'] = 'rmse_a'
# NB: Don't use optimz on rmv_a: it will just find the lowest rmv -- not the best!

#x0 = Singls['FULL'].split('N:80.*nu:1').mean_field('rmse_a')[0][0]
#x1 = optimz(Groups['EnKF_pre infl:?'],'rmse_a')
#Skill = lambda x: x-x1

fig, ax = plt.subplots()

for lbl, RT in Groups.items():
  if RT:
    ax.plot(RT.xticks,Skill(optimz(RT,'rmse_a')),**style(lbl))

for RT in Singls.values():
  if RT and len(RT.labels):
    for iC,(row,name) in enumerate(zip(RT.mean_field('rmse_a')[0],RT.labels)): 
      ax.plot(RT.xticks,Skill(row),**style(name))

for k,v in AxProps.items(): ax.set(**{k:v})
check += [toggle_lines(state=OnOff,txtwidth=22,txtsize=10)]

##
