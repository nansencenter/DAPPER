# To run script, use -i option: %run -i [this_script)
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

# The variable 'check' must be left in interactive namespace
# in order for the checkmarks to remain live after creating new figures.
if 'check' not in locals(): check = []

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

DFLT = {'ls':'-','marker':'o','ms':3,'lw':2}
STYLES = [
    ('Climatology'      , None        , OD(c=blend_rgb('k' ,1.0))),
    ('Var3D'            , '3D-Var'    , OD(c=blend_rgb('k' ,0.7))),
    ('(EnRML|PertObs)'  , None        , OD(c='C6')),
    ('iEnKS'            , None        , OD(c='C0')),
    ('EnKF'             , None        , OD(c='C1')),
    ('Lag:1'            , None        , OD(ls='--')),
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
AxProps['title'] = 'L95'
AxProps['xlabel'] = 'Time interval between observations ($\Delta t$)'

S = ResultsTable('data/remote/Stoch_iEnS/bench_L95/dkObs_run4')

NGroups = OD()
for N in [20, 25, 35, 50, 100]:
  NGroups[str(N)] = S.split('N:'+str(N))

R = NGroups['25']
Singls['R'] = R
Groups['PertObs'] = R.split('^EnKF .*PertObs')
Groups['EnRML 1'] = R.split('^EnRML .*Lag:1')
Groups['EnRML L'] = R.split('^EnRML .*Lag:0.4')

AxProps['ylim'] = (0.15,1.1)
AxProps['xticks'] = R.xticks
AxProps['xticklabels'] = 0.05*R.xticks

#OnOff = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

##############################
# Plot
##############################
AxProps['ylabel'] = 'rmse_a'

fig, ax = plt.subplots()

for RT in Singls.values():
  if RT and len(RT.labels):
    for iC,(row,name) in enumerate(zip(RT.mean_field('rmse_a')[0],RT.labels)): 
      ax.plot(RT.xticks,Skill(row),**style(name))

for lbl, RT in Groups.items():
  if RT:
    ax.plot(RT.xticks,Skill(optimz(RT,'rmse_a')),**style(lbl))

G = Groups['PertObs']
infls = xtract_prop(G.labels,'infl')[np.nanargmin(G.mean_field('rmse_a')[0],0)]

for k,v in AxProps.items(): ax.set(**{k:v})
check += [toggle_lines(state=OnOff,txtwidth=22,txtsize=10)]

##

