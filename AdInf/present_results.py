from common import *

def xtract_prop(list_of_strings,propID,cast=float,fillval=np.nan):
  "E.g. xtract_prop(R1.labels,'infl',float)"
  props = []
  for s in list_of_strings:
    x = re.search(r'.*'+propID+':(.+?)(\s|$)',s)
    props += [cast(x.group(1))] if x else [fillval]
  return array(props)

def plot_best(ax,RT,field,color,label,**ls):
  mu = RT.mean_field(field)[0]
  best = np.nanmin(mu,0)
  ax.plot(abscissa,best,label=label,color=color,**{**linestyle, **ls})



##############################
# LorenzXY Wilks: vs speed-ratio (c)
##############################

# # Full run
# Base = ResultsTable('data/remote/AdInf/bench_LXY/c_run4')
# # Adds coordinate c=0.01. Unfortunately, uses slightly different infl values.
# Base.load('data/remote/AdInf/bench_LXY/c_run5')
# # Also adds c=0.01, but used explicitly NO parameterization for c=0.01.
# #Base.load('data/remote/AdInf/bench_LXY/c_run7') 
# Ro1, Base = Base.split('^EnKF ') # Ro1: EnKF
# cond = lambda s: s.startswith('EnKF_N') and not re.search('(FULL|CHEAT)',s)
# Ro1_N, Base = Base.split(cond) # Ro1_N: EnKF_N
# 
# # Full run. Uses dt=0.005 also for truncated model (i.e. DA).
# #Base = ResultsTable('data/remote/AdInf/bench_LXY/c_run9')  # 8 repetitions
# #Base = ResultsTable('data/remote/AdInf/bench_LXY/c_run8') # only 1 repetition, with truth at c=50 in a tiny limit cycle

##############################
# LorenzXY Lorenz: vs speed-ratio (c)
##############################

# # 4x2 repetition, T=200, 40 data points.
# Base = ResultsTable('data/remote/AdInf/bench_LXY/c_run1[2-5]')
# Base.rm(r'EnKF_N (?!.*N:40 nu:2).*FULL') # rm EnKF_N FULL except (...)
# Base.rm([1,2,6]) # rm inds 1,2,4
# Ro1, Base = Base.split('^EnKF ') # Ro1: EnKF
# cond = lambda s: s.startswith('EnKF_N') and not re.search('(FULL|CHEAT)',s)
# Ro1_N, Base = Base.split(cond) # Ro1_N: EnKF_N
# 
# # Adds detpO0
# Ro0 = ResultsTable('data/remote/AdInf/bench_LXY/c_run1[6-9]')
# Ro0_N, Ro0 = Ro0.split('EnKF_N')
# # # Adds detpO2 -- quite unstable
# # Ro2 = ResultsTable('data/remote/AdInf/bench_LXY/c_run2[4-7]')
# # # Adds detpO4 -- more unstable
# # Ro4 = ResultsTable('data/remote/AdInf/bench_LXY/c_run2[0-3]')

Base = ResultsTable('data/remote/AdInf/bench_LXY/c_run(2[89]|3[01])')


##############################
# Plot
##############################

abscissa = Base.abscissa
linestyle = {'ls':'-','marker':'o','ms':3}
fig, ax = plt.subplots()

for iC,(row,name) in enumerate(zip(Base.mean_field('rmse_a')[0],Base.labels)): 
  ax.plot(abscissa,row,label=name,**linestyle)

#infls = xtract_prop(Ro1.labels,'infl')[np.nanargmin(Ro1.mean_field('rmse_a')[0],0)]

plot_best(ax,Ro1  ,'rmse_a',blend_rgb('k',0.9),'best(infl) EnKF   detp=1')
plot_best(ax,Ro1_N,'rmse_a',blend_rgb('k',0.5),'best(infl) EnKF_N detp=1')

# plot_best(ax,Ro0  ,'rmse_a',blend_rgb('k',0.9),'best(infl) EnKF   detp=0',ls='--')
# plot_best(ax,Ro0_N,'rmse_a',blend_rgb('k',0.5),'best(infl) EnKF_N detp=0',ls='--')
# 
# plot_best(ax,Ro2  ,'rmse_a',blend_rgb('k',0.9),'best(infl) EnKF   detp=2',ls=':')
# plot_best(ax,Ro4  ,'rmse_a',blend_rgb('k',0.9),'best(infl) EnKF   detp=4',ls='-.')

#ax.set_xscale('log');
#ax.set_ylim(bottom=0.05)
ax.set_ylim(bottom=-0.01, top=0.3)
check = toggle_lines()

# To scale xaxis so that points are equi-distant,
# first remove abscissa from plotting calls,
# then apply the following:
#xax = ax.get_xaxis()
#ticks = [int(x) for x in np.linspace(0,len(abscissa)-1,8)]
#xax.set_ticks(ticks)
#xax.set_ticklabels('{:.3g}'.format(x) for x in abscissa[ticks])

# OBSOLETE: EXAMPLE PLOTTING with advanced coloring
# colors = ['ml' + c for c in 'oyvcr'] # Matlab
# for iC,(row,name) in enumerate(zip(mu,Base.labels)): 
#   m = re.search('tag(\d)(\d)',name) # tag hunting
#   if m:
#     g1 = int(m.group(1))
#     g2 = int(m.group(2))
#     if   g1==5: c = blend_rgb('ml'+'b',1-g2/10)
#     elif g1==7: c = blend_rgb('ml'+'g',1-g2/10)
#   else:         c = colors[iC%len(colors)]
#   plt.plot(abscissa,row,'-o',color=c,label=name)

