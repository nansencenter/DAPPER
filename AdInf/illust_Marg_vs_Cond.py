# Illustrate the diff between "inspecting" along a condition (i.e. conditioning)
# and marginalizing out the other DoFs.

from common import *
from scipy.stats import norm
from matplotlib.patches import Ellipse
import matplotlib.patches as patches


h = {}

fig = plt.figure(1,figsize=(12,5))
fig.clear()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.axis('off')
ax1.set_xlim(-2,17)
ax1.set_ylim(-2,17)

h['aBx'] = ax1.arrow(-1,0 ,12,0 ,head_width=0.5,color='k')
h['aBy'] = ax1.arrow(0 ,-1,0 ,12,head_width=0.5,color='k')
h['aBs'] = ax1.arrow(-1,-1,14,14,head_width=0.5,color='k')


h['tBx'] = ax1.text(10.5,-2  ,r'$B_1$'          ,size=24)
h['tBy'] = ax1.text(-2.5,10.5,r'$B_2$'          ,size=24)
h['ts']  = ax1.text(14  ,14  ,r'$s$'            ,size=24)
h['tsB'] = ax1.text(11  ,10  ,r'$B_1 = B_2 = s$',size=15)
h['tc']  = ax1.text(1   ,7   ,r'$c$ (const.)'   ,size=15)

h['rec'] = ax1.add_patch(patches.Rectangle( (0, 0), 10, 10,
  alpha=0.2,))

h['e'] = []
ells = [Ellipse(xy=[3,3], width=(2+i)/5, height=2+i, angle=45) for i in range(10)]
for e in ells:
    h['e'].append(ax1.add_artist(e))
    e.set_facecolor([0,1,0,0.15])
    e.set_edgecolor([0,0,0,0.4])
    e.set_linewidth(2)



ax2.axis('off')
ax2.set_xlim(-2,17)
ax2.set_ylim(-2,17)

h['aBs2'] = ax2.arrow(-1,0,17,0,head_width=0.5,color='k')
h['ap']   = ax2.arrow(0,-1,0,14,head_width=0.5,color='k')

h['ts2'] = ax2.text(15.5,-1.5,r'$s$',size=24)

h['psC'] = [ax2.plot([0,14],[7,7],lw=2,c='b')]
h['psC'].append(ax2.text(0-1,7-0.4,"$c$",size=15))
h['psC'].append(ax2.text(11,7.4,"$p(s|B_1=B_2)$",size=15,color='b'))

h['psM'] = [ax2.plot([0,7,14],[0,13,0],lw=2,c='r')]
h['psM'].append(ax2.text(8,12,"$p(s)$",size=15,color='r'))

xx = np.linspace(0,14,301)
h['e'].append(ax2.plot(xx,30*norm(loc=5,scale=1).pdf(xx),'g',lw=3,alpha=0.6)[0])


# Hide the right and top spines
#ax1.spines['right'].set_visible(False)
#ax1.spines['top'].set_visible(False)
## Only show ticks on the left and bottom spines
#ax1.yaxis.set_ticks([])
#ax1.xaxis.set_ticks([])


save_num = 0
def savef():
  global save_num
  plt.savefig('data/figs/AdInf/Marg_vs_Cond_' + str(save_num) + '.pdf')
  save_num += 1
  plt.pause(0.1)

def tog(h,prompt=True,save=True):
  toggle_viz(h,prompt=True)
  if save:
    savef()

for key in h:
  toggle_viz(h[key],prompt=False)


tog([h['aBx'], h['aBy'], h['tBx'], h['tBy']])
tog([h['rec'], h['tc']])
tog([h['aBs'], h['ts'], h['tsB']])
tog([h['aBs2'], h['ap'], h['ts2']])
tog(h['psC'])
tog(h['psM'])
tog(h['e'])
