# Illustrate analysis step of particle filter

##
from common import *
plt.style.use('AdInf/paper.mplstyle')
#plt.xkcd()
sd0 = seed(28)

def get_ax_geometry(ax):
  x0, x1 = ax.get_xlim(); dx = x1-x0
  y0, y1 = ax.get_ylim(); dy = y1-y0
  return x0, x1, y0, y1, dx, dy


def arrowed_spines(ax,pullback=True):
  "Draw arrows for spines"
  x0, x1, y0, y1, dx, dy = get_ax_geometry(ax)

  # For making a small cross at the intersection
  pbx = 0.01*dx if pullback else 0
  pby = 0.01*dy if pullback else 0

  # Draw spines
  args = {'color':'k','zorder':99}
  aX = ax.arrow(x0-pbx,y0,dx,0,head_width=0.02*dy,head_length=0.02*dx,width=0.005*dy,**args)
  aY = ax.arrow(x0,y0-pby,0,dy,head_width=0.02*dx,head_length=0.02*dy,width=0.004*dx,**args)

  # Enlarge axis
  ax.set_xlim(x0-0.02*dx,x1+0.02*dx)
  ax.set_ylim(y0-0.02*dy,y1+0.02*dy)

  # Remove default spines
  ax.spines['top'   ].set_visible(False)
  ax.spines['right' ].set_visible(False)
  ax.spines['left'  ].set_visible(False)
  ax.spines['bottom'].set_visible(False)

  return aX, aY

def delta_plot(ax,xx,hh=1.0,c='k',label=''):
  x0, x1, y0, y1, dx, dy = get_ax_geometry(ax)
  if isinstance(hh,float):
    hh = np.ones_like(xx)*hh
  hs = []
  for x,h in zip(xx,hh):
    HL = 0.015*dy
    hs += [ ax.arrow(x,0,0,0.30*dy*h-HL,color=c,head_width=0.015*dx,head_length=HL,alpha=0.5,width=0.7*dy) ]
    #ax.plot([x,x],[0,0.069*h*dy],color=c,alpha=0.8,lw=2)
  return hs


def NormPDF(xx,b=0,B=1):
  return 1/sqrt(2*pi*B)*exp(-(xx-b)**2/2/B)
  
##############################
# 
##############################

xx = linspace(-3,20,401)
N = 12

# Prior kernels
b1 = 10
b2 = 0
B1 = 4**2
B2 = 1**2
# Prior mixture weights
v1 = 0.4
v2 = 1-v1
# Likelihood
R  = 8**2
y  = 17
# Posterior kernels
P1 = 1/(1/B1 + 1/R)
P2 = 1/(1/B2 + 1/R)
p1 = P1/B1*b1 + P1/R*y
p2 = P2/B2*b2 + P2/R*y
# Posterior mixture weights
w1 = v1*NormPDF(y,b1,B1+R)
w2 = v2*NormPDF(y,b2,B2+R)
w1, w2 = w1/(w1+w2), w2/(w1+w2)

def prior(xx):
  return v1*NormPDF(xx,b1,B1) + v2*NormPDF(xx,b2,B2)
def lklhd(xx):
  return NormPDF(y,xx,R)
def postr(xx):
  return w1*NormPDF(xx,p1,P1) + w2*NormPDF(xx,p2,P2)

# Prior ensemble
E = zeros(N)
w = ones(N)/N
for n in range(N):
  u = rand()
  if u > 0.5:
    b = b1
    B = B1
  else:
    b = b2
    B = B2
  E[n] = b + sqrt(B)*randn()

## Analysis: weigting by likelihood
ww = NormPDF(y,E,R)
ww /= ww.sum()

# Resampling
idx,_ = resample(ww,'Systematic')
F     = E[idx]
G,_   = regularize(np.atleast_2d(0.7),np.atleast_2d(E).T,idx,no_uniq_jitter=True)
G     = G.ravel()


##############################
# Plotting 
##############################
fig = plt.figure(1)
fig.clear()
fig, ax = plt.subplots(num=1)

h = dict()
h['prior'] = ax.plot(xx,prior(xx),'b',lw=3.5,label='Prior')
h['lklhd'] = ax.plot(xx,lklhd(xx),'g',lw=3.5,label='Likelihood')
h['postr'] = ax.plot(xx,postr(xx),'r',lw=3.5,label='Posterior')

ax.set_ylim(bottom=0,top=0.13)
_, aY = arrowed_spines(ax)
aY.set_visible(False)
ax.legend(framealpha=0.0,loc='upper right')
ax.set_xlabel('$x$',fontsize=24)
ax.set_xticks([])
ax.set_yticks([])

h['E'] = delta_plot(ax,E        ,c='b',label='Prior particles')
h['A'] = delta_plot(ax,E,hh=ww*N,c='g',label='Post. particles')
h['F'] = delta_plot(ax,F        ,c='r',label='Resampled particles')
h['G'] = delta_plot(ax,G        ,c='r',label='Jittered particles')

#sys.exit(0)

savefig_n.index = 1
def tog(h,save=True,*a,**b):
  toggle_viz(h,prompt=False,*a,**b)
  if save: savefig_n()

# Hide all elements
toggle_viz(h.values(),prompt=False) # legend=0 also nice

tog(h['prior'])
tog(h['lklhd'])
tog(h['postr'])
tog(h['lklhd'],0)
tog(h['postr'],0)
tog(h['E'])
tog(h['lklhd'])
tog(h['E'],0)
tog(h['A'])
tog(h['lklhd'])
tog(h['A'],0)
tog(h['F'])
tog(h['F'],0)
tog(h['G'])
tog(h['postr'])
