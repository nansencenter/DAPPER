from common import *
plt.style.use('AdInf/paper.mplstyle')
rgba = lambda c: blend_rgb(c,0.4)

fig = plt.figure(1)
fig.clear()
plt.pause(0.1)
ax = plt.gca()


def txt(x,y,s,**kwargs):
  plt.text(x,y,s,
      horizontalalignment='center',
      verticalalignment='center',
      **kwargs
      )

x0 = 0.1
D  = 0.4
d  = 0.04

# The bold, roman \bm{x} is not in mathtex. stackoverflow.com/a/14324826/38281
# And I don't want usetex=True, coz that exports huge files.
txt(x0 + 0,x0 + D,'$\mathbf{x}_{k-1}$')
txt(x0 + D,x0 + D,'$\mathbf{x}_{k}$')
txt(x0 + 0,x0 + 0,'$\\beta_{k-1}$')
txt(x0 + D,x0 + 0,'$\\beta_{k}$')

opts = {'head_width':0.03,'head_length':0.03}

# Horizontal
plt.arrow(x0+2*d    ,x0+D,D-4.0*d,0       ,color='k',lw=2.5,**opts)
plt.arrow(x0+2*d    ,x0+0,D-4.0*d,0       ,color='k',lw=2.5,**opts)
# Cross:
plt.arrow(x0+2*d    ,x0+0,D-3.6*d,+D-1.5*d,color='r',ls='--',lw=2.5,**opts)
plt.arrow(x0+2*d    ,x0+D,D-3.6*d,-D+1.5*d,color='b',ls=':' ,lw=2.5,**opts)
# Vertical
x = x0+D-0.2*d
plt.plot([x, x],[x0+1.5*d,x0+D-1.5*d],'k',lw=2.5,ls='-')
plt.arrow(x,x0+D-2*d,0,0.3*d,color='k',lw=2.5,**opts)


plt.arrow(x0+1.42*d+D,x0+D,D-4.0*d,0       , lw=2.5,color=rgba('k'),**opts)
plt.arrow(x0+1.42*d+D,x0+0,D-4.0*d,0       , lw=2.5,color=rgba('k'),**opts)
plt.arrow(x0+1.42*d+D,x0+0,D-3.6*d,+D-1.5*d, lw=2.5,color=rgba('r'),ls='--',**opts)
plt.arrow(x0+1.42*d+D,x0+D,D-3.6*d,-D+1.5*d, lw=2.5,color=rgba('b'),ls=':' ,**opts)
plt.plot([x+D, x+D],[x0+1.5*d,x0+D-1.5*d],lw=2.5,ls='-',color=rgba('k'))
plt.arrow(x+D,x0+D-2*d,0,0.3*d,lw=2.5,color=rgba('k'),**opts)
txt(x0 + 2*D,x0 + D,'$\mathbf{x}_{k+1}$',color=rgba('k'))
txt(x0 + 2*D,x0 + 0,'$\\beta_{k+1}$',color=rgba('k'))

#ax.grid(color='k',axis='both',which='both')
plt.axis('off')
plt.xlim(0,1)
plt.ylim(0,1)
