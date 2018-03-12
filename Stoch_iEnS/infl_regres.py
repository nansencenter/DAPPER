from common import *
# dkObs: 2, 4, 6, 8
yp = array([1.25, 1.45, 1.60, 1.90])
ys = array([1.03, 1.10, 1.28, 1.40])
xx = array([1.08, 1.09, 1.25, 1.40])

fig, ax = plt.subplots()

ax.plot(xx,xx,'-o',label='-N')
ax.plot(xx,yp,'-o',label='PertObs')
ax.plot(xx,ys,'-o',label='Sqrt')

fp = np.polyfit(xx, yp, deg=1)
fs = np.polyfit(xx, ys, deg=1)
print(fp)
print(fs)

ax.plot(xx,np.poly1d(fp)(xx),'-o',label='Fit PertObs')
ax.plot(xx,np.poly1d(fs)(xx),'-o',label='Fit Sqrt')


ax.legend()


