from common import *

fig, ax = plt.subplots()
t = np.linspace(0, 1, 201)
for freq in arange(1,7):
  ax.plot(t, np.sin(2*pi*freq*t), label=str(freq)+' Hz')

check = toggle_lines(ax)



