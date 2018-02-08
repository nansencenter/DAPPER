# CD to DAPPER folder
from IPython import get_ipython
IP = get_ipython()
if IP.magic("pwd").endswith('tutorials'):
    IP.magic("cd ..")
else:
    assert IP.magic("pwd").endswith("DAPPER")

# Load DAPPER
from common import *

# Load answers
from tutorials.resources.answers import answers, show_answer

# Load widgets
from ipywidgets import *

####################################
# DA video
####################################
import io
import base64
from IPython.display import HTML
def envisat_video():
  caption = """Illustration of DA for the ozone layer in 2002.
  <br><br>
  LEFT: Satellite data (i.e. all that is observed).
  RIGHT: Simulation model with assimilated data.
  <br><br>
  Could you have perceived the <a href='http://dx.doi.org/10.1175/JAS-3337.1'>splitting of the ozone hole.</a> only from the satellite data?
  <br><br>
  Attribution: William A. Lahoz, DARC.
  """
  video = io.open('./data/figs/anims/darc_envisat_analyses.mp4', 'r+b').read()
  encoded = base64.b64encode(video)
  vid = HTML(data='''
  <figure style="width:580px;">
  <video alt="{1}" controls style="width:550px;">
  <source src="data:video/mp4;base64,{0}" type="video/mp4" />
  </video>
  <figcaption style="background-color:#d9e7ff;">{1}</figcaption>
  </figure>
  '''.format(encoded.decode('ascii'),caption))
  return vid



####################################
# EnKF animation
####################################
#   from matplotlib.image import imread
#   
#   # Hack to keep line-spacing constant with/out TeX
#   LE = '\phantom{$\{x_n^f\}_{n=1}^N$}'
#   
#   txts = [chr(i+97) for i in range(9)]
#   txts[0] = 'We consider a single cycle of the EnKF,'+\
#             'starting with the analysis state at time $(t-1)$.'+LE+'\n'+\
#             'The contours are "equipotential" curves of $\|x-\mu_{t-1}\|_{P_{t-1}}$.'+LE
#   txts[1] = 'The ensemble $\{x_n^a\}_{n=1}^N$ is (assumed) sampled from this distribution.'+LE+'\n'+LE
#   txts[2] = 'The ensemble is forecasted from time $(t-1)$ to $t$ '+\
#             'using the dynamical model $f$.'+LE+'\n'+\
#             'We now denote it using the superscript $f$.'+LE
#   txts[3] = 'Now we consider the analysis at time t. The ensemble is used'+LE+'\n'+\
#             'to estimate $\mu^f_t$ and $P^f_t$, i.e. the new contour curves.'+LE
#   txts[4] = 'The likelihood is taken into account...'+LE+'\n'+LE
#   txts[5] = '...which implicitly yields this posterior.' +LE+'\n'+LE
#   txts[6] = 'Explicitly, however,'+LE+'\n'+\
#             'we compute the Kalman gain, based on the ensemble estimates.'+LE
#   txts[7] = 'The Kalman gain is then used to shift the ensemble such that it represents' +LE+'\n'+\
#             'the (implicit) posterior. The cycle can then begin again, now from $t$ to $t+1$.'+LE
#   
#   def crop(img):
#       top = int(    0.05*img.shape[0])
#       btm = int((1-0.08)*img.shape[0])
#       lft = int(    0.01*img.shape[1])
#       rgt = int((1-0.01)*img.shape[1])
#       return img[top:btm,lft:rgt]
#   
#   def illust_EnKF(i):
#       with sns.axes_style("white"):
#           plt.figure(1,figsize=(10,12))
#           axI = plt.subplot(111)
#           axI.set_axis_off()
#           axI.set_title(txts[i],loc='left',usetex=True,size=15)
#           axI.imshow(crop(imread('./tutorials/resources/illust_EnKF/illust_EnKF_prez_'+str(i+8)+'.png')))
#           # Extract text:
#           #plt.savefig("images/txts_"+str(i+8)+'.png')
#           #bash: for f in `ls txts_*.png`; do convert -crop 800x110+120+260 $f $f; done
#   
#   EnKF_animation = interactive(illust_EnKF,i=IntSlider(min=0, max=7,continuous_update=False))


wI = Image(
    value=open("./tutorials/resources/illust_EnKF/illust_EnKF_prez_8.png", "rb").read(),
    format='png',
    width=600,
    height=400,
)
wT = Image(
    value=open("./tutorials/resources/illust_EnKF/txts_8.png", "rb").read(),
    format='png',
    width=600,
    height=50,
)
def show_image(i=0):
    img = "./tutorials/resources/illust_EnKF/illust_EnKF_prez_"+str(i+8)+".png"
    txt = "./tutorials/resources/illust_EnKF/txts_"+str(i+8)+".png"
    wI.value=open(img, "rb").read()
    wT.value=open(txt, "rb").read()
    
wS = interactive(show_image,i=(0,7,1))
EnKF_animation = VBox([wS,wT,wI])



####################################
# Misc
####################################
def plot_ensemble(E):
    E_with_NaNs = np.hstack([np.tile(E,(1,2)),np.nan*ones((len(E),1))]).ravel()
    Heights     = plt.ylim()[1]*0.5*np.tile(arange(3),(len(E),1)).ravel()
    plt.plot(E_with_NaNs,Heights,'k',lw=0.5,alpha=0.4,label="ensemble")

def piece_wise_DA_step_lines(xf,xa=None):
    if xa is None:
        xa = xf
    else:
        assert len(xf)==len(xa)
    # Assemble piece-wise lines for plotting purposes
    pw_f  = array([[xa[k  ], xf[k+1], nan] for k in range(len(xf)-1)]).ravel()
    pw_a  = array([[xf[k+1], xa[k+1], nan] for k in range(len(xf)-1)]).ravel()
    return pw_f, pw_a




