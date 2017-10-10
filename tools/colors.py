from common import *

sns_bg = array([0.9176, 0.9176, 0.9490])

# Standard color codes
RGBs = {c: array(mpl.colors.colorConverter.to_rgb(c)) for c in 'bgrmyckw'}
#RGBs = [mpl.colors.colorConverter.to_rgb(c) for c in 'bgrmyckw']

# Matlab (new) colors.
ml_colors = np.array(np.matrix("""
     0    0.4470    0.7410;
0.8500    0.3250    0.0980;
0.9290    0.6940    0.1250;
0.4940    0.1840    0.5560;
0.4660    0.6740    0.1880;
0.3010    0.7450    0.9330;
0.6350    0.0780    0.1840 
"""))
# Load into matplotlib color dictionary
for code, color in zip('boyvgcr', ml_colors):
  mpl.colors.ColorConverter.colors['ml'+code] = color
  mpl.colors.colorConverter.cache ['ml'+code] = color

# Seaborn colors
sns_colors = np.array(np.matrix("""
0.298 , 0.447 , 0.690 ; 
0.333 , 0.658 , 0.407 ; 
0.768 , 0.305 , 0.321 ; 
0.505 , 0.447 , 0.698 ; 
0.8   , 0.725 , 0.454 ; 
0.392 , 0.709 , 0.803 ; 
0.1   , 0.1   , 0.1   ; 
1.0   , 1.0   , 1.0    
"""))
# Overwrite default color codes
for code, color in zip('bgrmyckw', sns_colors):
    mpl.colors.colorConverter.colors[code] = color
    mpl.colors.colorConverter.cache [code] = color


def blend_rgb(rgb, a, bg_rgb=ones(3)):
  """
  Fake RGB transparency by blending it to some background.
  Useful for creating gradients.

  Also useful for creating 'transparency' for exporting to eps.
  But there's no actualy transparency, so superposition of lines
  will not work. For that: export to pdf, or make do without.

   - rgb: N-by-3 rgb, or a color code.
   - a: alpha value
   - bg_rgb: background in rgb. Default: white
  Based on stackoverflow.com/a/33375738/38281
  """ 
  if isinstance(rgb,str):
    rgb = mpl.colors.colorConverter.to_rgb(rgb)
  return [a*c1 + (1-a)*c2 for (c1, c2) in zip(rgb, bg_rgb)]

