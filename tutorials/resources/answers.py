answers = {}

from IPython.display import display

answers['thesaurus'] = ["TXT",r"""
Data Assimilation (DA)  Statistical inference   Ensemble    
Filtering               Inverse problems        Sample     
Kalman filter (KF)      Inversion               
State estimation        Estimation              Stochastic 
Data fusion             Approximation           Random     
                        Regression              Monte-Carlo
Recursive               Fitting               
Sequential                                      data        
Iterative                                       measurements
Serial                                          observations
"""]

answers['why Gaussian'] =  ['MD',r"""
 * Pragmatic: leads to least-squares problems, which lead to linear systems of equations.
 * The central limit theorem (CLT) and all of its implications.
 * The condition "ML estimator = sample average" implies Gaussian sampling distributions.
 * See chapter 7 of: [Probability theory: the logic of science](https://books.google.com/books/about/Probability_Theory.html?id=tTN4HuUNXjgC) (Edwin T. Jaynes), which is an excellent book for understanding probability and statistics.
"""]

answers["error evolution"] = ["MD",r"""
* (a). $\frac{d \varepsilon}{dt} = \frac{d (x-z)}{dt}
= \frac{dx}{dt} - \frac{dz}{dt} = f(x) - f(z) \approx f(x) - [f(x) - \frac{df}{dx}\varepsilon ] = F \varepsilon$
* (b). Differentiate $e^{F t}$.
* (c1). Dissipates to 0.
* (c2). No. A balance is always reached between the uncertainty reduction $(1-K)$ and growth $F^2$. Also recall the asymptotic value of $P_k$ computed from [the previous section](T3 - Univariate Kalman filtering.ipynb#Exc-'Asymptotic-P':).
* (d). $\frac{d \varepsilon}{dt} \approx F \varepsilon + (f-g)$
* (e). [link](https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation)
"""]

# from markdown2 import markdown as md2html
from markdown import markdown as md2html
from IPython.display import HTML

bg_color = 'background-color:#d8e7ff;' #e2edff;'
def show_answer(excercise_tag):
    form, ans = answers[excercise_tag]
    ans = ans[1:] # Remove newline
    if   form == "HTML":
        source = ans
    elif form == "MD":
        source = md2html(ans)
    elif form == "TXT":
        source = '<code style="'+bg_color+'">'+ans+'</code>'
    source = ''.join([
        '<div ',
        'style="',bg_color,'padding:0.5em;">',
        str(source),
        '</div>'])
    display(HTML(source))
        
        
