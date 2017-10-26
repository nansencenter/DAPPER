# from markdown2 import markdown as md2html
from markdown import markdown as md2html
from IPython.display import HTML, display

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
        

answers = {}


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

answers['pdf_G_1'] = ['MD',r'''
    return 1/sqrt(2*pi*P)*exp(-0.5*(xx-mu)**2/P)
    #return sp.stats.norm.pdf(xx,loc=mu,scale=sqrt(P))
''']

answers['BR deriv'] = ['MD',r'''
[Wiki](https://en.wikipedia.org/wiki/Bayes%27_theorem#Derivation)
''']

answers['BR grid normalization'] = ['MD',r'''
Because $p(y) = \int p(x,y) \, dx = \int p(x) p(y|x) \, dx$, 
which is what gets computed by the sum over the grid values together with `dx`.
''']

answers['num mult'] = ['MD',r'''
$(m_{grid})^d$
''']

answers['BR Gauss'] = ['MD',r'''
We can ignore all factors that do not depend on $x$.
\begin{align}
p(x|y) &= \frac{p(x) \, p(y|x)}{p(y)}
\propto p(x) \, p(y|x) \\\
&\propto N(x|b,B) \, N(y|x,R) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (x-b)^2/B + (x-y)^2/R \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (1/B + 1/R)x^2 - 2(b/B + y/R)x \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( x - \frac{b/B + y/R}{1/B + 1/R} \Big)^2 \cdot (1/B + 1/R) \Big) \\\
&\propto N(x|\mu,P) \, ,
\end{align}
i.e., by identification,
$ p(x|y) = N(x|\mu,P) \, ,$
with
\begin{align}
    P &= 1/(1/B + 1/R) \, , \\\
  \mu &= P(b/B + y/R) \, .
\end{align}
''']

answers['KG 2'] = ['MD',r'''
 * Because it drags the estimate from $b$ "towards" $y$. Because it is beteen 0 and 1.
   It weights the observation noise level (R) vs. the total noise level (B+R).
   In the multivariate case, the same holds for its eigenvectors (if $H=I$).
''']

answers['BR Gauss code'] = ['MD',r'''
    P  = 1/(1/B+1/R)
    mu = P*(b/B+y/R)
    #     KG = B/(B+R)
    #     P  = (1-KG)*B
    #     mu = b + KG*(y-b)
''']

answers['LinReg deriv'] = ['MD',r'''
$$ \frac{d J_K}{d\alpha} = 0 = \ldots $$
''']

answers['LinReg F_k'] = ['MD',r'''
$$ F_k = \frac{k+1}{k} $$
''']

answers['LinReg func'] = ['MD',r'''
    kk = arange(1,k+1)
    alpha = sum(kk*yy[:k]) / sum(kk**2)
''']

answers['KF func'] = ['MD',r'''
    # Forecast
    muf[k+1] = F(k)*mua[k]
    PPf[k+1] = F(k)*PPa[k]*F(k) + Q
    # Analysis
    PPa[k+1] = 1/(1/PPf[k+1] + H*1/R*H)
    mua[k+1] = PPa[k+1] * (muf[k+1]/PPf[k+1] + yy[k]*H/R)
    #KG = PPf[k+1]*H / (H*PPf[k+1]*H + R)
    #PPa[k+1] = (1-KG)*PPf[k+1]
    #mua[k+1] = muf[k+1]+KG*(yy[k]-muf[k+1])
''']

answers['KF KG fail'] = ['MD',r'''
Because `PPa[0]` is infinite. And while the limit (as `PPf` goes to +infinity) of `KG = PPf*H / (H*PPf*H + R)` is `H (= 1)`, its numerical evaluation fails (as it should). Note that the infinity did not cause any problems numerically for the "weighted average" form.
''']

answers['LinReg plot'] = ['MD',r'''
Let $\alpha_K$ denote the linear regression estimates (of the slope) based on the observations $y_{1:K} = \\{y_1,\ldots,y_K\\}$.
Simiarly, let $\mu_K$ denote the KF estimate of $x_K$ based on $y_{1:K}$.
It can bee seen in the plot that
$
K \alpha_K = \mu_K \, .
$
''']

answers['KF = LinReg a'] = ['MD',r'''
We'll proceed by induction. With $P_0 = \infty$, we get $P_1 = R$, which initializes (4). Now, from (3):

$$
\begin{align}
P_{K+1} &= 1\Big/\big(1/R + \textstyle (\frac{K}{K+1})^2 / P_K\big)
\\\
&= R\Big/\big(1 + \textstyle (\frac{K}{K+1})^2 \frac{\sum_{k=1}^K k^2}{K^2}\big)
\\\
&= R\Big/\big(1 + \textstyle \frac{\sum_{k=1}^K k^2}{(K+1)^2}\big)
\\\
&= R(K+1)^2\Big/\big((K+1)^2 + \sum_{k=1}^K k^2\big)
\\\
&= R(K+1)^2\Big/\sum_{k=1}^{K+1} k^2
\, ,
\end{align}
$$
which concludes the induction.

The proof for (b) is similar.
''']

answers['Asymptotic P'] = ['MD',r'''
The fixed point $P_\infty$ should satisfy
$P_\infty = 1/\big(1/R + 1/[F^2 P_\infty]\big)$.
This yields $P_\infty = R (1-1/F^2)$.
''']



answers["error evolution"] = ["MD",r"""
* (a). $\frac{d \varepsilon}{dt} = \frac{d (x-z)}{dt}
= \frac{dx}{dt} - \frac{dz}{dt} = f(x) - f(z) \approx f(x) - [f(x) - \frac{df}{dx}\varepsilon ] = F \varepsilon$
* (b). Differentiate $e^{F t}$.
* (c1). Dissipates to 0.
* (c2). No. A balance is always reached between the uncertainty reduction $(1-K)$ and growth $F^2$. Also recall the asymptotic value of $P_k$ computed from [the previous section](T3 - Univariate Kalman filtering.ipynb#Exc-'Asymptotic-P':).
* (d). $\frac{d \varepsilon}{dt} \approx F \varepsilon + (f-g)$
* (e). [link](https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation)
"""]



answers['Gaussian sampling a'] = ['MD',r'''
Firstly, a linear (affine) transformation can be decomposed into a sequence of sums. This means that $\mathbf{x}$ will be Gaussian.
It remains only to calculate its moments.

By the [linearity of the expected value](https://en.wikipedia.org/wiki/Expected_value#Linearity),
$$E(\mathbf{x}) = E(\mathbf{L} \mathbf{z} + \mathbf{b}) = \mathbf{L} E(\mathbf{z}) + \mathbf{b} = \mathbf{b} \, .$$

Moreover,
$$\newcommand{\b}{\mathbf{b}} \newcommand{\x}{\mathbf{x}} \newcommand{\z}{\mathbf{z}} \newcommand{\L}{\mathbf{L}}
E((\x - \b)(\x - \b)^T) = E((\L \z)(\L \z)^T) = \L E(\z^{} \z^T) \L^T = \L \mathbf{I}_m \L^T = \L \L^T \, .$$
''']
answers['Gaussian sampling b'] = ['MD',r'''
Type `randn??` in a code cell and execute it.
''']
answers['Gaussian sampling c'] = ['MD',r'''
    z = randn((m,1))
    x = mu + L @ z
''']

answers['Gaussian sampling d'] = ['MD',r'''
    mu_vertical = 10*ones((m,1))
    E = mu_vertical + L @ randn((m,N))
    #E = np.random.multivariate_normal(mu,P,N).T
''']

answers['Average sampling error'] = ['MD',r'''
Procedure:

 1. Repeat the experiment many times.
 2. Compute the average error ("bias") of $\overline{\mathbf{x}}$. Verify that it converges to 0 as $N$ is increased.
 3. Compute the average *squared* error. Verify that it is approximately $\text{diag}(\mathbf{P})/N$.
''']

answers['ensemble moments'] = ['MD',r'''
    x_bar = np.sum(E,axis=1)/N
    P_bar = zeros((m,m))
    for n in range(N):
        anomaly = (E[:,n] - x_bar)[:,None]
        P_bar += anomaly @ anomaly.T
        #P_bar += np.outer(anomaly,anomaly)
    P_bar /= (N-1)
''']

answers['Why (N-1)'] = ['MD',r'''
 * [Unbiased](https://en.wikipedia.org/wiki/Variance#Sample_variance)
 * Suppose we compute the square root of this estimate. Is this an unbiased estimator for the standard deviation?
''']

answers['ensemble moments vectorized'] = ['MD',r'''
 * (a). Show that element $(i,j)$ of the matrix product $\mathbf{A}^{} \mathbf{B}^T$
 equals element $(i,j)$ of the sum of the outer product of their columns: $\sum_n \mathbf{a}_n \mathbf{b}_n^T$. Put this in the context of $\overline{\mathbf{P}}$.
 * (b). Use the following
 
code:

    x_bar = np.sum(E,axis=1,keepdims=True)/N
    A     = E - x_bar
    P_bar = A @ A.T / (N-1)   
''']

# Skipped
answers['Why matrix notation'] = ['MD',r'''
   - Removes indices
   - Highlights the linear nature of many computations.
   - Tells us immediately if we're working in state space or ensemble space
     (i.e. if we're manipulating individual dimensions, or ensemble members).
   - Helps with understanding subspace rank issues
   - Highlights how we work with the entire ensemble, and not individual members.
   - Suggest a deterministic parameterization of the distributions.
''']

answers['estimate cross'] = ['MD',r'''
    def estimate_cross_cov(E1,E2):
        N = E1.shape[1]
        assert N==E2.shape[1]
        A1 = E1 - np.mean(E1,axis=1,keepdims=True)
        A2 = E2 - np.mean(E2,axis=1,keepdims=True)
        CC = A1 @ A2.T / (N-1)
        return CC
''']

answers['errors'] = ['MD',r'''
 * (a). Error: discrepancy from estimator to the parameter targeted.
Residual: discrepancy from explained to observed data.
 * (b). Bias = *average* (i.e. systematic) error.
 * (c). [Wiki](https://en.wikipedia.org/wiki/Mean_squared_error#Proof_of_variance_and_bias_relationship)
''']


# Also comment on CFL condition (when resolution is increased)?
answers['Cov memory'] = ['MD',r'''
 * (a). $m$-by-$m$
 * (b). Using the [cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition#Computation),
    at least 2 times $m^3/3$.
 * (c). Assume $\mathbf{P}$ stored as float (double). Then it's 8 bytes/element.
 And the number of elements in $\mathbf{P}$: $m^2$. So the total memory is $8 m^2$.
 * (d). 8 trillion bytes. I.e. 8 million MB. 
''']

answers['EnKF v1'] = ['MD',r'''
    def my_EnKF(N):
        E = mu0[:,None] + P0_chol @ randn((m,N))
        for k in range(1,K+1):
            # Forecast
            t   = k*dt
            E   = f(E,t-dt,dt)
            E  += Q_chol @ randn((m,N))
            if not k%dkObs:
                # Analysis
                y        = yy[k//dkObs-1] # current obs
                hE       = h(E,t)
                PH       = estimate_cross_cov(E,hE)
                HPH      = estimate_mean_and_cov(hE)[1]
                Perturb  = R_chol @ randn((p,N))
                KG       = divide_1st_by_2nd(PH, HPH+R)
                E       += KG @ (y[:,None] - Perturb - hE)
            mu[k] = mean(E,axis=1)
''']

answers['rmse'] = ['MD',r'''
    rmses = sqrt(np.mean((xx-mu)**2, axis=1))
    average = np.mean(rmses)
''']

answers['Repeat experiment a'] = ['MD',r'''
 * (a). Set `p=1` above, and execute all cells below again.
''']

answers['Repeat experiment b'] = ['MD',r'''
 * (b). Insert `seed(i)` for some number `i` above the call to the EnKF or above the generation of the synthetic truth and obs.
''']

answers['Repeat experiment cd'] = ['MD',r'''
 * (c). Void.
 * (d). Use: `Perturb  = D_infl * R_chol @ randn((p,N))` in the EnKF algorithm.
''']

answers['jagged diagnostics'] = ['MD',r'''
Because they are only defined at analysis times, i.e. every `dkObs` time step.
''']

answers['RMSE hist'] = ['MD',r'''
 * The MSE will be (something close to) chi-square.
 * Gaussianity
''']

answers['Rank hist'] = ['MD',r'''
 * U-shaped: Too confident
 * A-shaped: Too uncertain
 * Flat: well calibrated
''']

# Pointless...
# Have a look at the phase space trajectory output from `plot_3D_trajectory` above.
# The "butterfly" is contained within a certain box (limits for $x$, $y$ and $z$).
answers['RMSE vs inf error'] = ['MD',r'''
It follows from [the fact that](https://en.wikipedia.org/wiki/Lp_space#Relations_between_p-norms)
$ \newcommand{\x}{\mathbf{x}} \|\x\|_2 \leq m^{1/2} \|\x\|\_\infty \text{and}  \|\x\|_1 \leq m^{1/2} \|\x\|_2$
that
$$ 
\text{RMSE} 
= \frac{1}{K}\sum_k \text{RMSE}_k
\leq \| \text{RMSE}\_{0:k} \|\_\infty
$$
and
$$ \text{RMSE}_k = \| \text{Error}_k \|\_2 / \sqrt{m} \leq \| \text{Error}_k \|\_\infty$$
''']

answers['Twin Climatology'] = ['MD',r'''
    config = Climatology(**defaults)
    avergs = config.assimilate(setup,xx,yy).average_in_time()
    print_averages(config,avergs,[],['rmse_a','rmv_a'])
''']

answers['Twin Var3D'] = ['MD',r'''
    config = Var3D(**defaults)
    ...
''']


