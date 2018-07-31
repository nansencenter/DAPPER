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


answers['thesaurus 1'] = ["TXT",r"""
Data Assimilation (DA)     Ensemble      Stochastic     Data        
Filtering                  Sample        Random         Measurements
Kalman filter (KF)         Set of draws  Monte-Carlo    Observations
State estimation           
Data fusion                
"""]

answers['thesaurus 2'] = ["TXT",r"""
Statistical inference    Ensemble member     Quantitative belief    Recursive 
Inverse problems         Sample point        Probability            Sequential
Inversion                Realization         Relative frequency     Iterative 
Estimation               Single draw                                Serial    
Approximation            Particle
Regression               
Fitting                  
"""]

answers['why Gaussian'] =  ['MD',r"""
 * Pragmatic: leads to least-squares problems, which lead to linear systems of equations.
   This was demonstrated by the simplicity of the parametric Gaussian-Gaussian Bayes' rule.
 * The central limit theorem (CLT) and all of its implications.
 * The intuitive condition "ML estimator = sample average" implies the sample is drawn from a Gaussian.
 * For more, see chapter 7 of: [Probability theory: the logic of science](https://books.google.com/books/about/Probability_Theory.html?id=tTN4HuUNXjgC) (Edwin T. Jaynes), which is an excellent book for understanding probability and statistics.
"""]

answers['pdf_G_1'] = ['MD',r'''
    pdf_values = 1/sqrt(2*pi*P)*exp(-0.5*(x-mu)**2/P)
    # Version using the scipy (sp) library:
    # pdf_values = sp.stats.norm.pdf(x,loc=mu,scale=sqrt(P))
''']

answers['pdf_U_1'] = ['MD',r'''
    def pdf_U_1(x,mu,P):
        # Univariate (scalar), Uniform pdf

        pdf_values = ones((x-mu).shape)

        a = mu - sqrt(3*P)
        b = mu + sqrt(3*P)

        pdf_values[x<a] = 0
        pdf_values[x>b] = 0

        height = 1/(b-a)
        pdf_values *= height

        return pdf_values
''']

answers['BR deriv'] = ['MD',r'''
<a href="https://en.wikipedia.org/wiki/Bayes%27_theorem#Derivation" target="_blank">Wikipedia</a>

''']

answers['BR grid normalization'] = ['MD',r'''
Because it can compute $p(y)$ as
the factor needed to normalize to 1,
as required by the definition of pdfs.

That's what the `#normalization` line does.

Here's the proof that the normalization (which makes `pp` sum to 1) is equivalent to dividing by $p(y)$:
$$\texttt{sum(pp)*dx} \approx \int p(x) p(y|x) \, dx = \int p(x,y) \, dx = p(y) \, .$$
''']

answers['Dimensionality a'] = ['MD',r'''
$N^m$
''']
answers['Dimensionality b'] = ['MD',r'''
$15 * 360 * 180 = 972'000 \approx 10^6$
''']
answers['Dimensionality c'] = ['MD',r'''
$10^{10^6}$
''']

answers['BR Gauss'] = ['MD',r'''
We can ignore factors that do not depend on $x$.

\begin{align}
p(x|y)
&= \frac{p(x) \, p(y|x)}{p(y)} \\\
&\propto p(x) \, p(y|x) \\\
&=       N(x|b,B) \, N(y|x,R) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (x-b)^2/B + (x-y)^2/R \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( (1/B + 1/R)x^2 - 2(b/B + y/R)x \Big) \Big) \\\
&\propto \exp \Big( \frac{-1}{2} \Big( x - \frac{b/B + y/R}{1/B + 1/R} \Big)^2 \cdot (1/B + 1/R) \Big) \, .
\end{align}

The last line can be identified as $N(x|\mu,P)$ as defined above.
''']

answers['KG 2'] = ['MD',r'''
Because it

 * drags the estimate from $b$ "towards" $y$.
 * is between 0 and 1.
 * weights the observation noise level (R) vs. the total noise level (B+R).
 * In the multivariate case (and with $H=I$), the same holds for its eigenvectors.
''']

answers['BR Gauss code'] = ['MD',r'''
    P  = 1/(1/B+1/R)
    mu = P*(b/B+y/R)
    # Gain version:
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
    # Analysis -- Kalman gain version:
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

answers["Hint: Lorenz energy"] = ["MD",r'''
Hint: what's its time-derivative?
''']

answers["Lorenz energy"] = ["MD",r'''
\begin{align}
\frac{d}{dt}
\sum_i
x_i^2
&=
2 \sum_i
x_i \dot{x}_i
\end{align}

Next, insert the quadratic terms from the ODE,
$
\dot x_i = (x_{i+1} − x_{i-2}) x_{i-1}
\, .
$

Finally, apply the periodicity of the indices.
''']

answers["error evolution"] = ["MD",r"""
* (a). $\frac{d \varepsilon}{dt} = \frac{d (x-z)}{dt}
= \frac{dx}{dt} - \frac{dz}{dt} = f(x) - f(z) \approx f(x) - [f(x) - \frac{df}{dx}\varepsilon ] = F \varepsilon$
* (b). Differentiate $e^{F t}$.
* (c).
    * (1). Dissipates to 0.
    * (2). No.
      A balance is always reached between
      the uncertainty reduction $(1-K)$ and growth $F^2$.  
      Also recall the asymptotic value of $P_k$ computed from
      [the previous tutorial](T3 - Univariate Kalman filtering.ipynb#Exc-3.14-'Asymptotic-P':).
* (d). [link](https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation)
* (e). $\frac{d \varepsilon}{dt} \approx F \varepsilon + (f-g)$
"""]

answers["doubling time"] = ["MD",r"""
    xx   = output_63[0][:,-1]      # Ensemble of particles at the end of integration
    v    = mean(v)                 # homogenize
    d    = sqrt(v)                 # std. dev.
    eps  = [FILL IN SLIDER VALUE]  # initial spread
    T    = [FILL IN SLIDER VALUE]  # integration time
    rate = log(d/eps)/T            # assuming d = eps*exp(rate*T)
    print("Doubling time (approx):",log(2)/rate)
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
    x = b + L @ z
''']

answers['Gaussian sampling d'] = ['MD',r'''
    b_vertical = 10*ones((m,1))
    E = b_vertical + L @ randn((m,N))
    #E = np.random.multivariate_normal(b,P,N).T
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
 * That the estimator and truth are independent, Gaussian random variables.
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


answers['forward_euler'] = ['MD', r'''
Missing line:

    xyz_step = xyz + dxdt(xyz, h, sigma=SIGMA, beta=BETA, rho=RHO) * h
''']

answers['log_growth'] = ['MD', r'''
Missing lines:

    nrm = sqrt( (x_pert_k - x_control_k) @ (x_pert_k - x_control_k).T )

    log_growth_rate = (1.0 / T) * log(nrm / eps)
''']


answers['power_method'] = ['MD', r'''
Missing lines:

        v = M @ v
        v = v / sqrt(v.T @ v)
    
    mu = v.T @ M @ v
''']

answers['power_method_convergence_rate'] = ['HTML', r'''
Suppose we have a random vector <span style="font-size:1.25em">$\mathbf{v}_0$</span>.  If <span style="font-size:1.25em">$\mathbf{M}$</span> is diagonalizable, then we can write <span style="font-size:1.25em">$\mathbf{v}_0$</span> in a basis of eigenvectors, i.e.,
<h3>$$v_0 = \sum_{j=1}^n \alpha_j \nu_j,$$ </h3>
where  <span style="font-size:1.25em">$\nu_j$</span> is an eigenvector for the eigenvalue  <span style="font-size:1.25em">$\mu_j$</span>, and  <span style="font-size:1.25em">$\alpha_j$</span> is some coefficient in  <span style="font-size:1.25em">$\mathbb{R}$</span>.  We consider thus, with probability one,  <span style="font-size:1.25em">$\alpha_1 \neq 0$</span>. 


In this case, we note that
<h3>
$$\mathbf{M}^k \mathbf{v}_0 = \mu_1^k  \left( \alpha_1 \nu_1 + \sum_{j=2}^n \alpha_j \left(\frac{\mu_j}{\mu_1}\right)^k \nu_j\right).
$$</h3>

But 
<h3>$$\frac{\rvert \mu_j\rvert}{\rvert\mu_1\rvert} <1$$</h3>
for each  <span style="font-size:1.25em">$j>1$</span>, so that the projection of  <span style="font-size:1.25em">$\mathbf{M}^k \mathbf{v}_0$</span> into each eigenvector  <span style="font-size:1.25em">$\{\nu_j\}_{j=2}^n$</span> goes to zero at a rate of at least  
<h3>
$$\mathcal{O}  \left(\left[ \frac{\lambda_2}{\lambda_1} \right]^k \right).$$
</h3>
We need only note that  <span style="font-size:1.25em">$\mathbf{M}^k \mathbf{v}_0$</span> and  <span style="font-size:1.25em">$\mathbf{v}_{k}$</span> share the same span.
''']


answers['lyapunov_exp_power_method'] = ['HTML', r'''
<ol>
<li>Consider, if  <span style="font-size:1.25em">$ \widehat{\mu}_k \rightarrow \mu_1$</span>  as  <span style="font-size:1.25em">$k \rightarrow \infty$</span>, then for all  <span style="font-size:1.25em">$\epsilon>0$</span> there exists a  <span style="font-size:1.25em">$T_0$</span> such that,<h3>$ \rvert \mu_1 \rvert - \epsilon < \rvert \widehat{\mu}_k\rvert < \rvert \mu_1 \rvert + \epsilon $, </h3>
<br>
for all <span style="font-size:1.25em">$k > T_0$</span>.  In particular, we will choose some  <span style="font-size:1.25em">$\epsilon$ </span> sufficiently small such that,
<h3>$$\begin{align}
\rvert \mu_1 \rvert - \epsilon > 0.
\end{align}$$</h3>
<br>
This is possible by the assumption <span style="font-size:1.25em">$\rvert \mu_1 \rvert >0$</span>.

We will write,
<h3>$\widehat{\lambda}_T =\frac{1}{T} \sum_{k=1}^{T_0} \log\left(\rvert \widehat{\mu}_1 \rvert\right) + \frac{1}{T} \sum_{k=T_0 +1}^T  \log \left(\rvert \widehat{\mu}_1 \rvert\right)$. </h3>
<br>
We note that  <span style="font-size:1.25em">$\log$</span> is monotonic, so that for  <span style="font-size:1.25em">$T> T_0$</span>,

<h3>$\frac{1}{T} \sum_{k=T_0 +1}^T  \log\left(\rvert \mu_1\rvert - \epsilon \right) < \frac{1}{T} \sum_{k=T_0 +1}^T  \log\left(\rvert\widehat{\mu}_k \rvert \right) <\frac{1}{T} \sum_{k=T_0 +1}^T  \log\left(\rvert \mu_1\rvert + \epsilon \right)$.</h3>
<br>
But that means,

<h3>$\frac{T - T_0}{T} \log\left(\rvert \mu_1\rvert - \epsilon \right) < \frac{1}{T} \sum_{k=T_0 +1}^T  \log\left(\rvert \widehat{\mu}_k \rvert \right) <\frac{T - T_0}{T}  \log\left(\rvert \mu_1\rvert  + \epsilon \right)$.</h3>
<br>
Notice that in limit,

<h3>$\lim_{T\rightarrow \infty}\frac{1}{T}\sum_{k=1}^{T_0} \log\left(\rvert \widehat{\mu}_k\rvert \right) = 0$, </h3>
<br>
and therefore we can show,

<h3>$\log\left(\rvert \mu_1 \rvert - \epsilon \right) < \lim_{T \rightarrow \infty} \widehat{\lambda}_T < \log\left(\rvert \mu_1 \rvert + \epsilon \right),$</h3>
<br>
for all  <span style="font-size:1.25em">$\epsilon >0$</span>.  This shows that

<h3>$ \lim_{T \rightarrow \infty} \widehat{\lambda}_T = \log\left(\rvert \mu_1\rvert \right). $</h3></li>
<br>
<li>
The Lyapunov exponents for the fixed matrix  <span style="font-size:1.25em">$\mathbf{M}$</span> are determined by the log, absolute value of the eigenvalues.
</li>
</ol>
''']


answers['fixed_point'] = ['HTML', r'''
Suppose for all components  <span style="font-size:1.25em">$x^j$</span> we choose  <span style="font-size:1.25em">$x^j = F$</span>.  The time derivative at this point is clearly zero.
''']

answers['probability_one'] = ['HTML', r'''
We relied on the fact that there is probability one that a Gaussian distributed vector has a nonzero projection into the eigenspace for the leading eigenvalue.  Consider why this is true.

Let  <span style="font-size:1.25em">$\{\mathbf{v}_j \}_{j=1}^n $</span> be any orthonormal basis such that  <span style="font-size:1.25em">$\mathbf{v}_1$</span> is an eigenvector for  <span style="font-size:1.25em">$\mu_1$</span>.  Let 
<h3>$$
\chi(\mathbf{x}) : \mathbb{R}^n \rightarrow \{0, 1\}
$$</h3>
<br>
be the indicator function on the span of  <span style="font-size:1.25em">$\{\mathbf{v}_j\}_{j=2}^n$</span>, i.e., the hyper-plane orthogonal to  <span style="font-size:1.25em">$\mathbf{v}_1$</span>.  The probability of choosing a Gaussian distributed random vector that has no component in the span of <span style="font-size:1.25em">$\mathbf{v}_1$</span> is measured by integrating
<h2>$$
\frac{1}{\left(2\pi\right)^n}\int_{\mathbb{R}} \cdots \int_{\mathbb{R}}\chi\left(\sum_{j=1}^n  \alpha_j v_j \right)
 e^{\frac{-1}{2} \sum_{j=1}^n \alpha_j^2 }
{\rm d}\alpha_1  \cdots {\rm d} \alpha_n.
$$</h2>
<br>
But  <span style="font-size:1.25em">$\chi \equiv 0$</span> whenever  <span style="font-size:1.25em">$\alpha_1 \neq 0$</span>, and  <span style="font-size:1.25em">${\rm d} \alpha_1 \equiv 0$</span> on this set.  This means that the probability of selecting a Gaussian distributed vector with  <span style="font-size:1.25em">$\alpha_1 =0$</span> is equal to zero.
<br>
In more theoretical terms, this corresponds to the hyper-plane having measure zero with respect to the Lebesgue measure.
''']

answers['gram-schmidt'] = ['HTML', r'''
The vectors, <span style="font-size:1.25em">$\{\mathbf{x}_0^1, \mathbf{x}_0^2 \}$</span> are related to the vectors <span style="font-size:1.25em">$\{\mathbf{x}_1^1, \mathbf{x}_1^2 \}$</span> by propagating forward via the matrix <span style="font-size:1.25em">$\mathbf{M}$</span>, and the Gram-Schmidt step.  Thus by writing,
<h3>$$\begin{align}
\widehat{\mathbf{x}}_1^2 &\triangleq \mathbf{y}^2_1  + \langle \mathbf{x}_1^1,  \widehat{\mathbf{x}}^2_1\rangle \mathbf{x}_1^1
\end{align}$$</h3>
<br>
it is easy to see
<h3>$$\begin{align}
\mathbf{M} \mathbf{x}_0^1 &= U^{11}_1 \mathbf{x}_1^1 \\
\mathbf{M} \mathbf{x}_0^2 &= U^{22}_1 \mathbf{x}_1^2 + U^{12}_1 \mathbf{x}_1^1.
\end{align}$$</h3>
<br>
This leads naturally to an upper triangular matrix recursion.  Define the following matrices, for <span style="font-size:1.25em">$k \in \{1,2, \cdots\}$</span>
<h3>$$\begin{align}
\mathbf{U}_k \triangleq \begin{pmatrix}
U_k^{11} & U_k^{12} \\
0 & U_k^{22}
\end{pmatrix} & & \mathbf{E}_{k-1} \triangleq \begin{pmatrix}
\mathbf{x}_{k-1}^{1} & \mathbf{x}_{k-1}^{2},
\end{pmatrix}
\end{align}$$</h3>
<br>
then in matrix form, we can write the recursion for an arbitrary step $k$ as
<h3>$$\begin{align}
\mathbf{M} \mathbf{E}_k = \mathbf{E}_{k+1} \mathbf{U}_{k+1}
\end{align}$$</h3>
<br>
where the coefficients of <span style="font-size:1.25em">$\mathbf{U}_k$</span> are defined by the Gram-Schmidt step. described above.
''']

answers['schur_decomposition'] = ['HTML', r'''
We can compute the eigenvalues as the roots of the characteristic polynomial.  Specifically, the characteristic polynomial is equal to
<h3>$$\begin{align}
\det\left( \mathbf{M} - \lambda \mathbf{I} \right) &= \det\left( \mathbf{Q} \mathbf{U} \mathbf{Q}^{\rm T} - \lambda\mathbf{I} \right) \\
&=\det\left( \mathbf{Q}\left[ \mathbf{U}   - \lambda\mathbf{I}\right] \mathbf{Q}^{\rm T} \right) \\
&=\det\left( \mathbf{Q}\right) \det\left( \mathbf{U}   - \lambda\mathbf{I}\right) \det\left(\mathbf{Q}^{\rm T} \right) \\
&=\det\left( \mathbf{Q} \mathbf{Q}^{\rm T} \right) \det\left( \mathbf{U}   - \lambda\mathbf{I}\right) \\
&=\det\left( \mathbf{U}   - \lambda\mathbf{I}\right)
\end{align}$$ </h3>
<br>
By expanding the determinant in co-factors, it is easy to show that the determinant of the right hand side equals
<h3>$$\begin{align}
\prod_{j=1}^n (U^{jj} - \lambda).
\end{align}$$ </h3>
<br>

By orthogonality, it is easy to verify that 
<h3>$$\begin{align}
\left(\mathbf{Q}^j\right)^{\rm T} \mathbf{M} \mathbf{Q}^j = U^{jj}.
\end{align}$$</h3>

''']

answers['lyapunov_vs_es'] = ['HTML', r'''
We define the <b><em>i</em>-th Lyapunov exponent</b> as
<h3>$$\begin{align}
\lambda_i & \triangleq \lim_{k\rightarrow \infty} \frac{1}{k}\sum_{j=1}^k \log\left(\left\rvert U_j^{ii}\right\rvert \right)
\end{align}$$</h3>
<br>
and the <b><em>i</em>-th (backward) Lyapunov vector at time <em>k</em></b> to be the $i$-th column of <span style="font-size:1.25em"> $\mathbf{E}_k$ </span>.
''']

answers['naive_QR'] = ['MD', r'''
Example solution:

        perts[:, i] = perts[:, i] - sqrt(perts[:, i].T @ perts[:, j]) perts[:, j]
    
    perts[:, i] = perts[:, i] / sqrt(perts[:,i].T @ perts[:, i])
''']

answers['real_schur'] = ['HTML', r'''
Let <span style='font-size:1.25em'>$\mathbf{M}$</span> be any matrix in <span style='font-size:1.25em'>$\mathbb{R}^{n\times n}$</span> with eigenvalues ordered
<h3>
$$
\begin{align}
\rvert \mu_1 \rvert \geq \cdots \geq \rvert \mu_s \rvert.
\end{align}
$$
</h3>
<br>
A real Schur decomposition of <span style='font-size:1.25em'>$\mathbf{M}$</span> is defined via 
<h3>
$$
\begin{align}
\mathbf{M} = \mathbf{Q} \mathbf{U} \mathbf{Q}^{\rm T}
\end{align}
$$
</h3>
<br>
where <span style='font-size:1.25em'>$\mathbf{Q}$</span> is an orthogonal matrix and <span style='font-size:1.25em'>$\mathbf{U}$</span> is a block upper triangular matrix, such that
<h3>$$
\begin{align}
\mathbf{U} \triangleq
\begin{pmatrix}
U^{11} & U^{12} & \cdots & U^{1n} \\
 0 & U^{22} & \cdots & U^{2n} \\
 \vdots & \vdots & \ddots & \vdots \\
 0 & 0 & \cdots & U^{nn}
\end{pmatrix}.
\end{align}
$$
</h3>
<br>
Moreover, the eigenvalues of <span style='font-size:1.25em'>$\mathbf{U}$</span> must equal the eigenvalues of <span style='font-size:1.25em'>$\mathbf{M}$</span>, such that: 
<ol>
<li> each diagonal block <span style='font-size:1.25em'>$U^{ii}$</span> is either a scalar or a $2\times 2$ matrix with complex conjugate eigenvalues, and </li>
<li> the eigenvalues of the diagonal blocks <span style='font-size:1.25em'>$U^{ii}$</span> are ordered descending in magnitude.
</ol>
''']                     


