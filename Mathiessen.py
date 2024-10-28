import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import pandas as pd

#Fermi distribution function f
def f(eps,mu,T):
    beta=11600.9/T
    return 1/(np.exp(beta*(eps-mu))+1)
#df/d\epsilon
def df(eps,mu,T):
    beta=11600.9/T
    return beta*np.exp(beta*(eps-mu))/(1+np.exp(beta*(eps-mu)))**2
#Returns density of states assuming a gap \delta such that the dispersion remains linear.
def g(eps, delta):
    if eps>0:
        return eps-delta/2
    else:
        return -(eps+delta/2)
#Calculates the number density for a given chemical potential
def n(mu,T,delta):
    return 1.47*10**(18)*integrate.quad(lambda x: g(x,delta)*(f(x,mu,T)-f(x,-mu,T)),delta/2,delta/2+mu+T/116)[0]
#Calculates the derivative of n with respect to \mu
def dn(mu,T,delta):
    beta=11600.9/T
    return 1.47*10**(18)*integrate.quad(lambda x: g(x,delta)*(df(x,mu,T)+df(x,-mu,T)),delta/2,delta/2+mu+T/116)[0]

'''delta=0.
mulist=np.linspace(-delta/2-0.1,0.1+delta/2,100)
nlist=[n(mu,50,delta) for mu in mulist]
plt.plot(mulist,nlist)'''
'''delta=0.1
mulist=np.linspace(-delta/2,delta/2,50)
nlist=[n(mu,50,delta) for mu in mulist]
plt.scatter(mulist,nlist)
plt.show()'''

delta=0.02

#Solves for \mu given n using the Newton-Rhapson method
def solver(ng,T):
    tol=1e-6
    mutrial=1.5*delta*np.sign(ng)
    ntrial=n(mutrial,T,delta)
    while abs((ng-ntrial)/ng)>tol:
        mutrial=mutrial+(ng-ntrial)/dn(mutrial,T,delta)
        ntrial=n(mutrial,T,delta)
    return mutrial

nlist=np.logspace(13,16,100)
mulist=[]
nchecklist=[]
T=100
for ni in nlist:
    mui=solver(ni,T)
    mulist.append(mui)
    nchecklist.append(n(mui,T,delta))
#Checks that the values of n obtained through the solver match the values calculated through the distribution function
'''plt.plot(nlist,nchecklist)
plt.show()
mulist=np.array(mulist)'''

#eta=0.9997797797797798
eta=0
#Relaxation time assuming a mixture of long-range and short-range scattering.
def tau(eps,eta):
    return abs(eps)/(1+eta*(eps**2-1))
#Calculates the electrical conductivity
def sigma(mu,T,eta,delta):
    return integrate.quad(lambda x: (df(x,mu,T)+df(x,-mu,T))*g(x,delta)*tau(x-delta/2,eta),delta/2+0.0000001,delta/2+abs(mu)+0.5)[0]
#Calculates the thermal conductivity
def kappa(mu,T,eta,delta):
    return integrate.quad(lambda x: ((x-mu)**2*df(x,mu,T)+(x+mu)**2*df(x,-mu,T))*g(x,delta)*tau(x-delta/2,eta)/T,delta/2+0.0000001,delta/2+abs(mu)+0.5)[0]
#Calculates the Lorenz ratio(normalized by the Wiedemann-Franz value)
def L(mu,T,eta,delta):
    return (kappa(mu,T,eta,delta)/(sigma(mu,T,eta,delta)*T))/(np.pi**2/(3*(11600.9)**2))


mulist=np.linspace(-delta/2-0.1001,delta/2+0.1001,200)
nlist=[n(mu,T,delta) for mu in mulist]
klist=[kappa(mu,T,eta,delta) for mu in mulist]
slist=[sigma(mu,T,eta,delta) for mu in mulist]
Llist=[L(mu,T,eta,delta) for mu in mulist]
plt.scatter(nlist,Llist)
plt.show()

np.savetxt("Long-range_Delta=0.02.txt",(mulist,nlist,Llist),delimiter=',')
#Find the minima for L
'''etalist=np.linspace(0.8,1,1000)
Lmin=1
mumin=0
etamin=0.99
for mu in mulist:
    for eta in etalist:
        t=L(mu,T,eta,delta)
        if t<Lmin:
            Lmin=t
            etamin=eta
            mumin=mu

print(Lmin,etamin,mumin)'''
#0.8587528094913409 0.9997797797797798 0.035363535353535355
#0.03751295659702157 0.99 0.7367585585585585 with gap 0.05
'''kappalist=[kappa(mu,T,eta) for mu in mulist]
plt.plot(mulist,kappalist)
plt.show()'''
'''df=pd.DataFrame(nlist)
df.to_csv("nlist.csv", index=False)
df=pd.DataFrame(mulist)
df.to_csv("mulist.csv", index=False)
'''