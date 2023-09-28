#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:16:12 2020

"""
import numpy as np


def computes_discount_function(g_list,T,step):
    '''
    Generates a discount function the gamma nested logit model
    
    Args:
        g_list <- nest dissimilarity values (list of floats)
        T <- number of steps in the discretization (integer)
        step <- discretization accuracy of discount function (float)        
    '''
    # Stores the discount function values
    y_array = np.zeros((len(g_list),T))
    # Stores the discount integral  
    int_y_array = np.zeros((len(g_list),T))
    q = 0
    # Loop computes the discount function + best guarantee
    for g in g_list:
        yes= True
        C = 0.5-step
        delta = 0.01 # accuracy of DP method
        Delta = int(1/delta)+1 # number of states for approximating largest value so far
        int_y = np.zeros(T) # saves integral function with respect to iteration over C
        y = np.zeros(T) # saves discount function with respect to iteration over C
        # Loop to find the maximum constant for which ODE is satisfied
        while (yes == True) and (C<0.8):
            C += step
    
            #Data structures of of DP
            dynamic_prog_val = np.zeros((T,Delta)) # value of integral given state
            dynamic_prog_dec = np.zeros((T,Delta)) # current discount value given state 
    
            #Initialization of DP     
            dynamic_prog_dec[0,:] = np.arange(Delta)*delta
            too_large = int(((1-C)*Delta)+1)
            dynamic_prog_dec[0,too_large:] = -1
            dynamic_prog_val[0,:] = (dynamic_prog_dec[0,:]+1e-15)*(1.0/T)
            dynamic_prog_val[0,too_large:] = -1
            
            for t in range(1,T):
                #First we compute for each integral value what is the corresponding 
                #maximal value that can be saved
                maximal_delta = np.zeros(Delta)
                for d in range(Delta):
                    int_y_current = dynamic_prog_val[t-1,d]
                    if int_y_current > -1:
                        # Lower bound on alpha*
                        lb = min(0.99999,int_y_current/C)
                        # Upper bound on alpha*           
                        ub = 0.99999
                        # Gap between bounds 
                        gap = 1
                        # Current x
                        x = float(t)/T 
                        min_val = 0
                        # Loop approximates the minimization with respect to alpha
                        while (gap > 0.0001) and (lb<0.99998) \
                            and (C*ub > int_y_current) and (ub>0.00002) \
                            and (min_val <= 1):
                            tb = (lb + ub)/2
                            rhs = C*tb - int_y_current
                            theta = tb/x*(1-x)/(1-tb)  
                            if theta < 0.0001:
                                lhs = C*g*np.power(theta, 1/g)
                            else:
                                lhs = C*(np.power(1+np.power(theta, 1/g),g)-1)
                            lhs = lhs*(1-tb)*(1-tb)*x/(1-x)/\
                                np.power(1- 1/(1+np.power(theta,1/g)),1-g)                
                            
                            theta1 = lb/x*(1-x)/(1-lb)
                            theta2 = ub/x*(1-x)/(1-ub)
                            if theta1 < 0.0001:
                                deta1 = x*g*np.power(theta1, 1/g)
                            else:    
                                deta1 = x*(np.power(1+np.power(theta1, 1/g),g)-1)
                            if theta2 < 0.0001:
                                deta2 = x*g*np.power(theta2, 1/g)
                            else:    
                                deta2 = x*(np.power(1+np.power(theta2, 1/g),g)-1)                            
                            max_val = (C*ub - int_y_current)/(1-x)/deta1
                            min_val = (C*lb - int_y_current)/(1-x)/deta2
                            #first order condition for the optimality of alpha*
                            gap = max_val - min_val
                            if rhs > lhs:
                                ub = tb
                            else:
                                lb = tb
                        rhs = C*ub - int_y_current
                        theta = lb/x*(1-x)/(1-lb)
                        if theta < 0.0001:
                            deta = g*np.power(theta, 1/g)
                        else:                    
                            deta = (np.power(1+np.power(theta, 1/g),g)-1)
                        if deta > 0:
                            current_max = int(max(-1,min(1,1-rhs/x/(1-x)/deta))*Delta)
                        else: 
                            current_max = -1
                        maximal_delta[d] = current_max
                    else:
                        maximal_delta[d] = -1
                
                for d in range(Delta):
                    # feasibility means that we don't exceed the maximum saving
                    feasible_states = np.where(maximal_delta>= d)[0]
                    # feasibility means that the function is monotone
                    feasible_states = feasible_states[feasible_states<=d]               
                    if feasible_states.shape[0]> 0:
                        # pick the maximum we can save                   
                        dynamic_prog_val[t,d] = np.max(dynamic_prog_val[t-1,\
                                                             feasible_states]) \
                                                + d*delta/T                                         
                        dynamic_prog_dec[t,d] = feasible_states[np.argmax(\
                                            dynamic_prog_val[t-1,feasible_states])]                 
                    else:
                        # by assigning -1, we kill this branch of the DP                    
                        dynamic_prog_val[t,d] = -1
                        dynamic_prog_dec[t,d] = -1                    
            if np.max(np.minimum(dynamic_prog_val[T-1,:] > 0,dynamic_prog_dec[T-1,:] > 0)):
                #Ultimately we should have not encountered any -1 for C to be feasible
                #It is preferable to save the least amount while achieving the target            
                selection = np.minimum(dynamic_prog_val[T-1,:] > 0,dynamic_prog_dec[T-1,:] > 0)
                saves_less = np.argmin(dynamic_prog_val[T-1,selection])
                saves_less = np.where(selection)[0][saves_less]
                y[T-1] = saves_less*delta
                int_y[T-1] = dynamic_prog_val[T-1,saves_less]
                for t in range(T-2,-1,-1):
                    y[t] = dynamic_prog_dec[t+1,int(np.round(y[t+1]/delta))]*delta
                    int_y[t] = dynamic_prog_val[t,int(np.round(y[t]/delta))]
            else:
                yes = False                
                C = C - step
                
        y_array[q,:] = y
        int_y_array[q,:] = int_y
        print(g,C)
        q+=1
    return(y_array,int_y_array)


def computes_approximation_guarantee(g_list,y,alpha_step,step,T):
    '''
    Computes the competitive ratio associated with a certain discount function
    
    Args:
        g_list <- nest dissimilarity values (list of floats)    
        y <- discount function (array float)
        alpha_step  <- discretization of alpha (float)
        step <- discretization of the performance guarantee (float)
        T <- number of elements in the discount function array (integer)   
    '''
    alpha_T = int(1/alpha_step)    
    perf = []
    for g in g_list:
        yes= True
        C = 0.5-step    
        int_y = np.cumsum(y)/T
        while (yes == True) and (C<1.0):
            C += step
            for t in range(1,T-1):
                x = float(t)/T 
                for alpha in [alpha_step*i for i in range(1,alpha_T)]:                
                    val = min(1,1-(C - int_y[t-1]/alpha)/x/(1-x)*alpha\
                              /(np.power(1+np.power(alpha/x*(1-x)/(1-alpha), 1/g),g)-1))
                    if val < y[t]:
                        yes = False
        print(g,C)
        perf.append((g,C))
    return(perf)