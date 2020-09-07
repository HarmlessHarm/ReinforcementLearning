import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        discount_factor: Gamma discount factor.
        theta: We stop evaluation once our value function change is less than theta for all states.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    K = 1000
    
    for k in range(K):
        Delta = 0
        V_prev = V
        V = np.zeros(env.nS)
        
        for s, actions in enumerate(policy):
            
            for a, p in enumerate(actions):
                
                sum_reward = 0
                for i in range(len(env.P[s][a])):  
                    
                    tup = env.P[s][a][i]
                    p_i = tup[0]
                    S_1 = tup[1]
                    R = tup[2]
                    
                    sum_reward += p_i * (R + V_prev[S_1])
                
                V[s] += p * sum_reward
                
        Delta = max(abs(V_prev - V))
                                 
        if (Delta < theta):
            print("Delta smaller:", k)
            break
    
    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    raise NotImplementedError
    return policy, V
