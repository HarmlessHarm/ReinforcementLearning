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
        V_prev = V.copy()
        V = np.zeros(env.nS)
        
        for s, actions in enumerate(policy):
            for a, p in enumerate(actions):
                sum_reward = 0
                # This only loops once (just nice that it can take uncertain transition probabilities)
                for (p_i, S_1, R, _) in env.P[s][a]:
                    
                    sum_reward += p_i * (R + discount_factor * V_prev[S_1])
                
                V[s] += p * sum_reward
                
        Delta = max(abs(V_prev - V))

        if (Delta < theta):
            print(f"Delta < theta after {k} iterations")
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
    
    policy_stable = False
    
    # While policy not stable
    while (policy_stable == False):
        # Step 2
        V = policy_eval_v(policy, env, discount_factor)
        
        # Step 3
        policy_stable = True
        
        # Duplicate policy to fill in in loop
        new_policy = policy.copy()
        for s, actions in enumerate(policy):

            temp_new_pol = np.zeros(len(actions))
            for a, p in enumerate(actions):
                
                sum_reward = 0
                # This only loops once (just nice that it can take uncertain transition probabilities)
                for (p_i, S_1, R, _) in env.P[s][a]:
                    sum_reward += p_i * (R + discount_factor * V[S_1])
                    
                temp_new_pol[a] = sum_reward
                
            mask = temp_new_pol == max(temp_new_pol)
            
            new_policy[s] = mask / sum(mask)
            
            if ((new_policy[s] != policy[s]).all()):
                policy_stable = False
        
        policy = new_policy.copy()

    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    policy = np.ones((env.nS, env.nA)) / env.nA
    
    K = 1000
    
    # maximum of K times iteration
    for k in range(K):
        Delta = 0
                
        Q_new = np.zeros((env.nS, env.nA))
        
        for s in range(env.nS):
            
            for a in range(env.nA):
                
                sum_reward = 0
                # This only loops once (just nice that it can take uncertain)
                for (p_i, S_1, R, _) in env.P[s][a]:
                    
                    sum_reward += p_i * (R + discount_factor * max(Q[S_1]))
                
                Q_new[s,a] = sum_reward
                
        Delta = np.amax(abs(Q - Q_new))
        Q = Q_new.copy()

        if (Delta < theta):
            print(f"Delta < theta after {k} iterations")
            break
    
    policy = np.asarray([np.array(q == max(q), dtype=int) / sum(q == max(q)) for q in Q])
    
    return policy, Q
