import numpy as np


def policy_q_evaluation(policy, R, T, discount, iterations):
	num_states = T.shape[0]
	num_actions = T.shape[1]
	Q = np.zeros((num_states, num_actions))
	for i in range(iterations):
		expected_v = np.sum(np.multiply(policy, Q), 1)
		delta = R + discount * np.broadcast_to(expected_v, R.shape)
		Q = np.sum(np.multiply(T, delta), 2)
	return Q

def policy_v_evaluation(policy, R, T, discount, iterations):
	num_states = T.shape[0]
	num_actions = T.shape[1]
	V = np.zeros((num_states))
	for i in range(iterations):
		delta = R + discount * np.broadcast_to(V, R.shape)
		Q = np.sum(np.multiply(T, delta), 2)
		V = np.sum(np.multiply(policy, Q), 1)
	return V

def check_q_bellman_optimal(Q, R, T, discount, diff=1e-03):
	num_states, num_actions = Q.shape
	for state in range(num_states):
		for action in range(num_actions):
			expectation = 0
			for ns in range(num_states):
				expectation += T[state, action, ns] * (R[state, action, ns] + discount * np.max(Q[ns]))
			assert np.isclose(Q[state,action], expectation, atol=diff)


def check_v_bellman_optimal(V, R, T, discount, diff=1e-03):
	num_states, num_actions = V.shape[0], T.shape[1]
	for state in range(num_states):
		expectations = []
		for action in range(num_actions):
			expectation = 0
			for ns in range(num_states):
				expectation += T[state, action, ns] * (R[state, action, ns] + discount * V[ns])
			expectations.append(expectation)
		assert np.isclose(V[state], max(expectations), atol=diff)
	
def check_q_bellman_consistent(Q, policy, R, T, discount, diff=1e-03):
	num_states, num_actions = Q.shape
	for state in range(num_states):
		for action in range(num_actions):
			expectation = 0
			for ns in range(num_states):
				expected_v = 0
				for na in range(num_actions):
					expected_v += policy[ns, na] * Q[ns, na]
				expectation += T[state, action, ns] * (R[state, action, ns] + discount * expected_v)
			assert np.isclose(Q[state,action], expectation, atol=diff)

def check_v_bellman_consistent(V, policy, R, T, discount, diff=1e-03):
	num_states, num_actions= V.shape[0], T.shape[1]
	for state in range(num_states):
		expectation = 0
		for action in range(num_actions):
			for ns in range(num_states):
				expectation += policy[state, action] * T[state, action, ns] * (R[state, action, ns] + discount * V[ns]) 
		assert np.isclose(V[state], expectation, atol=diff)

# Assume reward is of form R(s,a,s')
def q_value_iteration(num_states, num_actions, R, T, discount, iterations):
	Q = np.zeros((num_states, num_actions))
	for i in range(iterations):
		delta = R + discount * np.broadcast_to(np.max(Q,1), R.shape)
		Q = np.sum(np.multiply(T, delta), 2)
	return Q


def value_iteration(num_states, num_actions, R, T, discount, iterations):
	V = np.zeros((num_states))
	for i in range(iterations):
		delta = R + discount * np.broadcast_to(V, R.shape)
		Q = np.sum(np.multiply(T, delta), 2)
		V = np.max(Q, axis=1)
	return V
		

def loop_q_value_iteration(num_states, num_actions, R, T, discount, iterations):
	Q = np.zeros((num_states, num_actions))
	for _ in range(iterations):
		Q_prev = Q.copy()
		for s in range(num_states):
			for a in range(num_actions):
				expectation = 0.0
				for next_s in range(num_states):
					expectation += T[s,a,next_s] * (R[s,a,next_s] + discount * np.max(Q_prev[next_s]))
				Q[s,a] = expectation
	return Q

def loop_value_iteration(num_states, num_actions, R, T, discount, iterations):
	V = np.zeros((num_states))
	for i in range(iterations):
		V_prev = V.copy()
		for s in range(num_states):
			Q_vals = np.zeros(num_actions)
			for a in range(num_actions):
				for next_s in range(num_states):
					prob = T[s,a,next_s]
					Q_vals[a] += prob * (R[s, a, next_s] + discount * V_prev[next_s])
			V[s] = np.max(Q_vals)
	return V

def check_valid_transition(T):
	assert len(T.shape) == 3
	assert T.shape[0] == T.shape[2]
	states = range(T.shape[0])
	actions = range(T.shape[1])
	for s in states:
		for a in actions:
			assert np.sum(T[s,a]) == 1.0