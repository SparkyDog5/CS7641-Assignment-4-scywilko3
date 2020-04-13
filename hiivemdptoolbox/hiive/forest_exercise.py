import mdptoolbox.example
import mdptoolbox.mdp

for state_count in range(2, 12):
    P, R = mdptoolbox.example.forest(S=state_count, r1=4, r2=2, p=0.1, is_sparse=False)

    ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
    # vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
    # pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.96)
    # pi.max_iter = 100000
# expected = (26.244000000000014, 29.484000000000016, 33.484000000000016)
# print(all(expected[k] - pi.V[k] < 1e-12 for k in range(len(expected))))
#
    ql.setVerbose()
    ql.run()
    print(ql.policy)
    # print("Run number: ", state_count)
    # pi.setVerbose()
    # pi.run()
# print(vi.policy)