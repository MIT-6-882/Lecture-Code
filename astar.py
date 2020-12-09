"""DO NOT COMMIT THIS!!!
"""
from collections import namedtuple, defaultdict
from itertools import count
import heapq as hq


class AStar:
    """Planning with A* search
    """
    
    Node = namedtuple("Node", ["state", "parent", "action", "g"])

    def __init__(self, successor_fn, check_goal_fn, cost_fn, timeout=1000):
        self._get_successor_state = successor_fn
        self._check_goal = check_goal_fn
        self._get_cost = cost_fn
        self._heuristic = None
        self._timeout = timeout
        self._actions = None
        
    def __call__(self, state, heuristic=None, verbose=True):
        self._heuristic = heuristic or (lambda node : 0)
        return self._get_plan(state, verbose=verbose)

    def set_actions(self, actions):
        self._actions = actions

    def _get_plan(self, state, verbose=True):
        start_time = time.time()
        queue = []
        state_to_best_g = defaultdict(lambda : float("inf"))
        tiebreak = count()

        root_node = self.Node(state=state, parent=None, action=None, g=0)
        hq.heappush(queue, (self._get_priority(root_node), next(tiebreak), root_node))
        num_expansions = 0

        while len(queue) > 0 and (time.time() - start_time < self._timeout):
            _, _, node = hq.heappop(queue)
            # If we already found a better path here, don't bother
            if state_to_best_g[node.state] < node.g:
                continue
            # If the goal holds, return
            if self._check_goal(node.state):
                if verbose:
                    print("\nPlan found!")
                return self._finish_plan(node), {'node_expansions' : num_expansions}
            num_expansions += 1
            if verbose:
                print(f"Expanding node {num_expansions}", end='\r', flush=True)
            # Generate successors
            for cost, action, child_state in self._get_successors(node.state):
                # If we already found a better path to child, don't bother
                if state_to_best_g[child_state] <= node.g+1:
                    continue
                # Add new node
                child_node = self.Node(state=child_state, parent=node, action=action, g=node.g+cost)
                priority = self._get_priority(child_node)
                hq.heappush(queue, (priority, next(tiebreak), child_node))
                state_to_best_g[child_state] = child_node.g

        if verbose:
            print("Warning: planning failed.")
        return [], {'node_expansions' : num_expansions}
    
    def _get_successors(self, state):
        for action in self._actions:
            next_state = self._get_successor_state(state, action)
            cost = self._get_cost(state, action)
            yield cost, action, next_state

    def _finish_plan(self, node):
        plan = []
        while node.parent is not None:
            plan.append(node.action)
            node = node.parent
        plan.reverse()
        return plan

    def _get_priority(self, node):
        h = self._heuristic(node)
        if isinstance(h, tuple):
            return (tuple(node.g + hi for hi in h), h)
        return (node.g + h, h)

def report_table():
    columns = ["Approach", "Duration (s)", "# Env Steps Taken", "Returns"]

    with open("astar_results.p", "rb") as f:
        table = pickle.load(f)

    with open("uct_results.p", "rb") as f:
        uct_table = pickle.load(f)

    table += uct_table
    print(tabulate(table, headers=columns))


if __name__ == "__main__":
    import time
    import pickle
    from tabulate import tabulate
    from envs.water_delivery import WaterDeliveryEnv

    approaches = []
    durations = []
    env_steps_taken = []
    all_returns = []

    max_num_steps = 41
    mode = 'medium'
    num_alias_per_action = 10

    approaches.append('A*')

    start_time = time.time()

    env = WaterDeliveryEnv(mode=mode, num_alias_per_action=num_alias_per_action)
    Rmax = env.MAX_REWARD

    def successor_fn(wrapped_state, action):
        state, t = wrapped_state
        return (env.compute_transition(state, action), t+1)

    def cost_fn(wrapped_state, action):
        state, t = wrapped_state
        reward = env.compute_reward(state, action)
        return Rmax - reward

    def check_goal_fn(wrapped_state):
        state, t = wrapped_state
        return env.check_goal(state) or t == max_num_steps

    astar = AStar(successor_fn, check_goal_fn, cost_fn)
    astar.set_actions(env.get_all_actions())

    state, _ = env.reset()
    returns = 0.

    print("Running AStar...")
    plan, _ = astar((state, 0))
    print("Done.")

    print("Initial state:", env.state_to_str(state))
    for t in range(min(max_num_steps, len(plan))):
        action = plan.pop(0)
        print("Taking action", action)
        state, reward, done, _ = env.step(action)
        returns += reward
        print("State:", env.state_to_str(state))
        print("Reward, Done:", reward, done)
        if done:
            break

    duration = time.time() - start_time
    durations.append(duration)
    env_steps_taken.append(t)
    all_returns.append(returns)

    table = list(zip(approaches, durations, env_steps_taken, all_returns))
    with open("astar_results.p", "wb") as f:
        pickle.dump(table, f)

    report_table()
