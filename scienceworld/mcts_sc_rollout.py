import itertools
import numpy as np
from functools import partial
# from models import gpt
from models_custom import gpt
import requests
import logging
import random
import os
import datetime



global reflection_map
global failed_trajectories
reflection_map = []
failed_trajectories = []
def get_value(task, x, y, n_evaluate_sample, logger2, cache_value=True):
    # global reflection_map
    # global failed_trajectories
    
    # unique_trajectories = get_unique_trajectories(failed_trajectories)
    value_prompt = task.value_prompt_wrap(x, y)
    logger2.info(f"Current: {x}")
    logger2.info(f"Current: {y}")
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    logger2.info(f"VALUE PROMPT: {value_prompt}")
    
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    logger2.info(f"VALUE OUTPUTS: {value_outputs}")
    value = task.value_outputs_unwrap(value_outputs)
    logger2.info(f"VALUES: {value}")
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, logger2,cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y
                              , n_evaluate_sample,logger2, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_samples(task, x,logger2, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    # elif prompt_sample == 'cot':
    #     prompt = task.cot_prompt_wrap(x, y, reflection_map)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    # logger2.info(f"PROMPT: {prompt}")
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
  
    return [y + _ for _ in samples]


class Node:
    def __init__(self, state, taskdescription, parent=None):
        self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.taskdescription = taskdescription
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.score= 0
        self.exhausted = False # If all children are terminal

    def uct(self):
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def __str__(self):
        return f"Node(depth={self.depth}, score={self.score}, value={self.value:.2f}, visits={self.visits}, thought={self.state['thought']}, action={self.state['action']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'taskdescription': self.taskdescription,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'score': self.score
        }
    
def node_trajectory_to_text(node_string):
    lines = node_string.split('\n')
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            thought = line.split(", thought=")[1].split(", action=")[0].strip()
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except IndexError:
            continue
        
        if depth != 0:
            if thought:
                formatted_lines.append(f"Thought {depth}: {thought}")
            if action:
                formatted_lines.append(f"Action {depth}: {action}")
            if observation:
                formatted_lines.append(f"Observation {depth}: {observation}")
    
    return '\n'.join(formatted_lines)

def collect_all_nodes(node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(collect_all_nodes(child))
        return nodes

def collect_trajectory(node):
    trajectory = []
    while node:
        trajectory.append(str(node))
        node = node.parent
    return '\n'.join(reversed(trajectory))

def mcts_sc_search(args, env, task, idx, taskName,simplificationStr,logger2,iterations=15):
   
    global gpt
    exitCommands = ["quit", "exit"]
    # gpt=partial(gpt)
 
    env.load(taskName, idx, simplificationStr)
    templates, lut = env.get_possible_action_object_combinations()
    x=env.get_task_description()
    root = Node(state=None, taskdescription=x)
    all_nodes = []
    terminal_nodes=[]

    

    for i in range(iterations):
        logger2.info(f"Iteration {i + 1}...")
        node = select_node(root,logger2)

        while node is None or (node.is_terminal and node.score != 100):
            logger2.info(f"Need to backtrack or terminal node with score 0 found at iteration {i + 1}, reselecting...")
            node = select_node(root,logger2)
            # return terminal_node.state, terminal_node.value, [], terminal_node.reward, terminal_node.score
            
        
        if node is None:
            # logger2.info("All paths lead to terminal nodes with score 0. Ending search.")
            break

        if node.is_terminal and node.score == 100:
            # logger2.info(f"Terminal node with reward 100 found at iteration {i + 1}")
            print(f"Returning node score : {node.score}")
            return node.state, node.value, all_nodes, node.reward, node.score

        expand_node(env,node, args, task, logger2)

        while node.is_terminal or not node.children:
            # logger2.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root, logger2)
            expand_node(env,node, args, task,logger2)

        value = evaluate_node(node, args, task, logger2)
        reward, terminal_node = rollout(env,max(node.children, key=lambda child: child.value), args, task, idx, logger2,max_depth=30)
        terminal_nodes.append(terminal_node)

        
        if terminal_node.score == 100:
            logging.info("SUCCESSFUL TRAJECTORY FOUND DURING SIMULATION")
            return terminal_node.state, terminal_node.value, [], terminal_node.reward, terminal_node.score

    
        
        backpropagate(terminal_node, reward,logger2)
        all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

        terminal_nodes_with_score_100 = [(node, node.score) for node in collect_all_nodes(root) if node.is_terminal and node.score == 100]
        if terminal_nodes_with_score_100:
            logger2.info(f"Terminal node with score 100 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_score_100, key=lambda x: x[1])[0]
            print(f"returning the best score :{best_node.score}")
            return best_node.state, best_node.value, all_nodes, best_node.reward, best_node.score
    
        for j, (node, value) in enumerate(all_nodes):
            pass
            # logger2.info(f"Node {j+1}: {str(node)}")

        # logger2.info(f"State of all_nodes after iteration {i + 1}: {all_nodes}")
  
    # print(all_nodes)
    all_nodes_list = collect_all_nodes(root)
    all_nodes_list.extend(terminal_nodes)

    best_child = max(all_nodes_list, key=lambda x: x.value)
    # best_child = max(all_nodes, key=lambda x: x[1])[0]

    if best_child.score == 100:
        logger2.info("Successful trajectory found")
    else:
        # logger2.info("Unsuccessful trajectory found")
        print(f"Returning best node unsuccesful score : {best_child.score}")
    if best_child is None:
        best_child = root  
    return best_child.state, best_child.value, all_nodes, best_child.reward, best_child.score


def select_node(node,logger2):
    while node and node.children:
        logger2.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
        
        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]
        if len(terminal_children) == len(node.children):
            logger2.info(f"All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            continue  
        
        node_with_score_100 = next((child for child in terminal_children if child.score == 100), None)
        if node_with_score_100:
            logger2.info(f"Found terminal node with score 100 at depth {node.depth}.")
            return node_with_score_100
        
        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

        while node.is_terminal and node.score != 100:
            node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
        logger2.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
        
    return node  # This will return None if all paths from the root are exhausted

def expand_node(env,node, args, task, logger2):
    if node.depth >= 15:
        logger2.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(env,node, args, task, logger2)
    node.children.extend(new_nodes)

def collect_actions(node):
        actions = []
        current = node
        while current.parent is not None:
            actions.append(current.state['action'])
            current = current.parent
        return list(reversed(actions))

def generate_new_states(env,node, args, task, logger2):
    poss_actions=env.get_possible_actions()
    poss_objects=env.get_possible_objects()
    validActions = env.get_valid_action_object_combinations_with_templates()
    action_list = [d['action'] for d in validActions]
    prompt=generate_prompt(node) + f"\n Generate a Thought and an Action where the action should be based on the valid actions : {action_list}\n"
    sampled_actions = get_samples(task, prompt,logger2, f"Thought {node.depth + 1}: ", args["n_generate_sample"], prompt_sample='standard', stop="Observation")
    keywords = ["length", "filter"]

# Check if any of the keywords are in the single string inside the list
    flag = any(keyword in sampled_actions[0] for keyword in keywords)
    if flag:
        prompt = generate_prompt(node) + f"\n Generate a Thought and an Action where the action should be based on the possible actions and possible objects.\nPossible type of actions : {poss_actions} and OBJ can be taken from possible type of objects ; {poss_objects}\n "
        sampled_actions = get_samples(task, prompt,logger2, f"Thought {node.depth + 1}: ", args["n_generate_sample"], prompt_sample='standard', stop="Observation")
        logger2.info(f"SAMPLED ACTION: {sampled_actions}")
    else:
        logger2.info(f"SAMPLED ACTION: {sampled_actions}")
        
    unique_states = {}  # Store unique states here
    node_actions = collect_actions(node)
    
    for action in sampled_actions:
        new_state = node.state.copy()  # Make a copy of the parent node's state
        if node_actions is not None:
            env.reset()
            for i in node_actions:
                env.step(i)

        thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")), '')
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)


        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"
        
        if unique_key in unique_states:
            continue  # Skip if this state already exists
        
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""
            # userInputStr={action_type.lower()}[{action_param}]
           
            userInputStr=action_line
            # print(f"User : {userInputStr}")
            userInputStr = userInputStr.lower().strip()
            obs, r, done, info = env.step(userInputStr)
            logger2.info(f"Reward : {r}")
            # if r>0:
            #     logger2.info(f"Episodic Score : {info['score']}")
            # else:
            logger2.info(f"Episodic Score : {node.score+r}")
            # print(f"Episodic Score : {info['score']}")
            if info['score']==100 or r==-100:
                done=True
            else:
                done=False

       

            # Update the new state dictionary
            new_state['thought'] = thought_line
            new_state['action'] = action_line
            new_state['observation'] = obs

            new_node = Node(state=new_state, taskdescription=node.taskdescription, parent=node)
            new_node.is_terminal = done
            new_node.reward = r
            new_node.score=node.score+r
            # if r>0:
            #     new_node.score=info['score']
            # else:
            #     new_node.score=node.score+r
            # print(new_node.score)
            
            unique_states[unique_key] = new_node  # Add this state to unique_states
            # logger2.info(f"NEW NODE: {new_node}")
            # logger2.info(f"Feedback: {info['score']}")

            # if new_node.is_terminal and r == 0:
            #     trajectory = collect_trajectory(new_node)
            #     failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())  # Return unique nodes as a list

def evaluate_node(node, args, task, logger2):
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.taskdescription , child_prompts, args["n_evaluate_sample"], logger2)
    
    # logger2.info(f"Length of votes: {len(votes)}")
    # logger2.info(f"Length of node.children: {len(node.children)}")
    
    # Pre-allocate votes list
    votes = votes + [0] * (len(node.children) - len(votes))
    
    max_vote = max(votes) if votes else 1
    if max_vote == 0:
        max_vote = 1  # Avoid division by zero
    
    terminal_conditions = [1 if child.is_terminal else 0 for child in node.children]
    for i, condition in enumerate(terminal_conditions):
        if condition == 1:
            votes[i] = max_vote + 1
    
    for i, child in enumerate(node.children):
        child.value = votes[i] / max_vote  # Now safe from division by zero
    
    return sum(votes) / len(votes) if votes else 0

def print_tree(node, level=0):
    indent = "  " * level
    print(f"{indent}{node}")
    for child in node.children:
        print_tree(child, level + 1)


def backpropagate(node, value, logger2):
    while node:
        node.visits += 1
        if node.is_terminal:
            if node.score == 100:
                node.value = (node.value * (node.visits - 1) + value) / node.visits
                logger2.info(f"Backpropagating with score 100 at depth {node.depth}. New value: {node.value}.")
            elif node.score == -100:
                node.value = (node.value * (node.visits - 1) + (-100)) / node.visits
                logger2.info(f"Backpropagating with score 0 at depth {node.depth}. New value: {node.value}.")
        else:
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logger2.info(f"Backpropagating at depth {node.depth}. New value: {node.value}.")

        node = node.parent


def generate_prompt(node):
    trajectory = []
    task_description = node.taskdescription
    while node:
        new_segment = []
        if node.state['thought']:
            new_segment.append(f"Thought {node.depth}: {node.state['thought']}")
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return task_description + '\n'.join(reversed(trajectory))


def rollout(env,node, args, task, idx, logger2,max_depth=15):
    # logger2.info("ROLLING OUT")
    depth = node.depth
    n = 10
    rewards = [0]
    while not node.is_terminal and depth < max_depth:
    # while depth < max_depth:
        # Generate new states
        # logger2.info(f"ROLLING OUT {depth}")
        new_states = []
        values = []
        while len(new_states) == 0:
            new_states = generate_new_states(env,node, args, task, logger2)

        for state in new_states:
            if state.is_terminal:
                return state.score, state
                
        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        #new_state = new_state[0]
        while len(values) == 0:
            values = get_values(task, node.taskdescription, child_prompts, args["n_evaluate_sample"],logger2)
      
        max_value_index = values.index(max(values))
        rewards.append(max(values))
      
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            rewards = [-1]
   
    logger2.info("ROLLOUT FINISHED")
    return sum(rewards) / len(rewards), node