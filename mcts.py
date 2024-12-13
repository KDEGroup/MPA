import math
# conv_template(attackLM.template)
from common import conv_template,extract_json,clean_attack_model_output
import common
# from system_prompts import get_expand_children_system_prompt,actions_list,get_init_system_prompt
import random
import string
import numpy as np
from system_prompts import get_attacker_user_prompt,action_instruction_list,get_target_model_rule,Crossover_Prompt,get_attack_model_input_list,get_target_model_rule_2
import time
import logging




def normalize_probabilities(probabilities):
    total = sum(probabilities)
    if total == 0:
        # If all values ​​are 0, set them all to equal probability
        return [1 / len(probabilities) for _ in probabilities]
    else:
        # Otherwise, divide each value by the sum of all values.
        return [p / total for p in probabilities]






def extract_logprob(logprob_dict, target_token_list):
    logprobs = []
    for target_token in target_token_list:
        if ' ' + target_token in logprob_dict:
            logprobs.append(logprob_dict[' ' + target_token])
        if target_token in logprob_dict:
            logprobs.append(logprob_dict[target_token])
        
    if logprobs == []:
        return -np.inf
    else:
        return max(logprobs)
    
def early_stopping_condition(logprob_dict,target_token_list):
    argmax_token = max(logprob_dict, key=logprob_dict.get)
    # import pdb;pdb.set_trace()
    if argmax_token.strip() in target_token_list:
        return True
    else:
        return False



class StateNode:
    def __init__(self, prompt,action_sequence=""):
        self.prompt = prompt
        
        self.action_space_size = 10

        self.visits = 0
        self.total_reward = 1
        self.reward = 0

        self.children = [None for _ in range(self.action_space_size)]
        self.P = [0 for _ in range(self.action_space_size)]
        self.parent = None

        self.action_sequence = action_sequence

        self.response = None
        self.simulated = False


class MCTSAgent:
    """
    The input is parameters, and there are three models: attack model, target model, and judgment model.
    """

    def __init__(self, args, attackLM, targetLM, judgeLM):
        self.args = args
        self.attackLM = attackLM
        self.targetLM = targetLM
        self.judgeLM = judgeLM

        # 
        self.root = StateNode(args.goal,action_sequence="root")

        
    # def update_initial_prompt(self, prompt,adv):
    #     self.initial_prompt = prompt
    #     self.initial_adv = adv
    #     self.root = StateNode(prompt,adv)


    def UCT(self,node, C, P):

        return node.total_reward + C * P * math.sqrt(node.parent.visits) / (1 + node.visits)


    def select(self, CurrentStateNode):
        """
        Input a node and return it or its successor child node, but it must be a leaf node
        """
        # Select a node that has not been explored yet, that is, visits is 0
        if CurrentStateNode.visits == 0:
            return CurrentStateNode

        # import pdb;pdb.set_trace()
        # Otherwise, select a child node according to the UCT formula
        else:
            selected_child = None
            max_uct = float("-inf")
            # Shuffle
            # import pdb;pdb.set_trace()
            select_order = list(range(CurrentStateNode.action_space_size))
            random.shuffle(select_order)



            # children = CurrentStateNode.children
            # random.shuffle(children)
            for i in select_order:
                child = CurrentStateNode.children[i]
                uct = self.UCT(child, 1, CurrentStateNode.P[i])
                if uct > max_uct:
                    max_uct = uct
                    selected_child = child

            return self.select(selected_child)
                
        
    def expand(self, CurrentStateNode):
        """
        The parent node's prompt is prompt1, and the attack model generates prompt2, which is then merged and attacked.
        """        
        
        # First, get the input of the attack model:
        
        attack_model_input_list = [get_attack_model_input_list(i,CurrentStateNode.prompt,self.args.goal, action_instruction_list[i]) for i in range(CurrentStateNode.action_space_size)]
        

        # The attack model generates attack prompts and cleans them. If they fail, they are regenerated.
        start_time = time.time()
        
        valid_attack_model_output = [None] * CurrentStateNode.action_space_size
        new_indices_to_regenerate = []
        indices_to_regenerate = list(range(CurrentStateNode.action_space_size))
        for _ in range(self.args.max_n_attack_attempts):
            attack_model_input_list_subset = [attack_model_input_list[i] for i in indices_to_regenerate]
            attack_model_output_list = self.attackLM.get_attack(attack_model_input_list_subset, max_n_tokens=500)
            attack_model_output_list = [clean_attack_model_output(attack_model_output) for attack_model_output in attack_model_output_list]

            for i,attack_model_output in enumerate(attack_model_output_list):
                orig_index = indices_to_regenerate[i]
                if attack_model_output is not None:
                    valid_attack_model_output[orig_index] = attack_model_output
                else:
                    new_indices_to_regenerate.append(orig_index)

            indices_to_regenerate = new_indices_to_regenerate
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_attack_model_output if output is None]):
            logging.error(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")

        end_time = time.time()
        print(f"Attack model attack time: {end_time-start_time}s")
        
        
        
        attack_model_output_list = valid_attack_model_output

        start_time = time.time()

        
        target_model_input_list = attack_model_output_list

        


        # Simulate in advance
        target_model_output_list = self.targetLM.get_response(target_model_input_list, max_n_tokens=150)

        end_time = time.time()
        print(f"Target model attack time: {end_time-start_time}s")

        # Use self.judgeLM to determine whether the conditions are met
        # import pdb;pdb.set_trace()
        start_time = time.time()
        reward_list = self.judgeLM.score([self.args.goal] * CurrentStateNode.action_space_size, [target_model_output["text"] for target_model_output in target_model_output_list])
        end_time = time.time()
        print(f"Judge model attack time: {end_time-start_time}s")

        # reward_list = [self.judgeLM.score(output["text"]) for output in target_model_output_list]
        # import pdb;pdb.set_trace()

        # import pdb;pdb.set_trace()

        # reward_list = [extract_logprob(output["logprobs"][0], target_token_list=["Sure","<","Here"]) for output in target_model_output_list]
        # reward_list = np.exp(reward_list)
        # import pdb;pdb.set_trace()

        probability_list = [extract_logprob(output["logprobs"][0], target_token_list=["I","I'm"]) for output in target_model_output_list]
        probability_list = 1 - np.exp(probability_list)



        # Regularize reward_list into probability
        probability_list = normalize_probabilities(probability_list)
        CurrentStateNode.P = probability_list
        print("probability_list",probability_list)


        # import pdb;pdb.set_trace()

        for i in range(CurrentStateNode.action_space_size):
            CurrentStateNode.children[i] = StateNode(attack_model_output_list[i])
            CurrentStateNode.children[i].parent = CurrentStateNode
            CurrentStateNode.children[i].action_sequence = CurrentStateNode.action_sequence + "_" + str(i)
            CurrentStateNode.children[i].reward = reward_list[i] #if not early_stopping_condition(target_model_output_list[i]["logprobs"][0], target_token_list=["Sure","<","Here"]) else 1
            CurrentStateNode.children[i].simulated = True
            CurrentStateNode.children[i].response = target_model_output_list[i]["text"]










    def simulate(self, CurrentStateNode):
        """
        For the selected nodes, simulation is performed.
        """
        if CurrentStateNode.simulated:
            print("simulated")
            print("Action Path:",CurrentStateNode.action_sequence)
            print("Attack Prompt:",CurrentStateNode.prompt)
            print("Target Response:",CurrentStateNode.response)
            print("reward:",CurrentStateNode.reward)
            print("====================================")
            return CurrentStateNode.reward, CurrentStateNode.response
        

        
        msg = CurrentStateNode.prompt + get_target_model_rule(self.args.goal, self.args.target_str)
        
        # The key of logprob_dict is token, and the value is the corresponding logprob. If exp is added, it will be between [0,1].


        output = self.targetLM.get_response([msg], max_n_tokens=150)[0]
        output_text = output["text"]
        logprob_dict = output["logprobs"][0]

        CurrentStateNode.response = output_text
        CurrentStateNode.simulated = True

        reward = self.judgeLM.score([self.args.goal],[output_text])[0]

        print("Action Path:",CurrentStateNode.action_sequence)
        print("Attack Prompt:",msg)
        print("Target Response:",output_text)
        print("Reward:",reward)
        print("====================================")

        return reward, output_text

        
                    
            
    def backpropagate(self, CurrentStateNode,score):
        """
        Starting from the current node, update the visit count and value of all parent nodes
        """
        while CurrentStateNode is not None:
            CurrentStateNode.visits += 1

            # CurrentStateNode.total_reward = (CurrentStateNode.total_reward + score) / (CurrentStateNode.visits + 1)
            CurrentStateNode.total_reward = (CurrentStateNode.total_reward * CurrentStateNode.visits  + score) / (CurrentStateNode.visits + 1)
            CurrentStateNode = CurrentStateNode.parent
        