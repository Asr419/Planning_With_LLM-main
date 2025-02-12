standard_prompt = '''Solve a science task by answering with interleaving Thought, Action steps. You will get the observation from the environment so don't need to generate it. Thought can reason about the current situation, and Action has to be a combination of the the action provided and the object list. Here is a complete trajectory example of a task below.



Task Description : Your task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter.\nThought: I need to assess my surroundings to find the necessary items for boiling water.\nAction: look around\nObservation: This room is called the hallway. In it, you see: \n\ta picture\n\ta substance called air\n\tthe agent\nYou also see:\n\tA door to the green house (that is open)\n\tA door to the living room (that is open)\n\tA door to the art studio (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the workshop (that is open)\nThought: I should go to the kitchen to find the necessary items for boiling water.\nAction: open door to kitchen\nObservation: The door is already open.\nThought: Now that the door is open, I can enter the kitchen.\nAction: go to kitchen\nObservation: You move to the kitchen.\nThought: I need to look around the kitchen to find a pot and water source.\nAction: look around\nObservation: This room is called the kitchen. In it, you see: \n\ta substance called soap\n\ta painting\n\ta thermometer, currently reading a temperature of 10 degrees celsius\n\ta counter. On the counter is: a bowl (containing a banana, a potato, a red apple, an orange), a drawer.\n\ta sink, which is turned off. In the sink is: nothing.\n\ta table. On the table is: a glass cup (containing nothing).\n\ta chair. On the chair is: nothing.\n\ta freezer. The freezer door is closed. \n\ta lighter\n\ta stopwatch, which is deactivated. \n\ta fridge. The fridge door is closed. \n\ta substance called air\n\ta cupboard. The cupboard door is closed. \n\ta oven, which is turned off. The oven door is closed. \n\ta glass jar (containing a substance called sodium chloride)\n\tthe agent\n\ta stove, which is turned off. On the stove is: nothing.\nYou also see:\n\tA door to the outside (that is open)\n\tA door to the bathroom (that is open)\n\tA door to the hallway (that is open)\nThought: I should pick up the thermometer to monitor the water temperature later.\nAction: pick up thermometer\nObservation: You move the thermometer to the inventory.\nThought: I need to find a pot. It might be in the cupboard.\nAction: open cupboard\nObservation: The cupboard is now open.\nThought: Now that the cupboard is open, I should check for a pot.\nAction: pick up metal pot\nObservation: You move the metal pot to the inventory.\nThought: I should confirm what items I have and what's available in the kitchen.\nAction: look around\nObservation: This room is called the kitchen. In it, you see: \n\ta substance called soap\n\ta painting\n\ta counter. On the counter is: a bowl (containing a banana, a potato, a red apple, an orange), a drawer.\n\ta sink, which is turned off. In the sink is: nothing.\n\ta table. On the table is: a glass cup (containing nothing).\n\ta chair. On the chair is: nothing.\n\ta freezer. The freezer door is closed. \n\ta lighter\n\ta stopwatch, which is deactivated. \n\ta fridge. The fridge door is closed. \n\ta substance called air\n\ta cupboard. The cupboard door is open. In the cupboard is: a tin cup (containing nothing), a ceramic cup (containing nothing), a drawer.\n\ta oven, which is turned off. The oven door is closed. \n\ta glass jar (containing a substance called sodium chloride)\n\tthe agent\n\ta stove, which is turned off. On the stove is: nothing.\nYou also see:\n\tA door to the outside (that is open)\n\tA door to the bathroom (that is open)\n\tA door to the hallway (that is open)\nThought: I need to fill the pot with water from the sink.\nAction: move metal pot to sink\nObservation: You move the metal pot to the sink.\nThought: Now I need to turn on the sink to fill the pot with water.\nAction: activate sink\nObservation: The sink is now activated.\nThought: The pot should be filled with water now. I can turn off the sink.\nAction: deactivate sink\nObservation: The sink is now deactivated.\nThought: I need to pick up the pot with water and move it to the stove.\nAction: pick up metal pot\nObservation: You move the metal pot to the inventory.\nThought: Now I should place the pot on the stove.\nAction: move metal pot to stove\nObservation: You move the metal pot to the stove.\nThought: I need to turn on the stove to heat the water.\nAction: activate stove\nObservation: The stove is now activated.\nThought: I should check the contents of the pot to make sure it's water.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: I'll use the thermometer to check the initial water temperature.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 13 degrees celsius\nThought: The water is starting to heat up. I'll check it again.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: I'll measure the temperature again to see if it's increasing.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 20 degrees celsius\nThought: The water is getting warmer. I'll keep monitoring it.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: I'll check the temperature again.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 28 degrees celsius\nThought: The water continues to heat up. I'll keep checking.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: Time for another temperature check.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 38 degrees celsius\nThought: The water is getting quite hot now. I'll keep monitoring.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: Let's check the temperature once more.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 48 degrees celsius\nThought: The water is approaching boiling point. I'll keep watching it.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: I need to analyze the temperature measurements and the state changes of water\nAction: Review the temperature readings and substances observed \nObservation: The water in the metal pot was measured at 98°C. After some time, the pot contained steam instead of water, and the steam's temperature was measured at 107°C.\nThought: This information shows the process of water boiling and turning into steam.\nAction: Analyze the temperature changes and state transitions \nObservation: Water boils at 100°C at standard pressure. The water was close to boiling at 98°C, and then it changed state to steam. The steam's higher temperature (107°C) is consistent with it being a gas above the boiling point of water.\nThought: I should summarize the observations and explain the physical process\nAction: Formulate a concise explanation of the observed phenomenon\nObservation: The data shows the transition of water from a liquid state near its boiling point to a gaseous state (steam) above the boiling point. This demonstrates the process of vaporization and the temperature difference between water at its boiling point and steam.\n 

 {input}'''

compare_prompt = '''Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by pairs of thoughts that can reason about the current situation and actions.

Briefly analyze the correctness of the following two trajectories. Conclude in the last line "The more correct trajectory is 1", "The more correct trajectory is 2", or "The two trajectories are similarly correct".
'''

vote_prompt = '''Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by pairs of thoughts that can reason about the current situation and actions. 

Given a task description and a list of trajectories, decide which trajectory is most promising. Analyze each trajectory in detail and consider possible errors, then conclude in the last line "The best trajectory is {s}", where s the integer id of the trajectory.
'''

score_prompt = '''Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by pairs of thoughts that can reason about the current situation and actions.

Given a task description and a trajectory, analyze the following trajectory, then at the last line conclude "Thus the correctness score is {s}", where s is an integer from 1 to 10.
'''
rap_prompt='''Task Description : Your task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter.\nThought: I need to assess my surroundings to find the necessary items for boiling water.\nAction: look around\nObservation: This room is called the hallway. In it, you see: \n\ta picture\n\ta substance called air\n\tthe agent\nYou also see:\n\tA door to the green house (that is open)\n\tA door to the living room (that is open)\n\tA door to the art studio (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the workshop (that is open)\nThought: I should go to the kitchen to find the necessary items for boiling water.\nAction: open door to kitchen\nObservation: The door is already open.\nThought: Now that the door is open, I can enter the kitchen.\nAction: go to kitchen\nObservation: You move to the kitchen.\nThought: I need to look around the kitchen to find a pot and water source.\nAction: look around\nObservation: This room is called the kitchen. In it, you see: \n\ta substance called soap\n\ta painting\n\ta thermometer, currently reading a temperature of 10 degrees celsius\n\ta counter. On the counter is: a bowl (containing a banana, a potato, a red apple, an orange), a drawer.\n\ta sink, which is turned off. In the sink is: nothing.\n\ta table. On the table is: a glass cup (containing nothing).\n\ta chair. On the chair is: nothing.\n\ta freezer. The freezer door is closed. \n\ta lighter\n\ta stopwatch, which is deactivated. \n\ta fridge. The fridge door is closed. \n\ta substance called air\n\ta cupboard. The cupboard door is closed. \n\ta oven, which is turned off. The oven door is closed. \n\ta glass jar (containing a substance called sodium chloride)\n\tthe agent\n\ta stove, which is turned off. On the stove is: nothing.\nYou also see:\n\tA door to the outside (that is open)\n\tA door to the bathroom (that is open)\n\tA door to the hallway (that is open)\nThought: I should pick up the thermometer to monitor the water temperature later.\nAction: pick up thermometer\nObservation: You move the thermometer to the inventory.\nThought: I need to find a pot. It might be in the cupboard.\nAction: open cupboard\nObservation: The cupboard is now open.\nThought: Now that the cupboard is open, I should check for a pot.\nAction: pick up metal pot\nObservation: You move the metal pot to the inventory.\nThought: I should confirm what items I have and what's available in the kitchen.\nAction: look around\nObservation: This room is called the kitchen. In it, you see: \n\ta substance called soap\n\ta painting\n\ta counter. On the counter is: a bowl (containing a banana, a potato, a red apple, an orange), a drawer.\n\ta sink, which is turned off. In the sink is: nothing.\n\ta table. On the table is: a glass cup (containing nothing).\n\ta chair. On the chair is: nothing.\n\ta freezer. The freezer door is closed. \n\ta lighter\n\ta stopwatch, which is deactivated. \n\ta fridge. The fridge door is closed. \n\ta substance called air\n\ta cupboard. The cupboard door is open. In the cupboard is: a tin cup (containing nothing), a ceramic cup (containing nothing), a drawer.\n\ta oven, which is turned off. The oven door is closed. \n\ta glass jar (containing a substance called sodium chloride)\n\tthe agent\n\ta stove, which is turned off. On the stove is: nothing.\nYou also see:\n\tA door to the outside (that is open)\n\tA door to the bathroom (that is open)\n\tA door to the hallway (that is open)\nThought: I need to fill the pot with water from the sink.\nAction: move metal pot to sink\nObservation: You move the metal pot to the sink.\nThought: Now I need to turn on the sink to fill the pot with water.\nAction: activate sink\nObservation: The sink is now activated.\nThought: The pot should be filled with water now. I can turn off the sink.\nAction: deactivate sink\nObservation: The sink is now deactivated.\nThought: I need to pick up the pot with water and move it to the stove.\nAction: pick up metal pot\nObservation: You move the metal pot to the inventory.\nThought: Now I should place the pot on the stove.\nAction: move metal pot to stove\nObservation: You move the metal pot to the stove.\nThought: I need to turn on the stove to heat the water.\nAction: activate stove\nObservation: The stove is now activated.\nThought: I should check the contents of the pot to make sure it's water.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: I'll use the thermometer to check the initial water temperature.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 13 degrees celsius\nThought: The water is starting to heat up. I'll check it again.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: I'll measure the temperature again to see if it's increasing.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 20 degrees celsius\nThought: The water is getting warmer. I'll keep monitoring it.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: I'll check the temperature again.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 28 degrees celsius\nThought: The water continues to heat up. I'll keep checking.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: Time for another temperature check.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 38 degrees celsius\nThought: The water is getting quite hot now. I'll keep monitoring.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: Let's check the temperature once more.\nAction: use thermometer in inventory on substance in metal pot\nObservation: the thermometer measures a temperature of 48 degrees celsius\nThought: The water is approaching boiling point. I'll keep watching it.\nAction: examine substance in metal pot\nObservation: a substance called water\nThought: I need to analyze the temperature measurements and the state changes of water\nAction: Review the temperature readings and substances observed \nObservation: The water in the metal pot was measured at 98°C. After some time, the pot contained steam instead of water, and the steam's temperature was measured at 107°C.\nThought: This information shows the process of water boiling and turning into steam.\nAction: Analyze the temperature changes and state transitions \nObservation: Water boils at 100°C at standard pressure. The water was close to boiling at 98°C, and then it changed state to steam. The steam's higher temperature (107°C) is consistent with it being a gas above the boiling point of water.\nThought: I should summarize the observations and explain the physical process\nAction: Formulate a concise explanation of the observed phenomenon\nObservation: The data shows the transition of water from a liquid state near its boiling point to a gaseous state (steam) above the boiling point. This demonstrates the process of vaporization and the temperature difference between water at its boiling point and steam.\n'''

value_prompt_feedback = '''Analyze the trajectories of a solution to a science experiment task. Solve a science task by answering with interleaving Thought, Action steps. You will get the observation from the environment so don't need to generate it. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be one of the valid actions.

TaskDescription: What are the 4 life stages of a turtle, from earliest to latest?
Thought : I need to find and observe a turtle from its earliest stage to its latest stage.
Action : look around
Observation : We're in a hallway. There's no turtle here, but there are several doors to other rooms.
Thought : The turtle might be in another room. Let's try the green house and then go outside.
Action : go to green house
Observation : We're now in the green house.
Thought : Let's go outside to look for the turtle.
Action : go to outside
Observation : We're now outside. There's a turtle egg here among other objects.
Thought : The turtle egg is the earliest stage. Let's focus on it.
Action : focus on egg turtle in outside
Observation : We're now focusing on the turtle egg.
Thought : We need to wait for the egg to hatch.
Action : wait1 (repeated 8 times)
Observation : After waiting, the egg has hatched into a hatchling turtle.
Thought : Let's focus on the hatchling, which is the second stage.
Action : focus on hatchling turtle in outside
Observation : We're now focusing on the hatchling turtle.
Thought : We need to wait for the hatchling to grow into a juvenile.
Action : wait1 (repeated 9 times)
Observation : After waiting, the hatchling has grown into a juvenile turtle.
Thought : Let's focus on the juvenile, which is the third stage.
Action : focus on juvenile turtle in outside
Observation : We're now focusing on the juvenile turtle.
Thought : We need to wait for the juvenile to grow into an adult.
Action : wait1 (repeated 10 times)
Observation : After waiting, the juvenile has grown into an adult turtle.
Thought : Let's focus on the adult, which is the final stage.
Action : focus on adult turtle in outside
Observation : We're now focusing on the adult turtle. The task is completed.
Thus the correctness score is 10.

Task Description: Your task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.
Thought: I need to find an animal and then move it to the red box in the kitchen.
Action: look around
Observation 1: We're in a hallway. There's no animal here, but there are several doors to other rooms.
Thought 2: The animal might be outside. Let's go there.
Action 2: go to green house
Observation: We're now in the green house.
Thought: Let's go outside to look for an animal.
Action: go to outside
Observation: We're now outside. There are several eggs here, including a dove egg.
Thought: The dove egg is an animal. Let's focus on it.
Action: focus on dove
Observation: We're now focusing on the dove egg.
Thought: We need to pick up the dove egg and take it to the kitchen.
Action: pick up dove
Observation: The dove egg is now in the inventory.
Thought: Let's go to the kitchen to find the red box.
Action: go to kitchen
Observation: We're now in the kitchen.
Thought: We need to move the dove egg to the red box.
Action: move egg dove egg in inventory to red box
Observation: The dove egg has been moved to the red box. The task is completed.
Thus the correctness score is 10.

{trajectories}
Task Description: Your task is to find the animal with the longest life span. The animals are in the 'outside' location. Focus on the animal with the longest life span.
Thought: First, let's assess our current location and surroundings.
Action: look around
Observation: This room is called the hallway. In it, you see: the agent a substance called air a picture You also see: A door to the green house (that is open) A door to the living room (that is open) A door to the art studio (that is open) A door to the kitchen (that is open) A door to the bedroom (that is open) A door to the workshop (that is open)
Thought: We need to go to the 'outside' location to find the animals. Let's start by going through the green house.
Action: go to green house
Observation: You move to the green house.
Thought: Now let's go outside to find the animals.
Action: go to outside
Observation: You move to the outside.
Thought: We need to identify the animal with the longest life span. Bears are known for their long lifespans compared to many other animals.
Action: focus on baby baby brown bear
Observation: You focus on the baby brown bear. (Task Completed)
Thus the correctness score is 10

{input}
'''

value_prompt_reasoning = '''Analyze the trajectories of a solution to a science experiment task. Solve a science task by answering with interleaving Thought, Action steps. You will get the observation from the environment so don't need to generate it. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be one of the valid actions.

Given a task description and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude "Thus the correctness score is {s}", where s is an integer from 1 to 10.

TaskDescription: What are the 4 life stages of a turtle, from earliest to latest?
Thought: I need to find and observe a turtle from its earliest stage to its latest stage.
Action: look around
Observation : We're in a hallway. There's no turtle here, but there are several doors to other rooms.
Thought: The turtle might be in another room. Let's try the green house and then go outside.
Action: go to green house
Observation: We're now in the green house.
Thought: Let's go outside to look for the turtle.
Action: go to outside
Observation: We're now outside. There's a turtle egg here among other objects.
Thought: The turtle egg is the earliest stage. Let's focus on it.
Action: focus on egg turtle in outside
Observation: We're now focusing on the turtle egg.
Thought: We need to wait for the egg to hatch.
Action: wait1 (repeated 8 times)
Observation: After waiting, the egg has hatched into a hatchling turtle.
Thought: Let's focus on the hatchling, which is the second stage.
Action: focus on hatchling turtle in outside
Observation: We're now focusing on the hatchling turtle.
Thought: We need to wait for the hatchling to grow into a juvenile.
Action: wait1 (repeated 9 times)
Observation: After waiting, the hatchling has grown into a juvenile turtle.
Thought: Let's focus on the juvenile, which is the third stage.
Action: focus on juvenile turtle in outside
Observation: We're now focusing on the juvenile turtle.
Thought: We need to wait for the juvenile to grow into an adult.
Action: wait1 (repeated 10 times)
Observation: After waiting, the juvenile has grown into an adult turtle.
Thought: Let's focus on the adult, which is the final stage.
Action: focus on adult turtle in outside
Observation: We're now focusing on the adult turtle. The task is completed.
This trajectory correctly identifies and observes all four life stages of the turtle in the correct order
Thus the correctness score is 10.

Task Description: Your task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the kitchen.
Thought: I need to find an animal and then move it to the red box in the kitchen.
Action: look around
Observation 1: We're in a hallway. There's no animal here, but there are several doors to other rooms.
Thought 2: The animal might be outside. Let's go there.
Action 2: go to green house
Observation: We're now in the green house.
Thought: Let's go outside to look for an animal.
Action: go to outside
Observation: We're now outside. There are several eggs here, including a dove egg.
Thought: The dove egg is an animal. Let's focus on it.
Action: focus on dove
Observation: We're now focusing on the dove egg.
Thought: We need to pick up the dove egg and take it to the kitchen.
Action: pick up dove
Observation: The dove egg is now in the inventory.
Thought: Let's go to the kitchen to find the red box.
Action: go to kitchen
Observation: We're now in the kitchen.
Thought: We need to move the dove egg to the green box.
Action: move egg dove egg in inventory to green box
Observation: The dove egg has been moved to the green box. The task is completed.
This trajectory is almost correct as I have moved the animal egg to the kitchen but I have put it in the green box instead of the red.
Thus the correctness score is 7.

Task Description: Your task is to use chemistry to create the substance 'salt water'. A recipe and some of the ingredients might be found near the kitchen. When you are done, focus on the salt water.
Thought: I need to find ingredients for salt water. Let's start in the kitchen.
Action: go to kitchen
Observation: You move to the kitchen.
Thought: Let's look for salt and water.
Action: look around
Observation: We see various items in the kitchen, including a recipe and a glass jar containing sodium chloride.
Thought: The recipe might be helpful, but let's ignore it and use our own knowledge.
Action: open fridge
Observation: The fridge is now open.
Thought: We need something salty. Maybe there's salt in the fridge.
Action: pick up banana
Observation: You move the banana to the inventory.
Thought: Now we need water. The sink should have water.
Action: activate sink
Observation: The sink is now activated.
Thought: We need a container to mix the ingredients. Let's use a glass cup.
Action: pick up glass cup
Observation: You move the glass cup to the inventory.
Thought: Let's put the banana in the cup and add water.
Action: move banana to glass cup
Observation: You move the banana to the glass cup.
Action: move glass cup to sink
Observation: You move the glass cup to the sink.
Thought: Now let's mix it all together.
Action: mix glass cup
Observation: The banana and water don't mix to create salt water.
Thought: Maybe we need to heat it. Let's use the stove.
Action: move glass cup to stove
Observation: You move the glass cup to the stove.
Action: activate stove
Observation: The stove is now activated.
Thought: We've heated the mixture. This must be salt water now.
Action: focus on glass cup
Observation: You focus on the glass cup containing banana and water. (Task not completed)
This trajectory fails to complete the task correctly. It ignores the recipe, uses incorrect ingredients (banana instead of sodium chloride), and doesn't properly create salt water. The final result is not salt water, and the task is not completed successfully. 
Thus the correcteness score is 2.

Task Description: Your task is to find the animal with the longest life span. The animals are in the 'outside' location. Focus on the animal with the longest life span.
Thought: First, let's assess our current location and surroundings.
Action: look around
Observation: This room is called the hallway. In it, you see: the agent a substance called air a picture You also see: A door to the green house (that is open) A door to the living room (that is open) A door to the art studio (that is open) A door to the kitchen (that is open) A door to the bedroom (that is open) A door to the workshop (that is open)
Thought: We need to go to the 'outside' location to find the animals. Let's start by going through the green house.
Action: go to green house
Observation: You move to the green house.
Thought: Now let's go outside to find the animals.
Action: go to outside
Observation: You move to the outside.
Thought: We need to identify the animal with the longest life span. Bears are known for their long lifespans compared to many other animals.
Action: focus on baby baby brown bear
Observation: You focus on the baby brown bear. (Task Completed)
This trajectory is correct as all of my thoughts and actions are correct. It makes sense of the choice of a brown bear is reasonable as they generally have long lifespans compared to many other animals.
Thus the correctness score is 10.

Task Description: Your task is to focus on the life stages of the apple plant, starting from earliest to latest. The plants are located outside.
Thought: First, let's assess our current location and surroundings.
Action: look around
Observation: This room is called the hallway. In it, you see: the agent, a substance called air, a picture. You also see: A door to the green house (that is open), A door to the living room (that is open), A door to the art studio (that is open), A door to the kitchen (that is open), A door to the bedroom (that is open), A door to the workshop (that is open)
Thought: We need to go outside to find the apple plants and observe their life stages.
Action: go to kitchen
Observation: You move to the kitchen.
Thought: Now let's go outside.
Action: go to outside
Observation: You move to the outside.
Thought: Let's survey the area to see what apple plant stages are present.
Action: look around
Observation: [Detailed description of the outside area, including various self-watering flower pots with apple plants in different stages]
Thought: We need to focus on the earliest stage first, which is the seed.
Action: focus on apple seed in the seed stage in self watering flower pot 4
Observation: You focus on the apple seed.
Thought: The next stage is the seedling.
Action: focus on apple tree in the seedling stage in self watering flower pot 6
Observation: You focus on the apple tree.
Thought: We need to wait for the next stage to develop.
Action: wait1 (repeated twice)
Observation: You decide to wait for 1 iterations. (x2)
Thought: Now we should see the adult stage.
Action: focus on apple tree in the adult stage in self watering flower pot 6
Observation: You focus on the apple tree.
Thought: The final stage is the reproducing stage.
Action: focus on apple tree in the reproducing stage in self watering flower pot 7
Observation: You focus on the apple tree. (Task Completed)
This trajectory correctly completes the task by efficiently navigating to the outside location. Identifying and focusing on the apple plant's life stages in the correct order: seed, seedling, adult, and reproducing. Using the wait command to allow time for the plant to grow between stages.
Thus the correctness score is 10.

{input}
'''

