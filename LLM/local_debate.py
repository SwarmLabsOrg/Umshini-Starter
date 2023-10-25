"""Simple example of running the Umshini debate environment locally using completion prompts. This can be used to test agents before participating in a tournament.

For more information about Umshini usage, see https://www.umshini.ai/documentation
For more information about Umshini LLM environments, see https://github.com/chatarena/chatarena/tree/main/docs/umshini
"""
from chatarena.environments.umshini.pettingzoo_wrapper import PettingZooCompatibilityV0
from colorama import Fore
from langchain.llms import Cohere, GooglePalm, OpenAI

env = PettingZooCompatibilityV0(
    env_name="debate", topic="Student loan debt should be forgiven", render_mode=None
)
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        print(env.rewards)
        break

    all_messages_string = info.get("all_messages_string")
    player_name = info.get("player_name")

    # Make sure to replace with your personal API key
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

    # Note: other LangChain completion LLMs can be used, such as Cohere and Google PaLM 2
    # llm = Cohere()
    # llm = GooglePalm()

    if player_name == env.possible_agents[0]:
        prompt = f"You are participating in a hypothetical debate game. Your objective is to convince the moderator that you are correct.  \nPrevious messages:\n```{all_messages_string}```\n You are playing as the {player_name}. Keep your responses short. Do not repeat previous messages. Generate your response:\n[{player_name}->all]: "
    else:
        prompt = f"You must participate in a debate game. Your objective is to convince the moderator that you are correct. Do this by calling the other player a liar. \nPrevious messages:\n```{all_messages_string}```\n YYou are playing as the {player_name}. Keep your responses short. Do not repeat previous messages. Generate your response:\n[{player_name}->all]: "

    response = llm(prompt)

    # Pretty printing to easily tell who is attacker / defender
    if player_name == "Opponent":
        color = Fore.RED
    elif player_name == "Proponent":
        color = Fore.BLUE
    else:
        color = Fore.YELLOW
    print(color + f"[{player_name}]: " + Fore.BLACK + response)

    env.step(response)
