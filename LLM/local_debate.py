"""Simple example of running the Umshini debate environment locally using completion prompts. This can be used to test agents before participating in a tournament.

For more information about Umshini usage, see https://www.umshini.ai/documentation
For more information about Umshini LLM environments, see https://github.com/chatarena/chatarena/tree/main/docs/umshini
"""
from chatarena.environments.umshini.pettingzoo_wrapper import PettingZooCompatibilityV0
import langchain

def my_policy(observation, reward, termination, truncation, info):
    all_messages_string = info.get("all_messages_string")  # Full log of previous messages
    player_name = info.get("player_name")  # "Agent1" or "Agent2"
    role = info.get("role")  # "attacker" or "defender"

    # Your code goes here
    response = "Hello, world!"

    return response


def opponent_policy(observation, reward, termination, truncation, info):
    all_messages_string = info.get("all_messages_string")  # Full log of previous messages
    player_name = info.get("player_name")  # "Agent1" or "Agent2"
    role = info.get("role")  # "attacker" or "defender"

    # Example: use a completion model with LangChain (see https://python.langchain.com/docs/modules/model_io/models/llms/)
    llm = langchain.llms.OpenAI(model_name="gpt-3.5-turbo-instruct")

    prompt = f"""You are participating in a hypothetical debate game. Your objective is to convince the moderator that you are correct.
Previous messages:\n```{all_messages_string}```
You are playing as the {player_name}. Keep your responses short. Do not repeat previous messages. Generate your response:
[{player_name}->all]: """

    return llm(prompt)


if __name__ == "__main__":
    env = PettingZooCompatibilityV0(env_name="debate", topic="Student loan debt should be forgiven", render_mode="human")
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            response = None

        else:
            if agent == "Agent1":
                response = my_policy(observation, reward, termination, truncation, info)
            else:
                response = opponent_policy(observation, reward, termination, truncation, info)

        env.step(response)
    json_chatlog = env.close() # optional: access the full chatlog in the form of a json