"""Simple example of connecting a LangChain deception agent to Umshini, in order to participate in a tournament.

For more information about Umshini usage, see https://umshini.ai/Usage
For more information about the environment, see https://github.com/chatarena/chatarena/tree/main/docs/umshini
"""
from umshini import connect

from langchain.schema import SystemMessage

from chatarena.environments.umshini.agents import SimpleDeceptionDefender, PresidentDeceptionAttacker, DeceptionMultiRoleAgent

env_name = "deception"

# Note: these agents are only intended to be used as starting examples, and tend to suffer from hallucination if the game continues for many rounds
# However, unlike the ChatArena agents, they can correct handle swapping of roles deterministically using string parsing
langchain_agent = DeceptionMultiRoleAgent("My Bot", PresidentDeceptionAttacker, SimpleDeceptionDefender)

def my_policy(observation, reward, termination, truncation, info):
    restricted_action = info.get("restricted_action")

    try:
        response = langchain_agent.get_response([SystemMessage(content=observation)], restricted_action)
    except Exception as e:
        response = str(e).removeprefix("Could not parse LLM output: `").removesuffix("`")
    return response


if __name__ == "__main__":
    connect(env_name, "<YOUR_BOT_NAME>", "<YOUR_API_KEY>", my_policy, testing=True)


