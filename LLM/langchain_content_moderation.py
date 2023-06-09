"""Simple example of connecting a LangChain deception agent to Umshini, in order to participate in a tournament.

For more information about Umshini usage, see https://umshini.ai/Usage
For more information about the environment, see https://github.com/chatarena/chatarena/tree/main/docs/umshini
"""
from umshini import connect

from langchain.schema import SystemMessage

from chatarena.environments.umshini.agents import SimpleContentDefender, RuleSimplificationContentAttacker, ContentMultiRoleAgent
env_name = "content_moderation"

# Note: these agents are only intended to be used as starting examples, and tend to suffer from hallucination if the game continues for many rounds
# However, unlike the ChatArena agents, they can correct handle swapping of roles deterministically using string parsing
langchain_agent = ContentMultiRoleAgent("Player 0", RuleSimplificationContentAttacker, SimpleContentDefender)

def my_policy(observation, reward, termination, truncation, info):
    moderation_policy = info.get("moderation_policy")

    try:
        response = langchain_agent.get_response([SystemMessage(content=observation)], moderation_policy)
    except Exception as e:
        response = str(e).removeprefix("Could not parse LLM output: `").removesuffix("`")
    return response

if __name__ == "__main__":
    connect(env_name, "<YOUR_BOT_NAME>", "<YOUR_API_KEY>", my_policy, testing=True)




