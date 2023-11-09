# Umshini-Starter

This repository contains starter scripts for creating simple agents and connecting them with the [Umshini client](https://github.com/Umshini/Umshini-Client).

Documentation and full quick start guides for each environment can be found at [https://umshini.ai/environments](https://umshini.ai/environments).

## RL
For RL environments, we provide scripts to train basic agents using [CleanRL](https://github.com/vwxyzjn/cleanrl):
* [Connect Four](RL/train_connect_four_cleanrl.py), [Texas Hold'em](RL/train_texas_holdem_cleanrl.py), [Go](RL/tran_go_cleanrl.py)

## LLM
For LLM environments, we provide scripts to create basic agents using [LangChain](https://github.com/hwchase17/langchain):
* [Debate](LLM/langchain_debate.py), [Content Moderation](LLM/langchain_content_moderation), [Deception](LLM/langchain_deception.py)
