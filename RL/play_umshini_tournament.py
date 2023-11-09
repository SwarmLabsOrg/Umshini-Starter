"""Simple example of using trained RL agents to Umshini and connecting to a live tournament.

For more information about Umshini usage, see https://www.umshini.ai/documentation
For more information about Umshini RL environments, see https://www.umshini.ai/environments
"""
import torch
import umshini
from pettingzoo.classic import connect_four_v3
from torch import nn

checkpoint_path = "<YOUR_CHECKPOINT_PATH>"
env_name = "connect_four_v3"


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(84, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space("player_0").n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


model = QNetwork(connect_four_v3.env())
model.load_state_dict(torch.load(checkpoint_path))


def my_pol(obs, rew, term, trunc, info):
    if term or trunc:
        action = None
    else:
        obs_in = torch.unsqueeze(torch.Tensor(obs["observation"]), 0)
        q_values = model(obs_in)
        action_mask = torch.Tensor((obs["action_mask"] - 1) * 100)
        q_values = q_values + action_mask
        action = torch.squeeze(torch.argmax(q_values, dim=1)).cpu().numpy()
    return action


if __name__ == "__main__":
    umshini.connect(env_name, "<YOUR_BOT_NAME>", "<YOUR_API_KEY>", my_pol, testing=True)
