"""Static catalogs: 57 Atari games and 20 deep_rl_zoo algorithms.

Both come straight from the upstream codebase. Kept here (rather than imported
from deep_rl_zoo) so the frontend can render even if a deep_rl_zoo import
fails — the API-vs-environment failure modes stay decoupled.
"""
from __future__ import annotations

# 57-game canonical Atari benchmark (env_name as accepted by
# deep_rl_zoo.gym_env.create_atari_environment, plus a 3-letter UI code matching
# the design prototype).
GAMES_57: list[tuple[str, str]] = [
    ("Alien", "ALN"), ("Amidar", "AMI"), ("Assault", "AST"), ("Asterix", "ATX"),
    ("Asteroids", "ASR"), ("Atlantis", "ATL"), ("BankHeist", "BNK"), ("BattleZone", "BTZ"),
    ("BeamRider", "BMR"), ("Berzerk", "BZK"), ("Bowling", "BWL"), ("Boxing", "BOX"),
    ("Breakout", "BKO"), ("Centipede", "CTP"), ("ChopperCommand", "CHP"), ("CrazyClimber", "CCL"),
    ("Defender", "DFD"), ("DemonAttack", "DMA"), ("DoubleDunk", "DBD"), ("Enduro", "END"),
    ("FishingDerby", "FSH"), ("Freeway", "FWY"), ("Frostbite", "FRB"), ("Gopher", "GPH"),
    ("Gravitar", "GRV"), ("Hero", "HRO"), ("IceHockey", "ICE"), ("Jamesbond", "JBN"),
    ("Kangaroo", "KGR"), ("Krull", "KRL"), ("KungFuMaster", "KFU"), ("MontezumaRevenge", "MTZ"),
    ("MsPacman", "PAC"), ("NameThisGame", "NMT"), ("Phoenix", "PHX"), ("Pitfall", "PIT"),
    ("Pong", "PNG"), ("PrivateEye", "PEY"), ("Qbert", "QBR"), ("Riverraid", "RVR"),
    ("RoadRunner", "RDR"), ("Robotank", "RBT"), ("Seaquest", "SEA"), ("Skiing", "SKI"),
    ("Solaris", "SOL"), ("SpaceInvaders", "SPI"), ("StarGunner", "STG"), ("Surround", "SUR"),
    ("Tennis", "TEN"), ("TimePilot", "TPL"), ("Tutankham", "TUT"), ("UpNDown", "UND"),
    ("Venture", "VNT"), ("VideoPinball", "VPN"), ("WizardOfWor", "WIZ"), ("YarsRevenge", "YAR"),
    ("Zaxxon", "ZAX"),
]

# 20 algorithms from deep_rl_zoo, grouped by family. Each tuple is
# (module_name, display_name, family). module_name is the package path under
# deep_rl_zoo/, used to spawn run_atari / eval_agent subprocesses.
ALGORITHMS: list[tuple[str, str, str]] = [
    # Value-based
    ("dqn", "DQN", "value"),
    ("double_dqn", "Double DQN", "value"),
    ("prioritized_dqn", "Prioritized DQN", "value"),
    ("drqn", "DRQN", "value"),
    ("r2d2", "R2D2", "value"),
    ("ngu", "NGU", "value"),
    ("agent57", "Agent57", "value"),
    # Distributional
    ("c51_dqn", "C51 DQN", "distributional"),
    ("rainbow", "Rainbow", "distributional"),
    ("qr_dqn", "QR-DQN", "distributional"),
    ("iqn", "IQN", "distributional"),
    # Policy-based
    ("reinforce", "REINFORCE", "policy"),
    ("reinforce_baseline", "REINFORCE+B", "policy"),
    ("actor_critic", "Actor-Critic", "policy"),
    ("a2c", "A2C", "policy"),
    ("sac", "SAC", "policy"),
    ("ppo", "PPO", "policy"),
    ("ppo_icm", "PPO+ICM", "policy"),
    ("ppo_rnd", "PPO+RND", "policy"),
    ("impala", "IMPALA", "policy"),
]


def family_label(family: str) -> str:
    return {
        "value": "VALUE-BASED",
        "distributional": "DISTRIBUTIONAL",
        "policy": "POLICY-BASED",
    }.get(family, family.upper())
