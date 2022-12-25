from gym.envs.registration import register

register(
    id='Gobang9x9-v0',
    entry_point='gobang_game.envs:GobangEnv',
    kwargs={
        'board_size':9,
        'n_feature_planes':6,
    }
)

register(
    id='Gobang9x9-v1',
    entry_point='gobang_game.envs:SelfPlayGobangEnv',
    kwargs={
        'board_size':9,
        'n_feature_planes':6,
        'render_mode':True,
        'is_self_play':False
    }
)

register(
    id='SPGobang9x9-v1',
    entry_point='gobang_game.envs:SelfPlayGobangEnv',
    kwargs={
        'board_size':9,
        'n_feature_planes':6,
        'render_mode':False,
        'is_self_play':True
    }
)

register(
    id='MCTSGobang9x9-v1',
    entry_point='gobang_game.envs:MCTSGobangEnv',
    kwargs={
        'board_size':9,
        'n_feature_planes':6,
        'render_mode':False,
        'is_self_play':True,
        'c_puct': 4,
        'n_iters': 1200,
    }
)
