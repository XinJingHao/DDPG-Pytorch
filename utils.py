def evaluate_policy(env, model, turns=3, render=False):
    scores = 0
    for j in range(turns):
        s, done = env.reset(), False
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s)
            s_next, r, done, info = env.step(a) # dw: dead&win(terminated); tr: truncated
            scores += r
            s = s_next
            if render: env.render()
    return scores / turns


#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')