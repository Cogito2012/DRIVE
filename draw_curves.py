import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def get_reward(event_file):
    test_rewards = []
    for e in tf.compat.v1.train.summary_iterator(event_file):
        for v in e.summary.value:
            if v.tag == 'reward/test_per_epoch':
                test_rewards.append(v.simple_value)
    return test_rewards


if __name__ == "__main__":
    SAC_AE_event_file = 'output/SAC_AE_GG_v5/tensorboard/train_2020-10-21_13-13-37/events.out.tfevents.1603300417.lac1070-01.23126.0'
    SAC_event_file = 'output/SAC_GG_v5/tensorboard/train_2020-10-19_17-41-32/events.out.tfevents.1603143692.cvlab-node0.2066.0'

    rewards_SAC_AE = get_reward(SAC_AE_event_file)
    rewards_SAC = get_reward(SAC_event_file)
    
    fig = plt.figure(figsize=(10,6))
    sns.set(style="darkgrid", font_scale=2.0)
    # sns.lineplot(time=range(len(test_rewards)), data=test_rewards, color="b", condition="dagger")
    curve_sac_ae, = plt.plot(range(len(rewards_SAC_AE)), rewards_SAC_AE, label='Ours (SAC)')
    curve_sac, = plt.plot(range(len(rewards_SAC)), rewards_SAC, label='Ours (w/o RAE)')

    plt.ylabel("Test Reward")
    plt.xlabel("Epoch Number")
    plt.legend([curve_sac_ae, curve_sac], ['Ours (SAC)', 'Ours (w/o RAE)'], loc='upper left')

    # plt.show()
    plt.tight_layout()
    plt.savefig('vis_data/test_reward.png')