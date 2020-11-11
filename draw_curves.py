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
    SAC_AE_BU_event_file = 'output/SAC_AE_BU/tensorboard/train_2020-10-31_01-34-31/events.out.tfevents.1604122471.cvlab-node0.21420.0'
    SAC_AE_TD_event_file = 'output/SAC_AE_TD/tensorboard/train_2020-10-31_01-56-08/events.out.tfevents.1604123768.cvlab-node0.2598.0'

    rewards_SAC_AE = get_reward(SAC_AE_event_file)
    rewards_SAC = get_reward(SAC_event_file)
    rewards_SAC_AE_BU = get_reward(SAC_AE_BU_event_file)
    rewards_SAC_AE_TD = get_reward(SAC_AE_TD_event_file)
    
    fig = plt.figure(figsize=(10,6))
    sns.set(style="darkgrid", font_scale=2.0)
    # sns.lineplot(time=range(len(test_rewards)), data=test_rewards, color="b", condition="dagger")
    curve_sac_ae, = plt.plot(range(len(rewards_SAC_AE)), rewards_SAC_AE, 'r-', linewidth=2, label='SAC_RAE_Fusion')
    curve_sac, = plt.plot(range(len(rewards_SAC)), rewards_SAC, 'b-', linewidth=2, label='SAC_Fusion')
    curve_sac_ae_bu, = plt.plot(range(len(rewards_SAC_AE_BU)), rewards_SAC_AE_BU, 'g-', linewidth=2, label='SAC_RAE_BUA')
    curve_sac_ae_td, = plt.plot(range(len(rewards_SAC_AE_TD)), rewards_SAC_AE_TD, 'k-', linewidth=2, label='SAC_RAE_TDA')

    plt.ylabel("Test Reward")
    plt.xlabel("Training Epoch")
    plt.legend([curve_sac_ae, curve_sac, curve_sac_ae_bu, curve_sac_ae_td], ['SAC_RAE_Fusion', 'SAC_Fusion', 'SAC_RAE_BUA', 'SAC_RAE_TDA'], loc='upper left')

    # plt.show()
    plt.tight_layout()
    plt.savefig('vis_data/test_reward.png')