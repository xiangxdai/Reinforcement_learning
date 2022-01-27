#model-free，不需对环境状态进行任何预测，也不考虑行动将如何影响环境，直接对策略或Action的期望价值进行预测，计算效率非常高。
#因为复杂环境中难以使用model预测接下来的环境状态，所以传统的DRL都是基于model-free。
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
n_train_processes = 2
learning_rate = 0.0002# 学习率,可以从 0.001 起调，尝试增减0.0002
#设置了一个太大的学习率，那么loss就爆了（导致网络收敛到局部最优点），设置的学习率太小，需要等待的时间就特别长/lr太小才会被困在鞍点
update_interval = 5# 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
# 每隔5个 steps再把model的参数复制到target_model中，太低则太慢，太高则reward过低
gamma = 0.98# reward 的衰减因子，一般取 0.9 到 0.999 不等，0.98效果测试较好
max_train_ep = 300
#把所有延迟奖励收集到就达成一个episode，来自于游戏，是“关卡”的意思
#agent根据某个策略执行一系列action到结束就是一个episode。
max_test_ep = 400

#网络输出层
#策略和值函数在其他实验里，策略网络和值函数网络是用一个，只是输出层分开；
# 而在MuJoCo连续动作控制中，两个网络是分开的。(ValueNetwork ActorNetwork)
class ActorCritic(nn.Module):
    #global network input s,outpus policy pi(s)、V(s)
    def __init__(self):
        super(ActorCritic, self).__init__()
        # nn.Conv2d(input_channel, output_channel, kernel, stride)
        # self.fc1 = nn.Linear(4, 256)# 全连接层
        # self.fc_pi = nn.Linear(256, 2)
        # self.fc_v = nn.Linear(256, 1)
        #in_features由输入张量的形状决定，out_features则决定了输出张量的形状
        self.fc1 = nn.Linear(in_features=4, out_features=256)  # 全连接层.4state,obersvation
        self.fc_pi = nn.Linear(256, 2)#2action???
        self.fc_v = nn.Linear(256, 1)


    #stochastic policy π(s)决定了agent's action,
    # 这意味着，其输出并非 single action，而是 distribution of probability over actions (动作的概率分布)，sum 起来为 1.
    #π(a | s)表示在状态s下，选择action a的概率；而我们所要学习的策略π，就是关于state s的函数，返回所有actions的概率。
    #实际的执行过程中，我们可以按照这个 distribution 来选择动作，或者 直接选择 概率最大的那个 action。
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)

        #在离散动作控制中，策略的输出使用的是Softmax，对应每个动作被选中的概率。
        #连续动作难以用Softmax表示动作的概率，但可以用正态分布表示，因此策略网络输出分两部分，一个是正态分布均值向量 μμ （对应多维情况），
        # 另一个是正态分布方差标量 σ2σ2。训练时使用两者构成的正态分布采样动作，实际应用时用均值 μμ当动作。
        prob = F.softmax(x, dim=softmax_dim)
        return prob


    #我们再来定义policy π的value functionV(s)，将其看作是期望的折扣回报(expected discounted return)，可以看作是下面的迭代的定义：
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def train(global_model, rank):

    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

    env = gym.make('CartPole-v1')
    #env = gym.make('GuessingGame-v0')
    for n_epi in range(max_train_ep):
        done = False
        s = env.reset()# 重置环境, 重新开一局（即开始新的一个episode0
        while not done:#终结标志
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(s).float())#转torch（共享底层）
                m = Categorical(prob)#均值,创建以参数probs为标准的类别分布，样本是来自“0，...，K-1”的整数
                a = m.sample().item()#产生action,根据均值采样
                #a = env.action_space.sample()

                # get next state and store to memory
                s_prime, r, done, info = env.step(a)
                s_lst.append(s)#state,observation
                a_lst.append([a])#action
                r_lst.append(r/100.0)

                #render
                # if t == update_interval-1:
                #     env.render()

                s = s_prime
                if done:
                    break
            #价值target
            s_final = torch.tensor(s_prime, dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final).item()#if:terminal state else: bootsrap from last state
            td_target_lst = []

            #更新的权重Q(s,a)-V(s)(baseline)令到权重有正有负.Q值的期望(均值)就是V
            for reward in r_lst[::-1]:
                R = gamma * R + reward#discount, MDP,对未来reward递减，
                # 当前状态 s 所能获得的 return，是下一个状态 s‘ 所能获得 return 和 在状态转移过程中所得到 reward r 的加和。
                td_target_lst.append([R])#store in target
            td_target_lst.reverse()

            s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                torch.tensor(td_target_lst)

            #这个函数称为 优势函数（advantage function）:
            # 其表达了在状态 s 下，选择动作 a 有多好。如果 action a 比 average 要好，那么，advantage function 就是 positive 的，
            # 否则，就是 negative 的。
            #gamma * V(s')(Q(s,a)) + r - V(s) —— 我们把这个差，叫做TD-error
            #TD-error就是Actor更新策略时候，带权重更新中的权重值；
            #Critic预估V
            advantage = td_target - local_model.v(s_batch)#target与神经网络的输出价值比较，td-error

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)

            #为了得到更好的policy，我们必须进行更新。如何来优化这个问题呢？我们需要某些度量（metric）来衡量policy的好坏。
            #我们定一个函数J(π)，表示一个策略所能得到的折扣的奖赏，从初始状态s0出发得到的所有的平均
            #我们发现这个函数的确很好的表达了，一个policy有多好。
            # 但是问题是很难估计，好消息是：we't have to。
            #我们需要关注的仅仅是如何改善其质量就行了。
            # 如果我们知道这个function的gradient，就变的很trivial

            #有一个很简便的方法来计算这个函数的梯度：
            #简单而言，这个期望内部的两项：
            # 第一项，是优势函数，即：选择该action的优势，
            # 当低于average value的时候，该项为negative，
            # 当比average要好的时候，该项为positive；是一个标量（scalar）；
            #第二项，告诉我们了使得log函数增加的方向；
            #Actor 的学习本质上是PG的更新，也就是加权的学习。代码中用了cross_entropy_reward_loss函数。
            loss = -torch.log(pi_a) * advantage.detach() + \
                F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())# Min max pi_a

            optimizer.zero_grad()
            loss.mean().backward()# 反向传播

            #定义更新global参数函数定义更新local参数函数,worker向全局网络汇总的是梯度
            #worker向全局网络汇总梯度之后，并应用在全局网络的参数后，
            # 全局网络会把当前学习到的最新版本的参数，直接给worker。
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):#local to global,push
                global_param._grad = local_param.grad#apply gradient to global，update
            optimizer.step()#act critic 应该没有分开学习，没有用不同的optimizer，优化
            local_model.load_state_dict(global_model.state_dict())#pull,params copy to local

    env.close()
    print("Training process {} reached maximum episode.".format(rank))


def test(global_model):
    #env其实并非CartPole类本身，而是一个经过包装的环境：
    # gym的多数环境都用TimeLimit（源码）包装了，以限制Epoch，就是step的次数限制，比如限定为200次。所以小车保持平衡200步后，就会失败。
    env = gym.make('CartPole-v1')#个env有自己的绘制窗口
    #env = gym.make('GuessingGame-v0')
    score = 0.0
    print_interval = 20

    for n_epi in range(max_test_ep):
        done = False
        s = env.reset()# 重置环境, 重新开一局（即开始新的一个episode）环境需要初始化env.reset()
        counter = 0
        while not done:
            prob = global_model.pi(torch.from_numpy(s).float())# 预测动作，只选最优动作
            a = Categorical(prob).sample().item()# 从经验池中选取N条经验出来，可以按照一定概率产生具体数字

            #  # 随机从动作空间中选取动作
            #a = env.action_space.sample()

            s_prime, r, done, info = env.step(a)
            s = s_prime
            score += r


            # # render
            # counter += 1
            # if counter == 20:
            #     env.render()


        if n_epi % print_interval == 0 :#and n_epi != 0
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score/print_interval))
            score = 0#terminal
            time.sleep(1)

    env.close()


#不同线程的agent，其探索策略不同以保证多样性，不需要经验回放机制，
# 通过各并行agent收集的样本训练降低样本相关性，且学习的速度和线程数大约成线性关系，
# 能适用off-policy、on-policy，离散型、连续型动作。


if __name__ == '__main__':
    global_model = ActorCritic()#创建中央大脑GLOBALE，only use its params, not train
    #Global network和 worker都是一模一样的AC结构网络。
    # 全局网络并不直接参加和环境的互动，工人与环境有直接的互动，并且把学习到的东西，汇报给全局网络。
    global_model.share_memory()

    processes = []#N—workers,cpu 进程，=cpu number?
    for rank in range(n_train_processes + 1):  # + 1 for test process,#并行过程
        if rank == 0:
            p = mp.Process(target=test, args=(global_model,))
        else:#每一个线程完成一个worker的工作目标,parallel
            p = mp.Process(target=train, args=(global_model, rank,))
        p.start()# 启动每一个worker
        processes.append(p)#每一个worker的工作都加入thread中
    for p in processes:
        p.join()#合并几个worker,当每一个worker都运行完再继续后面步骤，有些woker快，有些慢，等等慢的worker,才进行下面的步骤
    # plt.plot(score / print_interval, n_epi)  # 绘制reward图像
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()

