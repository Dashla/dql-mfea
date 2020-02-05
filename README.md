# Simultaneously Evolving Deep Reinforcement Learning Models via Multifactorial Optimization 
## Abstract
In the recent years, Multifactorial Optimization (MFO) has attracted a lot of interest in the optimization community. MFO is known for its inherent skills to address multiple complex optimization tasks at the same time, while inter-task information transfer is used to improve their convergence speed. These skills make Multifactorial Evolution appealing to be applied to evolve Deep Reinforcement Learning (DQL) models, which is the scenario tackled in this paper. Complex DQL models usually find difficult to converge to optimal solutions, due to the lack of exploration or sparse rewards. In order to overcome these drawbacks, pre-trained models are commonly used to make Transfer Learning, transferring knowledge from the pre-trained to the target domain. Besides, it has been shown that the lack of exploration can be reduced by using meta-heuristic optimization approaches. In this paper we aim to explore the use of the MFO framework to optimize DQL models, making an analysis between MFO and the traditional Transfer Learning and metaheuristic approaches in terms of convergence, speed and policy quality.

Source code of MFEA used in the paper: [link](https://github.com/HuangLingYu96/MFEA)
## Codification
<div align="center">
<img src="/uploads/19a99a34f98d7d915f6b7082ef83ca5a/Codification.png"  width="500" height="300" title="Codification schema">
</div>


## Experimentation

In this work we experiment with MFEA and it is used to train/evolve DQL networks. The environments used are: [*Cartpole*](https://gym.openai.com/envs/CartPole-v0/), [*Acrobot*](https://gym.openai.com/envs/Acrobot-v1/) and [*Pendulum*](https://gym.openai.com/envs/Pendulum-v0/). First the performance of MFEA on this scenarios is analyzed:


| Evolution | Model Performance |
:-------------------------:|:-------------------------:
<img src="/uploads/025f302667cbb45f300790cd1b6486db/1Cartpole.png"  width="550" height="300" title="Cartpole"/> | <img src="/uploads/9d0fc2d90b4fbf5cd175d885820de73b/cartpole.gif"  width="350" height="250" title="Cartpole"/>
<img src="/uploads/fad775bbbd469e5a7efed2bd30b1caf2/1Acrobot.png"  width="550" height="300" title="Acrobot"/> | <img src="/uploads/7e6d535ce7000ed1929309b7b9fd3a50/acrobot.gif"  width="250" height="250" title="Acrobot"/>
<img src="/uploads/64a25495d7dded22b96ec959963a7620/1Pendulum.png"  width="550" height="300" title="Pendulum"> | <img src="/uploads/2f5ee3bf4abd7acac2a6a863f50b1b89/pendulum.gif"  width="250" height="250" title="Pendulum">



We observe quick convergence times and good results, as the *Pendulum* environment is more complex the convergence is also slower

Then, the multi-environment evolution skills are tested:

| Evolution           | Result  |
:-------------:| :-----:
| <img src="/uploads/6172d9b8a257b3eb07052433812fcb61/cartpole_all.png"  width="550" height="350" title="Cartpole(v0-3)">  | <img src="/uploads/0e4ecbc0e7d3b99be2df024bd86d4469/cartpole4.gif"  width="350" height="250" title="Cartpole(v0-3)"> |
| <img src="/uploads/3e4777db8f6877759f77d3589d9c1ced/acrobot_all.png"  width="550" height="350" title="Acrobot(v0-3)">    | <img src="/uploads/a7c6a3467d8c70d8d6410104a2d3126a/acrobot4.gif"  width="250" height="250" title="Acrobot(v0-3)"> |
| <img src="/uploads/4079af76a88d6537f85ce46632b7ee61/pendulum_all.png"  width="550" height="350" title="Pendulum(v0-3)">  | <img src="/uploads/a6036bbc452c960cfc06f0f5a9ea59a9/pendulum4.gif"  width="250" height="250" title="Pendulum(v0-3)"> |

MFEA is able to evolve multiple scenarios with the codification proposed and good results are achieved. In scenarios like *Pendulum* it finds more difficult to converge and so, worst results are harvested.

Finally, the effectiveness of the crossovers is studied. The knowledge transference in MFEA is done via this mechanism, thus, it is relevant to check its effectiveness:

<div align="center">
    <img src="/uploads/6de521b5bfe0be2efdabdce8b22cfe1d/crossover_matrix.png" width="500" height="375">
</div>

## Conclusions
In this work we have introduced and developed the application of a multifactorial evolutionary algorithm (MFEA) to solve multiple DQL environments at the same time. We have introduced a new encoding schema based on the usual transfer learning procedure and analyzed the positive and negative traits of this algorithm.

Empirically, we have analyzed the skills of MFEA to solve multiple reinforcement scenarios simultaneously. MFEA raises as well optimizer for our scenario, evolving up to nine tasks correctly using a unique evolution process and just one population, which is shared between all the environments/tasks. Then, after the success on evolving multiple tasks, we reviewed the effectiveness of the genetic knowledge transference applied to evolve this kind of neural networks. 

In terms of knowledge transference, as it is performed in a random fashion, it has been difficult to probe the existence of tendencies, showing all environments to contribute similarly in the evolutionary process.
