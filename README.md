>This work can be cited as:
>
>@article{martinez2020simultaneously,
  title={Simultaneously Evolving Deep Reinforcement Learning Models using Multifactorial Optimization},
  author={Martinez, Aritz D and Osaba, Eneko and Del Ser, Javier and Herrera, Francisco},
  journal={arXiv preprint arXiv:2002.12133},
  year={2020}
}

# Setting up the environment

In order to run the experimentation we recommend to create a conda environment from the .yml or requirements file as follows:

To create a new conda encironment from the requirements file:

```bash
conda create -n dql-mfea --file requirements.txt
```

using the .yml

```bash
conda env create -f dql-mfea.yml
```

Then, activate the environment and run install.sh script:

```bash
conda activate dql-mfea
./install.sh
```

this will install all the dependencies. Finally run the experiment, to replicate the full experimentation run:

```bash
./EXPERIMENTATION.sh
```

or

```bash
python3 exp.py --path (folder where to save data) --filename (path to .txt file where the environments to evolve are specified) 
```

to run your custom experiments.


# Simultaneously Evolving Deep Reinforcement Learning Models via Multifactorial Optimization 
## Abstract
In the recent years, Multifactorial Optimization (MFO) has attracted a lot of interest in the optimization community. MFO is known for its inherent skills to address multiple complex optimization tasks at the same time, while inter-task information transfer is used to improve their convergence speed. These skills make Multifactorial Evolution appealing to be applied to evolve Deep Reinforcement Learning (DQL) models, which is the scenario tackled in this paper. Complex DQL models usually find difficult to converge to optimal solutions, due to the lack of exploration or sparse rewards. In order to overcome these drawbacks, pre-trained models are commonly used to make Transfer Learning, transferring knowledge from the pre-trained to the target domain. Besides, it has been shown that the lack of exploration can be reduced by using meta-heuristic optimization approaches. In this paper we aim to explore the use of the MFO framework to optimize DQL models, making an analysis between MFO and the traditional Transfer Learning and metaheuristic approaches in terms of convergence, speed and policy quality.

Source code of MFEA used in the paper: [link](https://github.com/HuangLingYu96/MFEA)
## Codification
<div align="center">
<img src="/uploads/6f3a6b1a35bf53a8cdc94ab86a6a3883/Codification.png"  width="500" height="300" title="Codification schema">
</div>


## Experimentation

In this work we experiment with MFEA and it is used to train/evolve DQL networks. The environments used are: [*Cartpole*](https://gym.openai.com/envs/CartPole-v0/), [*Acrobot*](https://gym.openai.com/envs/Acrobot-v1/) and [*Pendulum*](https://gym.openai.com/envs/Pendulum-v0/). First the performance of MFEA on this scenarios is analyzed:


| Evolution | Model Performance |
:-------------------------:|:-------------------------:
<img src="/uploads/15a151b7299ddca5166fe29b3f38f6e9/1Cartpole.png"  width="550" height="300" title="Cartpole"/> | <img src="/uploads/55e5e56c801ea181604965edf19302d8/cartpole.gif"  width="350" height="250" title="Cartpole"/>
<img src="/uploads/1f55f9240b1078aa0b58b69e1e8c2ff9/1Acrobot.png"  width="550" height="300" title="Acrobot"/> | <img src="/uploads/1ce54972bb9223fad1963a2e90f8c796/acrobot.gif"  width="250" height="250" title="Acrobot"/>
<img src="/uploads/39372523a1c631f9cf22457063649029/1Pendulum.png"  width="550" height="300" title="Pendulum"> | <img src="/uploads/486666ff961f3103bc4bf77f3c2de1fd/pendulum.gif"  width="250" height="250" title="Pendulum">



We observe quick convergence times and good results, as the *Pendulum* environment is more complex the convergence is also slower

Then, the multi-environment evolution skills are tested:

| Evolution           | Result  |
:-------------:| :-----:
| <img src="/uploads/3c06e797826f895c0c8525b9e8b2a28d/cartpole_all.png"  width="550" height="350" title="Cartpole(v0-3)">  | <img src="/uploads/e803e57687b90b8ca44079a43c403ccf/cartpole4.gif"  width="350" height="250" title="Cartpole(v0-3)"> |
| <img src="/uploads/3bd629190f948c199744b1b30c8f1c7f/acrobot_all.png"  width="550" height="350" title="Acrobot(v0-3)">    | <img src="/uploads/092bb71181be6260a0e16140ac9b1ad5/acrobot4.gif"  width="250" height="250" title="Acrobot(v0-3)"> |
| <img src="/uploads/ac0fcbcc0cc6a1377306c78315e5b0e6/pendulum_all.png"  width="550" height="350" title="Pendulum(v0-3)">  | <img src="/uploads/0bafa527b803585d280457161853a0f8/pendulum4.gif"  width="250" height="250" title="Pendulum(v0-3)"> |

MFEA is able to evolve multiple scenarios with the codification proposed and good results are achieved. In scenarios like *Pendulum* it finds more difficult to converge and so, worst results are harvested.

Finally, the effectiveness of the crossovers is studied. The knowledge transference in MFEA is done via this mechanism, thus, it is relevant to check its effectiveness:

<div align="center">
    <img src="/uploads/69ea886ead6db1be6583482ddaa117e2/crossover_matrix.png" width="500" height="375">
</div>

## Conclusions
In this work we have introduced and developed the application of a multifactorial evolutionary algorithm (MFEA) to solve multiple DQL environments at the same time. We have introduced a new encoding schema based on the usual transfer learning procedure and analyzed the positive and negative traits of this algorithm.

Empirically, we have analyzed the skills of MFEA to solve multiple reinforcement scenarios simultaneously. MFEA raises as well optimizer for our scenario, evolving up to nine tasks correctly using a unique evolution process and just one population, which is shared between all the environments/tasks. Then, after the success on evolving multiple tasks, we reviewed the effectiveness of the genetic knowledge transference applied to evolve this kind of neural networks. 

In terms of knowledge transference, as it is performed in a random fashion, it has been difficult to probe the existence of tendencies, showing all environments to contribute similarly in the evolutionary process.
