# + Individual experiments
python3 -u exp.py --path CARTPOLE --filename ./experiment_list/cartpole.txt
python3 -u exp.py --path CARTPOLE-V0 --filename ./experiment_list/cartpole0.txt
python3 -u exp.py --path CARTPOLE-V2 --filename ./experiment_list/cartpole2.txt
python3 -u exp.py --path CARTPOLE-V3 --filename ./experiment_list/cartpole3.txt
python3 -u exp.py --path ACROBOT --filename ./experiment_list/acrobot.txt
python3 -u exp.py --path ACROBOT-V0 --filename ./experiment_list/acrobot0.txt 
python3 -u exp.py --path ACROBOT-V2 --filename ./experiment_list/acrobot2.txt
python3 -u exp.py --path ACROBOT-V3 --filename ./experiment_list/acrobot3.txt
python3 -u exp.py --path PENDULUM --filename ./experiment_list/pendulum.txt
python3 -u exp.py --path PENDULUM-V0 --filename ./experiment_list/pendulum0.txt
python3 -u exp.py --path PENDULUM-V2 --filename ./experiment_list/pendulum2.txt
python3 -u exp.py --path PENDULUM-V3 --filename ./experiment_list/pendulum3.txt

# + Cartpole + Pendulum + Acrobot
python3 -u exp.py --path ALL-v1 --filename ./experiment_list/exp1.txt
python3 -u exp.py --path ALL-v1-NoCruce --filename ./experiment_list/exp1.txt

# + Cartpole + Pendulum + Acrobot Different environment
python3 -u exp.py --path ALL-multiplev --filename ./experiment_list/exp2.txt

# + Check TL between same environmets
python3 -u exp.py --path CARTPOLE-TL --filename ./experiment_list/exp3.txt
python3 -u exp.py --path PENDULUM-TL --filename ./experiment_list/exp4.txt
python3 -u exp.py --path ACROBOT-TL --filename ./experiment_list/exp5.txt
