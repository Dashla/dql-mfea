pip install keras==2.2.4 keras-rl==0.4.2
pip install tensorflow==1.14.0

# Install gym
echo ".........."
echo "Downloading gym from https://github.com/openai/gym.git"

if [[ -d "gym" ]];
then
	read -p "gym folder is going to be deleted. Agree?: [y]/n: " var
	
	var=${var:="y"}
	
	if [ ! "$var" = "y" ];
	then
		echo "Aborting"
		exit
	else
		sudo rm -r gym
	fi

fi

git clone https://github.com/openai/gym.git
cd gym
pip install -e .
cd ../
cp -r gym_update/* gym/gym/
