sudo apt-get install python3-virtualenv -y

virtualenv -p /usr/bin/python3.8 CGM-Wiener

source CGM-Wiener/bin/activate

pip install -r requirements.txt