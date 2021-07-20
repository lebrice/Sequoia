echo "Installing packages"
pip3 install -r requirements.txt
mkdir results
# To fix torchmeta batch not found bug
pip3 install pytorch-lightning --upgrade
