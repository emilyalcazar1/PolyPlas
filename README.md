# **POLYPLAS v1.0.0**

## An open-source educational python program for topology optimization considering von Mises elastoplasticity. Instructions are listed below, see publication [HERE](https://link.springer.com/article/10.1007/s00158-025-04055-2) for more details.

Ensure you have the following already installed prior to the next steps:
- Python 3.10 or higher

### Instructions for PolyPlas Use

#### Install Anaconda
1. Install Anaconda by following instructions on their official website: https://docs.anaconda.com/anaconda/install/
2. Run the installer and follow the directions as seen on-screen
3. Verify the installation by running the following line in your terminal or command prompt
   ```bash
   conda --version
   ``` 
#### Clone the repository
1. Open the terminal or command prompt
2. Navigate to the location where you wish to clone the repo
3. Run the following bash command to clone the PolyPlas repo by
   ```bash
   git clone https://github.com/emilyalcazar1/PolyPlas.git
   ```
4. Navigate to the cloned repo location to prepare for the next steps 

#### Create Anaconda Environment
1. Create the environment using the environment.yml file in the repo by the following
   ```bash
   conda env create -f environment.yml --name polyplas_env
   ```
2. Activate the environment with the command
   ```bash
   conda activate polyplas_env
   ```
#### Run the PolyPlas Script
1. Ensure the correct environment is activated by the command
   ```bash
   conda activate polyplas_env
   ```
2. Run the selected script from the terminal by the following command
   (note multiple script files available based on the numerical examples in the paper) 
   ```bash
   python PolyPlasScript.py
   ```
4. Alternatively run the script from an IDE such as Visual Studio Code

## License

This project is licensed under the [MIT License](LICENSE). Please use, modify, and distribute this project as per the terms of the license.






