# ML&Delta;TB
Machine Learning - &Delta; - Tight Binding (ML&Delta;TB) is a package that performs an environment-dependent learning of the corrections 
on the Slater integrals, in the context of the Empirical Tight Binding method.

## Setup
There are different scripts that can/must be used in the ML&Delta;TB framework. In order to use the scripts, the `mlDeltatb` package needs
to be built and installed from the source code. To do this, it is necessary to have the QuantumATK software installed, 
as well as `tensorflow` with at least version `2.12.0`.   

The first step, is to set up a virtual environment. The second step, is to install `tensorflow` and `mlDeltatb` in the virtual environment.   
**Using the python executable shipped with QuantumATK** (path is usually `QuantumATK/QuantumATK-{version}/atkpython/python` or 
`QuantumATK/QuantumATK-{version}/atkpython/bin/python`), do the following:   
- create a virtual environment: `python -m venv --system-site-packages MY_ENV_DIR`, where `MY_ENV_DIR` is the 
chosen environment name
- if on Linux:
  - activate the environment: `source MY_ENV_DIR/bin/activate`
  - update pip and setuptools: `python -m pip install setuptools pip --upgrade`
  - install tensorflow: `python -m pip install tensorflow`
  - install build: `python -m pip install build`
  - from the `mlDeltatb` main folder, build the package: `python -m build`
  - install `mlDeltatb`: `python -m pip install dist/mlDeltatb-0.0.1-py3-none-any.whl`
- if on Windows:
  * use python in the Scripts folder to update pip and setuptools: 
`MY_VENV_DIR/Scripts/python.exe -m pip install setuptools pip --upgrade`
  * use the same python to install tensorflow: `MY_VENV_DIR/Scripts/python.exe -m pip install tensorflow`
  * install build:  `MY_VENV_DIR/Scripts/python.exe -m pip install build`
  * from the `mlDeltatb` main folder, build the package: `MY_VENV_DIR/Scripts/python.exe -m build`
  * install `mlDeltatb`: `MY_VENV_DIR/Scripts/python.exe -m pip install dist/mlDeltatb-0.0.1-py3-none-any.whl`

***Note***: *on Linux, you can deactivate the virtual environment by use the command: `deactivate`.*   

Now, all the scripts in the next sections will have to be executed using the python executable created in the virtual 
environment. On Linux, this just means doing the following:
- `source MY_VENV_DIR/bin/activate`
- `python {scripts}`, where `scripts` is any of the scripts described in the next sections
- `deactivate`   

On Windows, just use the python executable present in the Scripts folder of the virtual environment:
- `MY_VENV_DIR/Scripts/python.exe {scripts}`, where `scripts` is any of the scripts described in the next sections

## Preliminary scripts
These scripts need to be executed before the training procedure can start. 

**Important**: This section is needed in order to generate from scratch the dataset and initial ETB parametrization used in the article.
In order to just reproduce the article results, you can skip this part, and use the unzipped `dataset.zip` folder as the `dataset_path` keyword required in the "Pre-training, Training and Validation scripts" section. Moreover, whenever it is required to use the `start_basis_set.pickle` file, you can find the already generated file in `scripts/fitting_results`. 

***Note***: *Remember to use the python executable of the virtual environment, following the steps of the previous section.*   

1. `scripts/Training_set_generation.py`: this script is used to compute the DFT references that will populate the 
training and test set. It is tailored for III-As-Sb alloys. The input options can be chosen by renaming the file 
`DSgen_options_template.json` to `DSgen_options.json`. The options in the json file are:

        {  
        "cation": the group III atom in the alloy, possible choices: ["ga", "in", "al"]   
        "a_antimonide": the lattice constant of the III-Sb compound (in Angstrom)    
        "a_arsenide": the lattice constant of the III-As compound (in Angstrom)     
        "alpha_antimonide": the DFT-HSE exchange fraction for the III-Sb compound   
        "alpha_arsenide": the DFT-HSE exchange fraction for the III-As compound  
        "n_ens": the number of realizations for each of the input percentages  
        "relax": whether the supercell will be internally relaxed  
        }

    All quantities are linearly interpolated between the two bulks according to the variation of the Sb-content <br>   
***Execution***:
   - copy `DSgen_options.json` in the same folder where the script is,
   - run `python Training_set_generation.py p`, where `p` is the Sb content in the alloy given in percentage (between 0 and 100).

   
2. `scripts/Fit_III-V.py`: this script is used to compute the starting ETB parametrization which the ML&Delta;TB method will 
learn to improve. It performs an automatic fitting of the bulk ETB parameters for the III-As and III-Sb materials 
simultaneously. So far, it is implemented for Gallium and Aluminum as cations. <br>   
***Execution***:  
   - generate DFT bulk band structures as target (III-As and III-Sb bulks), using the `.py` files in `scripts/fitting_targets/`,
   - copy the generated `fitting_targets` folder in your working directory,
   - from your working directory, run `python Fit_III-V.py compounds input_path output_path`, where `input_path` is the location 
of the `fitting_targets`, `output_path` is the location where the results are desired to be saved, 
and `compounds` is a string indicating which materials to fit. Possible choices are:
       - "gaas"
       - "gasb"
       - "gaas_gasb"
       - "alas_alsb"   

      The resulting parametrization (named `start_basis_set.pickle` as default) is saved in `output_path/fitting_results/`, 
   and is used as a starting point for the ML method, in `mlDeltatb/Common/ETBStartingPoint.py`. Refer to the `Training.py` script
   in the section below to know how to use this basis set
   

3. (Optional) `scripts/Extract_gap.py`: this script can be used to compute band gap statistics of the dataset generated in 
`scripts/Training_set_generation.py`. The results will be created in the dataset folder, in a file named 
`Bandgap_report.dat`<br>   
***Execution***:   
   - run `python Extract_gap.py input_path p`, where `input_path` is the location of the folders generated by 
   `scripts/Training_set_generation.py` (one folder for each percentage of Sb), and `p` is the Sb content given in 
   percentage (and the name of the corresponding folder containing the DFT references).

## Pre-training, Training and Validation scripts
***Note***: *Remember to use the python executable of the virtual environment, following the steps of the "Setup" section.*  

After having generated the DFT dataset and the starting ETB parametrization (you can also use the already generated ones, see the "Important" note in the previous section), the training procedure can start.
In order to train and validate the model, a series of options must be chosen after renaming the `config_template.json` 
file to `config.json`.   
The options to choose are:   
        
        {
        "n_atoms": number of atoms in each structure of the dataset   
        "batch_size": how many structures to consider before updating the gradient     
        "emin_emax": energy window (in eV) for the band structure fitting during training, affects the loss function  
        "n_kpoints": the range of kpoints to consider on the DFT reference  
        "k_downsampling": how many kpoints to skip in the DFT reference band structure, useful for faster training
        "start_basis_set_name": The name of the Slater-Koster parametrization file generated with the Fit-III-V.py script.    
        Advanced user option. Leave this option as default, if no modification of the generated basis set is desired.    

        "dataset": {  
        "concentrations": a list of the Sb- percentages generated with `scripts/Training_set_generation.py`. The numbers
        must be strings     
        "train_structures_per_concentration": the number of structures to include in the training set for each of the
        Sb concentrations in the dataset   
          }
        
        "model": {  
        "layer_nodes": list containing the number of nodes for each layer of the neural network to train, it defines 
        the network architecture  
        "correction_type": the type of correction to implement, possible choices: ["onsite_subshell" (4 output nodes), 
        "onsite_orbital" (10 output nodes)]. "onsite_subshell" is the type of correction defined in the article.   
          }
        
        "training": {  
        "epochs": the total number of epochs for training. 1 epoch = 1 Evaluation of the loss function on the entire 
        training set   
        "checkpoint_epochs": the number of epochs that pass before saving the partial results  
        "resume_training": if set to `true`, training will resume on the epoch set in "resume_epoch". The file 
        `saved_weights_{resume_epoch}.pickle`, containing the weights of the neural network saved during a previous checkpoint, 
        needs to be in the input folder.
        "resume_epoch": the epoch on which to resume training, does not have any effect if "resume_training" is set 
        to `false`  
          }
        
        "validation": {  
        "epoch_min": the starting epoch on which to evaluate the loss function on the test set  
        "epoch_max": the final epoch on which to evaluate the loss function on the test set. If equal to "epoch_min", 
        the evaluation will be performed only on that epoch.
        "epoch_step": the number of epochs between one evaluation on the test set and the next. It must be set the 
        same as `["training"]["checkpoint_epochs"]` in order to be consistent with the saved weights   
          }    
        }

Once all the options are set, the following commands can be executed (in order):
1. Pre-training:  
   - copy `config.json` to a new folder (`MY_NEW_FOLDER`)
   - from your working folder, run `python PreTraining.py dataset_path input_path output_path`, where `dataset_path` is 
   the location of the DFT references generated with `Training_set_generations.py`, `input_path` is the path to
   `MY_NEW_FOLDER`, and `output_path` is the desired location of the results.   
   This script generates a file, named `pre_trained_weights.pickle`, which contains the weights of the network forced to
   output corrections as close as possible to 0, for all possible points in the training set.   
2. Training:   
   - copy `config.json` and `pre_trained_weights.pickle` to a new folder (`MY_NEW_FOLDER`)
   - from `<output_path>/fitting_results/` of point `2.` in the "Preliminary scripts" section, copy `start_basis_set.pickle` to `MY_NEW_FOLDER`
   - from your working folder, run `python Training.py dataset_path input_path output_path`, where `dataset_path` is 
   the location of the DFT references generated with `Training_set_generations.py`, `input_path` is the path to
   `MY_NEW_FOLDER`, and `output_path` is the desired location of the results.   
   This script trains the model, and saves the converged model weights in `output_path/Results_train`; as well as other
   quantities necessary for the next scripts (`rcut.pickle`, `var_x.pickle`, `mean_x.pickle`).   

3. Validating:   
   - copy `config.json` to a new folder (`MY_NEW_FOLDER`)
   - from `scripts/fitting_results`, copy `start_basis_set.pickle` to `MY_NEW_FOLDER`
   - from `output_path/Results_train` of the previous step, copy `rcut.pickle`,`var_x.pickle` and `mean_x.pickle` to 
   `MY_NEW_FOLDER`
   - from `output_path/Results_train` of the previous step, copy the weights 
   [`saved_weights_{epoch_min}.pickle`,..., `saved_weights_{epoch_max}.pickle`], of the converged model, to `MY_NEW_FOLDER`. 
   Make sure all the epochs defined by `["validation"]` in the `config.json` file are copied.   
   - from your working folder, run `python Validating.py dataset_path input_path output_path`, where `dataset_path` is 
   the location of the DFT references generated with `Training_set_generations.py`, `input_path` is the path to
   `MY_NEW_FOLDER`, and `output_path` is the desired location of the results.   
   This script validates the model, by showing the result of the fitting on the training set as well as comparing the 
   model prediction to the DFT references of the test set.

## Using the model after training
***Note***: *Remember to use the python executable of the virtual environment, following the steps of the "Setup" section.*   

After the model has been trained, it is possible to use it on III-As-Sb alloy structures using the script in
`scripts/Compute_new_bandstructure.py`. In order to do so, some files need to be copied in a new folder (`MY_NEW_FOLDER`):
- `start_basis_set.pickle`: this file is located in `scripts/fitting_results`.
- `rcut.pickle`, `mean_x.pickle`, `var_x.pickle`: these files were generated during training, and they are necessary to 
reproduce the same tensorflow model and configuration that was trained. They are located in `output_path/Results_train`,
where `output_path` is the same path given as an argument in the `Training.py` script.
- `saved_weights_{epoch}.pickle`: this file contains the weights of the ML model converged at epoch `epoch`. 
It is located in `output_path/Results_train`, where `output_path` is the same path given as an argument in the 
`Training.py` script.   

Next, a band structure can be computed on the new atomistic configuration.    
In order to do so, from your working folder run `python Compute_new_bandstructure.py /path/to/MY_NEW_FOLDER configuration epoch guess_fermi_level` where:
- `/path/to/MY_NEW_FOLDER` is the location of the folder created at the beginning of this section
- `configuration` is the `.hdf5` file containing the QuantumATK `BulkConfiguration` object that we want to simulate with ETB
- `epoch` is the desired epoch of the converged model, the same of `saved_weights_{epoch}.pickle`
- `guess_fermi_level` is the user's estimation for the structure's Fermi level (in eV). Needed for the iterative solver.
