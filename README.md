# AGM_grid_resiliency_reserve_policy

# Dependencies:

1. ```pip install tensorflow==2.2.0```

2. ```pip install gym```


3. ```pip install ray==1.0.1```


4. ```pip install matplotlib```

5. ```conda install pandas```

6. ```pip install numpy==1.19.5```

7. ```conda install -c conda-forge Julia``` [This step is for Eagle installation - if you don’t have prior installation]. Visit Julia website for installation on a local machine.

8. ```pip install pyomo ```

9. ```conda install glpk ipopt_bin -c cachemeorg```

10. ```pip install "xpress<8.9"``` [This step is for Eagle installation - to use the xpress solver license with Pyomo]. Make sure if you have the xpressmp license if you would like to use it on a local machine before this step.

11. Install required Julia packages - open a julia session and install the following packages:
   
    ```julia> using Pkg```

    ```julia> Pkg.add("PyCall")```

    ```julia> Pkg.add("JuMP")```

    ```julia> Pkg.add("GLPK")```

    ```julia> Pkg.add("CSV")```

    ```julia> Pkg.add("DataFrames")    ```

12. Install our own environment library: Go to 'gym_opf_env' folder (contains setup.py), then use command ```conda develop .```. After this you will see something similar to: ```completed operation for: /lustre/eaglefs/scratch/aeseye/agm_grid_resiliency/rl_reserve_policy/gym_opf_env```. To verify our environment is successfully installed, start a python session, and run ```import opf_envs```, you should see no error, that means you have installed our environment correctly.