## Reinforcement Learning

#### Tool for ipython colaboration

**nbstripout**

A tool that will empty all output fields from a `.ipynb` when it's added to git
which allows for better colaboration.

##### Installation

```bash
pip install nbstripout
nbstripout --install
```
Make sure you run the install command while in a directory with git.   
This will add a script to the git pipeline to clean up ipynb files.


#### Setup environment

Install environment
```bash
conda env create -f environment.yml
```

Activate environment
```bash
conda activate rl2020
```

Deactivate environment
```bash
conda deactivate
```

Remove environment
```bash
conda remove --name rl2020 --all
```

