# Trade-off analysis
Optimization and visualization code for "Trade-offs between greenhouse gas mitigation and economic objectives with drained peatlands in Finnish landscapes" manuscript

The following files are contained in this directory:
README.md -- brief instructions to run the analysis. <br/>
LICENSE -- an MIT license for this git. <br/>
Trade_off_SpaFHy_TEMPLATE.py -- python script to run the optimization analysis. <br/>
Figures.ipynb -- ipython notebook to construct the figures from the manuscript (reduced to a single case) -- Note all 3 solutions need to be solved before running. <br/>
environment.yml -- conda environment file, detailing packages used in the analysis

This code relies entirely on python script. 

The conda environment used to run the script can be replicated using the environement.yml file.

We used the Anaconda to install python, and load the environment (anaconda.com).

From the command line, to install the environment, move to the specific directory where the git has been cloned, and run:

conda env create -f environment.yml

# Three instances need to be run

python Trade_off_SpaFHy_TEMPLATE.py --area Uusimaa --trade NPV_PEAT --constraint INC --emmision_type BM_GHG & <br/>
python Trade_off_SpaFHy_TEMPLATE.py --area Uusimaa --trade INC_PEAT --constraint NPV --emmision_type BM_GHG & <br/>
python Trade_off_SpaFHy_TEMPLATE.py --area Uusimaa --trade NPV_INC --constraint PEAT --emmision_type BM_GHG & <br/>

#For only Biomass:

python Trade_off_SpaFHy_TEMPLATE.py --area Uusimaa --trade NPV_PEAT --constraint INC --emmision_type BM & <br/>
python Trade_off_SpaFHy_TEMPLATE.py --area Uusimaa --trade INC_PEAT --constraint NPV --emmision_type BM & <br/>
python Trade_off_SpaFHy_TEMPLATE.py --area Uusimaa --trade NPV_INC --constraint PEAT --emmision_type BM & <br/>
