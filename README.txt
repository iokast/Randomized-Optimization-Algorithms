This file contains the instructions for how to run the code for Assignment 2

Important Notes
1) This project uses a modified version of ABAGAIL, located in the ABAGAIL sub-folder
2) The folders NNOUTPUT, CONTPEAKS, FLIPFLOP and TSP must be created in the same folder as the Jython code before running it.
3) The files s_test.csv, s_trg.csv and s_val.csv  must be in the same folder as the NN*.py files
4) To run the Jython files, please modify the files (line 5 for non-neural network experiments and line 9 for neural network experiments) so that the ABAGAIL.jar file is in the system path.
5) Code in this report and this README file were modified from those written by Jonathan Tay (https://github.com/JonathanTay)

Report:
1) bwetzel6-analysis.pdf - Assignment 2 report

Code Files: 
1) NN0.py - Code for Backpropagation training of neural network
2) NN1.py - Code for Randomised Hill Climbing training of neural network
3) NN2.py - Code for Simulated Annleaing training of neural network
4) NN3.py - Code for Genetic Algorithm training of neural network
5) continuouspeaks.py - Code to use Randomised Optimisation to solve the Continuous Peaks problem
5) flip flop.py - Code to use Randomised Optimisation to solve the Flip Flop problem
6) tsp.py - Code to use Randomised Optimisation to solve the Traveling Salesman Problem

There are also a number of folders
1) Datasets - contains the code to generate the datasets for this assignment from the original files from the UCI ML Repository
2) NNOUTPUT - output folder for the Neural Network experiments
3) CONTPEAKS - output folder for the Continuous Peaks experiments
4) FLIPFLOP - output folder for the Flip Flop experiments
5) TSP - output folder for the Traveling Salesman Problem experiments
6) ABAGAIL - folder with source, ant build file, and jar for ABAGAIL

Data Files
1) s_test.csv - The test set
2) s_trg.csv - The training set
3) s_val.csv - The validation set

To generate the data files from the original data, run the parse_data.py code and the DUMPER.py code in the Datasets folder.

The assignment code is written in Python 3.6.3. Library dependencies are: 
numpy: 1.13.3
pandas: 0.20.3
sklearn: 0.19.1

Java code was built with ant 1.10.2 on java 1.8.0_152. 
The code files in the code files section were written in Jython 2.7.0. 

Within the output folders, the data files are csv files (with .txt extensions). The file names correspond to experiments:
<ALGORITHM><PARAMETERS>_LOG.txt
*in the case of NN, the number of nodes per layer was added at the end of the filename.

In the data folders, there are also excel spreadsheets containing raw data and compiled data used to create plots.