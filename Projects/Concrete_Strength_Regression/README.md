# Concrete Strength Project

In this project, I will perform everything from data wranging to regression modeling evaluation for data on concrete composition and strength.  The data comes from Kaggle user MAAJDL (https://www.kaggle.com/datasets/maajdl/yeh-concret-data).  The "About Dataset" section below contains the provided datacard.  

The following files can be run:  
* run_model.py to initialize, tune, train, and save a model  
* print_model_params.py to print a specific model's parameters  
* predict.py to use a model to predict  

Data and models are stored in the following locations:  
* data: data from Kaggle and model params from notebooks  
* data/raw_data: data used to tune and train the models (all the same data in this case)  
* models: models tuned and trained in run_model.py  
* results: results from predict.py  

---

## About Dataset
### Context
#### Abstract:
Concrete is the most important material in civil engineering.
The concrete compressive strength is a highly nonlinear function of age and ingredients.

### Content
#### Concrete Compressive Strength Data Set
#### Data Set Information:
Number of instances 1030  
Number of Attributes 9  
Attribute breakdown 8 quantitative input variables, and 1 quantitative output variable  
Missing Attribute Values None

#### Attribute Information:
Given are the variable name, variable type, the measurement unit and a brief description. The concrete compressive strength is the regression problem. The order of this listing corresponds to the order of numerals along the rows of the database.

Name -- Data Type -- Measurement -- Description

Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable  
Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable  
Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable  
Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable  
Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable  
Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable  
Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable  
Age -- quantitative -- Day (1~365) -- Input Variable  
Concrete compressive strength -- quantitative -- MPa -- Output Variable  

### Acknowledgements
#### Source:
Original Owner and Donor  
Prof. I-Cheng Yeh  
Department of Information Management  
Chung-Hua University,  
Hsin Chu, Taiwan 30067, R.O.C.  
e-mail:icyeh '@' chu.edu.tw  
TEL:886-3-5186511  

Date Donated: August 3, 2007

From: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

#### Relevant Papers:
**Main**
1) I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998).

**Others**

2) I-Cheng Yeh, "Modeling Concrete Strength with Augment-Neuron Networks," J. of Materials in Civil Engineering, ASCE, Vol. 10, No. 4, pp. 263-268 (1998).

3) I-Cheng Yeh, "Design of High Performance Concrete Mixture Using Neural Networks," J. of Computing in Civil Engineering, ASCE, Vol. 13, No. 1, pp. 36-42 (1999).

4) I-Cheng Yeh, "Prediction of Strength of Fly Ash and Slag Concrete By The Use of Artificial Neural Networks," Journal of the Chinese Institute of Civil and Hydraulic Engineering, Vol. 15, No. 4, pp. 659-663 (2003).

5) I-Cheng Yeh, "A mix Proportioning Methodology for Fly Ash and Slag Concrete Using Artificial Neural Networks," Chung Hua Journal of Science and Engineering, Vol. 1, No. 1, pp. 77-84 (2003).

6) Yeh, I-Cheng, "Analysis of strength of concrete using design of experiments and neural networks," Journal of Materials in Civil Engineering, ASCE, Vol.18, No.4, pp.597-604 (2006).

#### Citation Request:
NOTE: Reuse of this database is unlimited with retention of copyright notice for Prof. I-Cheng Yeh and the following published paper:

I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998).

### Inspiration
Can you predict the strength of concrete?