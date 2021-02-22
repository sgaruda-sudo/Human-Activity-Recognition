<h1 align="center">
	Human Activity Recognition Using LSTM
</h1>

### *Paper*: [Human Activity Recognition Using LSTM](https://github.com/sgaruda-sudo/Human-Activity-Recognition/blob/master/paper_HumanActivityRecognition.pdf)

### **For Running the script follow the [instructions here](https://github.com/sgaruda-sudo/Human-Activity-Recognition/blob/master/README.md#instructions-to-run-the-script) or run the same using [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NT3deC4z0B6B1Z7L_-MWMvVzBjH2Yz3n?usp=sharing)**
*Change the hyper parameters in ```constants.py```*

1. [Input Pipeline]()- 
    * [HAPT Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) (Training set- Users(1-21), Test set-Users(22-27), validation set-Users(28-30)
    * Data Denoising
    * Z Score normalization
2. [Model Architecture](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/master/HAR/models/model_arch.py)
3. [Training Routine](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/d8fa7a96e5d6b1076e6ace9a0188cba215939dde/HAR/main.py#L50)
4. Model CallBacks:
    * [Check point Callback](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/d8fa7a96e5d6b1076e6ace9a0188cba215939dde/HAR/main.py#L33) - For saving model at desired interval(epoch frequency)
    * [Tensorboard Callback](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/d8fa7a96e5d6b1076e6ace9a0188cba215939dde/HAR/main.py#L25) - For logging training stats,Profiling
    * [CSV Logger Callback](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/d8fa7a96e5d6b1076e6ace9a0188cba215939dde/HAR/main.py#L40) - To save training logs in a csv file
6.Hyper parameter tuning
7. [Evaluation](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/d8fa7a96e5d6b1076e6ace9a0188cba215939dde/HAR/main.py#L58)
    * [Confusion Matrix](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/tree/master/HAR#confusion-matrix-in-)
    * [Classification Report](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/tree/master/HAR#classification-report)
8. [Visualization](https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/master/HAR/media/test_data.png)
## Outputs from several stages of project
* **Peek into a sample set from Training Data:**
	<p align="center">
	  <img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/master/HAR/media/sample_train.png" /> 
	</p>
	<p align="center">
	    <em>Processed and Augmented Images</em>
	</p>
	
* **Model Architecture:**
	<p align="center">
	  <img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/master/HAR/media/model_architecture.png" />
	</p>
	<p align="center">
	    <em>Model Architecture</em>
	</p>
* **Training Results:**
	<p align="center">
	  <img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/master/HAR/media/epoch_accuracy.svg" height="200" />
	  <img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/master/HAR/media/epoch_loss.svg" height="200"/>
	  <img src=https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/master/HAR/media/legend.PNG />
	</p>
<p align="center">
    <em><b>(a) Epochs vs Accuracy &emsp;&emsp;&emsp;&emsp; &emsp; &emsp; &emsp; &emsp;(b) Epochs vs Loss</b></em>
</p>

* **Hyper parameter tuning:**

	<table align="center">
	<thead align="center">
	<tr class="header">
	<th align="center" colspan="4">Hyperparameter Tuning</th>
	</tr>
	</thead>
	<tbody>
	<tr class="odd">
	<td align="left">LSTM_1 Neurons</td>
	<td align="left">LSTM_2 Neurons</td>
	<td align="left">Optimizer</td>
	<td align="left">Validation_Accuracy</td>
	</tr>
	<tr class="even">
	<td align="left"><strong>128</strong></td>
	<td align="left"><strong>64</strong></td>
	<td align="left"><span><strong>RMSProp</strong></span></td>
	<td align="left"><strong>83.1%</strong></td>
	</tr>
	<tr class="odd">
	<td align="left">112</td>
	<td align="left">16</td>
	<td align="left">RMSProp</td>
	<td align="left">81.4%</td>
	</tr>
	<tr class="even">
	<td align="left">96</td>
	<td align="left">48</td>
	<td align="left">RMSProp</td>
	<td align="left">80.8%</td>
	</tr>
	<tr class="odd">
	<td align="left">80</td>
	<td align="left">64</td>
	<td align="left">RMSProp</td>
	<td align="left">80.1%</td>
	</tr>
	<tr class="even">
	<td align="left">128</td>
	<td align="left">80</td>
	<td align="left">Adam</td>
	<td align="left">81.1</td>
	</tr>
	<tr class="odd">
	<td align="left">112</td>
	<td align="left">32</td>
	<td align="left">Adam</td>
	<td align="left">80.3</td>
	</tr>
	<tr class="even">
	<td align="left">128</td>
	<td align="left">64</td>
	<td align="left">Adam</td>
	<td align="left">79.1</td>
	</tr>
	<tr class="odd">
	<td align="left">80</td>
	<td align="left">80</td>
	<td align="left">Adam</td>
	<td align="left">77.1</td>
	</tr>
	</tbody>
	</table>
* **Results and Evaluation**
	* #### Test accuracy - 90.35%
	* #### Confusion Matrix (in %):
	<p align="center">
	  <img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/master/HAR/media/normalized_cm.png" width="600" height="580"/>
	</p>

	* #### Classification report:
	<table border="1" class="classificationrep" align="center">
	  <thead>
	    <tr style="text-align: right;">
	      <th></th>
	      <th>precision</th>
	      <th>recall</th>
	      <th>f1-score</th>
	      <th>support</th>
	    </tr>
	  </thead>
	  <tbody>
	    <tr>
	      <th>WALKING</th>
	      <td>0.88</td>
	      <td>0.97</td>
	      <td>0.92</td>
	      <td>24805</td>
	    </tr>
	    <tr>
	      <th>WALKING_UPSTAIRS</th>
	      <td>0.98</td>
	      <td>0.84</td>
	      <td>0.90</td>
	      <td>24046</td>
	    </tr>
	    <tr>
	      <th>WALKING_DOWNSTAIRS</th>
	      <td>0.95</td>
	      <td>0.92</td>
	      <td>0.93</td>
	      <td>22722</td>
	    </tr>
	    <tr>
	      <th>SITTING</th>
	      <td>0.87</td>
	      <td>0.99</td>
	      <td>0.93</td>
	      <td>29032</td>
	    </tr>
	    <tr>
	      <th>STANDING</th>
	      <td>0.94</td>
	      <td>0.88</td>
	      <td>0.91</td>
	      <td>30490</td>
	    </tr>
	    <tr>
	      <th>LAYING</th>
	      <td>0.97</td>
	      <td>0.98</td>
	      <td>0.97</td>
	      <td>30677</td>
	    </tr>
	    <tr>
	      <th>STAND_TO_SIT</th>
	      <td>0.82</td>
	      <td>0.45</td>
	      <td>0.59</td>
	      <td>2049</td>
	    </tr>
	    <tr>
	      <th>SIT_TO_STAND</th>
	      <td>0.85</td>
	      <td>0.66</td>
	      <td>0.74</td>
	      <td>1614</td>
	    </tr>
	    <tr>
	      <th>SIT_TO_LIE</th>
	      <td>0.69</td>
	      <td>0.69</td>
	      <td>0.69</td>
	      <td>2397</td>
	    </tr>
	    <tr>
	      <th>LIE_TO_SIT</th>
	      <td>0.64</td>
	      <td>0.56</td>
	      <td>0.60</td>
	      <td>2372</td>
	    </tr>
	    <tr>
	      <th>STAND_TO_LIE</th>
	      <td>0.44</td>
	      <td>0.62</td>
	      <td>0.51</td>
	      <td>2719</td>
	    </tr>
	    <tr>
	      <th>LIE_TO_STAND</th>
	      <td>0.53</td>
	      <td>0.53</td>
	      <td>0.53</td>
	      <td>2327</td>
	    </tr>
	    <tr>
	      <th>accuracy</th>
	      <td>0.904</td>
	      <td>0.904</td>
	      <td>0.904</td>
	      <td>0.904</td>
	    </tr>
	    <tr>
	      <th>macro avg</th>
	      <td>0.80</td>
	      <td>0.76</td>
	      <td>0.77</td>
	      <td>175250</td>
	    </tr>
	    <tr>
	      <th>weighted avg</th>
	      <td>0.91</td>
	      <td>0.90</td>
	      <td>0.90</td>
	      <td>175250</td>
	    </tr>
	  </tbody>
	</table>

* **Visualization of predictions on a piece of test data set:**

	<p align="center">
	  <img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-2020-team14/blob/master/HAR/media/test_data.png" />
	</p>
	
  	
## Instructions to run the script:
Before running the script Install the [requirments](https://github.com/sgaruda-sudo/Human-Activity-Recognition/blob/master/requirements.txt) from ```requirements.txt``` using ```pip install -r requirements.txt```

* **Make the following changes in main.py based on the Mode(training mode, hyper parameter tuning mode, finetuning mode, evaluation mode) you want you the script to run in**

	1. To Train the model, change the train FLAG in ```main.py``` to ```True```  

		```flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')```
		
	2. To log the data , specify path to tensorboard callback, model chekpoint call back, CSVlogger call back in ```constants.py```
		```
		tensorboard_PATH = './log_dir/fit/'
		checkpoint_PATH = './log_dir/cpts/'
		csv_log_PATH = './log_dir/csv_log/' 
		```
	3. To Evaluate the model on pretrained model:
		* change the train FLAG in ```main.py``` to ```False```  
		  ```flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')```
	 	* The specifiy the path to pretrained model in ```constants.py```  and path to save results 
		  ```trained_model_PATH = 'weights/Accuracy_90.h5'```
		  ```results_PATH = 'results/'```
		* To Save the predictions set ```SAVE_CSV=1``` in ```constants.py```
		* Or Alternative to step1, run ```python evaluate.py``` in the environment created using dependencied in ```requirements.txt``` in your terminal
		
	4. To Visualize the predictions on test data set:
	 	* The specifiy the path to save results in ```constants.py``` 
		  ```results_PATH = 'results/'```
		* Then run ```python vis.py``` in the environment created using dependencied in ```requirements.txt``` in your terminal, visulaization plot will be stored in the path provided


### Directory Structure for HAR:
```
.HAR
├── ./HAPT_DataSet
│   ├── ./HAPT_DataSet/1_7ZD-u-h8hFPuN2PYJvLMBw.png
│   ├── ./HAPT_DataSet/README.txt
│   ├── ./HAPT_DataSet/RawData
│   ├── ./HAPT_DataSet/activity_labels.txt
│   ├── ./HAPT_DataSet/features.txt
│   └── ./HAPT_DataSet/features_info.txt
├── ./README.md
├── ./constants.py
├── ./evaluate.py
├── ./input_pipeline
│   ├── ./input_pipeline/Numpy_arrays_colab.zip
│   ├── ./input_pipeline/dataset_building.py
│   ├── ./input_pipeline/notes.py
│   ├── ./input_pipeline/test.npz
│   ├── ./input_pipeline/train.npz
│   └── ./input_pipeline/valid.npz
├── ./log_dir
│   ├── ./log_dir/cpts
│   │   └── ./log_dir/cpts/saved_model8_Acc82.83_Loss34
│   ├── ./log_dir/csv_log
│   └── ./log_dir/fit
│       └── ./log_dir/fit/tensbrdlogs
├── ./main.py
├── ./media
├── ./models
│   └── ./models/model_arch.py
├── ./requirements.txt
├── ./results
│   ├── ./results/classification_report.csv
│   ├── ./results/confusion_matrix.png
│   └── ./results/eval_data.csv
├── ./vis.py
└── ./weights
    ├── ./weights/Accuracy_90.h5
    └── ./weights/final_model
```
