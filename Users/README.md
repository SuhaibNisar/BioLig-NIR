# Instructions

## Entering spectral data into Excel
- Create a folder with the name of the user who wishes to run samples. It is recommended for each individual user to create their own folder.
- **Copy** the Excel file SamplesData.xlsx into the user folder.
- In the Excel file, enter your sample details in the *Names* sheet, most importantly enter the sample name under *Sample Description* and the date of the sample in the dates tab **(dd/mm/yyyy)**. The date format is important as it is used to select samples for running.
- Copy your spectral data (collected in reflectance mode from 800-2500 nm at 4 nm intervals) into the *Reflectance* sheet. **Make sure it is in the same order as in the *Names* sheet.**
- (Optional) If you would like to plot wet lab compositional data alongside the NIR results, enter the data for the respective sample in the *Compositions* Sheet.

## Running the NIR models
- Open either *NIR_Script.py* or *NIR_Script.ipynb* in the main repository folder to run the model. Enter your name (same case and spelling as the folder name), the date you would like to run samples for (in **(dd/mm/yyyy)**), whether you would like to plot the wet lab compositional data (*optional*), and whether you want to run the alternative model for pseudo-lignin quantification (*optional*).
- Once the required details are entered, the model will run the desired samples and save the results to the user folder in terms of both figures and a CSV file containing the NIR results. The results will be saved using the date of when the samples were collected.