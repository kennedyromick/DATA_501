# DATA 501 Project - Beyond the Track: Analyzing Safety Car Incidents in Formula 1 for Enhanced Race Strategy

By: Akib Hasan Aryan and Kennedy Romick

## About the Project

### Objective: 
Investigate patterns, trends, and causal factors surrounding safety car incidents in F1 races
### Aims:
1. Answer when and why safety cars are deployed
2. Indentify correlations among incidents properties
3. Develop a predictive model
### Research Questions:
1. Under what conditions do safety cars occur most often?
2. Are there correlations between specific incident properties during safety car periods?
3. How can historical safety car incident data be leveraged to predict the probability of safety car deployments at different stages of future races?
### Input Data: 
- F1 Race Data from FastF1 (https://docs.fastf1.dev/index.html)
### Features: 
- Diplays previews of the race data
- Displays an interactive Parallel Coordinate Plot
- Runs a Monte Carlo simulation to predict the probability of a safety car occuring for each race, and validates against the next season

## How to run the project?

1. Clone or download this repository to your local machine.
2. Install all the libraries mentioned in the [requirements.txt](https://github.com/kennedyromick/DATA_501/blob/main/requirements.txt) file with the command `pip3 install -r requirements.txt`
3. Open your terminal/command prompt from your project directory and run the file `app.py` by executing the command `streamlit run app.py` (you may need to specify `python -m streamlit run app.py --server.enableXsrfProtection false`).
4. You will be automatically redirected the your localhost in brower where you can see you WebApp in live.

Source Code: [github link](https://github.com/everydaycodings/Data-Analysis-Web-App)
