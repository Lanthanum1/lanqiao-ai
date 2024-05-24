#task-start
import pandas as pd
import numpy as np
import pickle

def predictY(test_data):
    #TODO
    with open("/home/project/model1.pkl", 'rb') as f:
        model1 = pickle.load(f) 
    with open("/home/project/model2.pkl", 'rb') as f:
        model2 = pickle.load(f)
    with open("/home/project/model3.pkl", 'rb') as f:
        model3 = pickle.load(f)

    output1 = model1.predict(test_data)
    output2 = model2.predict(test_data)
    output3 = model3.predict(test_data)
    output_ensemble = output1 * 0.3 + output2 * 0.1 + output3 * 0.6

    return output_ensemble

def main():
    test_data = {
        'acousticness_yr': 0.777447,
        'tempo_yr': 112.511724,
        'liveness_yr': 0.202646,
        'loudness_yr': -14.219955,
        'danceability_yr': 0.478132,
        'instrumentalness': 0.001380,
        'tempo_ar': 122.664884,
        'instrumentalness_yr': 0.189863,
        'popularity_yr': 19.231500,
        'speechiness': 0.039900,
        'loudness_ar': -13.281221,
        'acousticness_ar': 0.583168,
        'explicit': 0.000000,
        'year': 1958.000000,
        'valence_yr': 0.490335,
        'energy_ar': 0.464832,
        'duration_ms_yr': 213321.451000,
        'danceability_ar': 0.592695,
        'popularity_ar': 29.978947,
        'energy_yr': 0.308868
    }
    df_test = pd.DataFrame([test_data])
    prediction = predictY(df_test)
    print(prediction)

if __name__ == "__main__":
    main()
#task-end