# if high school/state is not in list, median values are chosen.


import pandas as pd
import pickle
import csv

csv_file_path = 'feeder_schools.csv'

schools_dict = {}

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        school_name = row['School']
        acceptance_rate = float(row['Acceptance Rate'])
        schools_dict[school_name] = acceptance_rate

income_weight = 0.5
michigan_weight = 0.3

gpa_ranges = [(4.0, 0.4), (3.75, 0.3), (3.5, 0.2), (3.25, 0)]

school = str(input('Please enter high school: '))
sat = int(input('Please enter SAT: '))
income = int(input('Please enter household income: '))
michigan = str(input('In state? (Y/N): '))
gpa = float(input('Please enter GPA: '))

user_input_dict = {
    'School': school,
    'Mean SAT': sat,
    'Income': income,  
    'Michigan': michigan,
    'GPA': gpa
}

with open('linear_regression_model_knn.pkl', 'rb') as file:
    pipeline = pickle.load(file)

user_prediction = pipeline.predict(pd.DataFrame([user_input_dict]))

income_value = user_input_dict['Income']
michigan_value = 1 if user_input_dict['Michigan'] == 'Yes' else 0

if income_value <= 65000:
    income_weighted = income_weight
else:
    income_weighted = 0

total_weight = 0.5 + 0.3 + income_weight + michigan_weight + sum(weight for _, weight in gpa_ranges)
income_weight /= total_weight
michigan_weight /= total_weight

for i in range(len(gpa_ranges)):
    gpa_ranges[i] = (gpa_ranges[i][0], gpa_ranges[i][1] / total_weight)

michigan_weighted = michigan_weight * michigan_value

user_gpa = user_input_dict['GPA']
gpa_weight = 0.0

for gpa_range, weight in gpa_ranges:
    if user_gpa >= gpa_range:
        gpa_weight = weight
        break

predicted_chance = (
    0.4 * schools_dict[user_input_dict['School']] +
    0.3 * user_prediction[0] +
    income_weight * income_weighted +
    michigan_weight * michigan_weighted +
    sum(weight * gpa_weight for gpa_range, weight in gpa_ranges)
)

percent_chance = predicted_chance * 100

print(f'Your predicted acceptance chance is: {percent_chance:.2f}%')
