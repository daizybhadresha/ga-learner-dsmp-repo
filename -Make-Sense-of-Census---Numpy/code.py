# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data=np.genfromtxt(path, delimiter=",", skip_header=1)

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
census = np.concatenate((data, new_record))


# --------------
#Code starts here

age = census[:,0]

max_age = np.max(age)
min_age = np.min(age)
age_mean = np.mean(age)
age_std = np.std(age)



# --------------
#Code starts here

race_0 = census[census[:,2] == 0]
race_1 = census[census[:,2] == 1]
race_2 = census[census[:,2] == 2]
race_3 = census[census[:,2] == 3]
race_4 = census[census[:,2] == 4]

len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)

all_len = np.array([len_0,len_1,len_2,len_3, len_4])
minority_race = np.argmin(all_len)
print(minority_race) 




# --------------
#Code starts here

senior_citizens = census[census[:,0] > 60]
working_hours_sum = np.sum(senior_citizens[:,6])
senior_citizens_len = len(senior_citizens)
avg_working_hours = np.mean(working_hours_sum)/senior_citizens_len
print("Avg. Working Hours: ",avg_working_hours)


# --------------
#Code starts here

high = census[census[:,1] > 10]
low = census[census[:,1] <= 10]

avg_pay_high = np.mean(high[:,-1])
avg_pay_low = np.mean(low[:,-1])

print("% of people having with education yrs > 10 income > 50k: ",avg_pay_high)
print("% of people having with education yrs <= 10 income > 50k: ",avg_pay_low)





