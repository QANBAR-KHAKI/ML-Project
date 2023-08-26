import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------- LOADING THE .NPY DATA -----------------

path1 = r'F:\6TH SEMESTER\MACHINE LEARNING\ML PROJECT\bbh\testing\testAccelerometer.npy'
path2 = r'F:\6TH SEMESTER\MACHINE LEARNING\ML PROJECT\bbh\testing\testGravity.npy'
path3 = r'F:\6TH SEMESTER\MACHINE LEARNING\ML PROJECT\bbh\testing\testGyroscope.npy'
path4 = r'F:\6TH SEMESTER\MACHINE LEARNING\ML PROJECT\bbh\testing\testJinsAccelerometer.npy'
path5 = r'F:\6TH SEMESTER\MACHINE LEARNING\ML PROJECT\bbh\testing\testJinsGyroscope.npy'
path6 = r'F:\6TH SEMESTER\MACHINE LEARNING\ML PROJECT\bbh\testing\testLinearAcceleration.npy'
label_path = r'F:\6TH SEMESTER\MACHINE LEARNING\ML PROJECT\bbh\testing\testLabels.npy'

Accelerometer = np.load(path1)
Gravity = np.load(path2)
Gyroscope = np.load(path3)
JinsAccelerometer = np.load(path4)
JinsGyroscope = np.load(path5)
LinearAcceleration = np.load(path6)
test_labels = np.load(label_path)

print(Accelerometer)
print(Gravity.shape)
print(Gyroscope.shape)
print(JinsAccelerometer.shape)

# ======================= FOR THE ACCELEROMETER DATASET ============

print("The shape of the Accelerometer is : ", np.shape(Accelerometer))

# feature_Matrix = []
feature_Matrix = np.zeros(
    (2288, 21))  # ==> is our final matrix after performing all the calculation for only one device data
print("The shape of feature matrix is : ", feature_Matrix.shape)
# for i in range(2288):
#     # result_matrix = np.zeros((0, 7))
#     result_matrix = []
#     example = Accelerometer[i]
#     for j in range(3):
#         col = example[:, j]
#         # now apply basic eighteen function here, but I am taking only 7 functions here
#         mean_ = np.mean(col)
#         stand_deviation = np.std(col)
#         maximum = np.max(col)
#         ptp = np.ptp(col)
#         minimum = np.min(col)
#         argMin = np.argmin(col)
#         argMax = np.argmax(col)
#         result_matrix.extend([mean_, stand_deviation, maximum, ptp, minimum, argMin, argMax])
#         reshaped_matrix = np.reshape(1,)
#         feature_Matrix[i] = reshaped_matrix

for i in range(2288):
    result_matrix = np.zeros((7, 3))
    #     result_matrix = []
    example = Accelerometer[i]
    for j in range(3):
        col = example[:, j]
        # now apply basic eighteen function here, but I am taking only 7 functions here
        mean_ = np.mean(col)
        stand_deviation = np.std(col)
        maximum = np.max(col)
        ptp = np.ptp(col)
        minimum = np.min(col)
        argMin = np.argmin(col)
        argMax = np.argmax(col)
        values = [mean_, stand_deviation, maximum, ptp, minimum, argMin, argMax]
        result_matrix[:, j] = values

    # result_matrix = np.append(result_matrix, [mean_, stand_deviation, maximum, ptp, minimum, argMin, argMax]).reshape(7, -1)
    reshaped_matrix = result_matrix.flatten()
    # print("The shape of the reshaped_matrix is : ", reshaped_matrix.shape)
    # print("The shape of the result_matrix is : ", result_matrix.shape)
    feature_Matrix[i] = reshaped_matrix

# NOW LET'S RUN RANDOM FOREST ALGORITHM
algo = RandomForestClassifier(n_estimators=100, criterion='gini')

