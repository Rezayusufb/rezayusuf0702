#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("F:\\reza\\diabetes.csv")


# In[3]:


df.head()


# * Data yang digunakan adalah data diabetes, jika seseorang tersebut tidak terkena diabetes maka nilainya 0
# * Jika orang tersebut terkena diabetes maka nilainya 1
# 

# In[4]:


df.Outcome.value_counts()


# In[6]:


import seaborn as sns


# In[13]:


sns.countplot(x="Outcome", data=df, palette="bwr")
plt.show()


# In[14]:


countNoDiabetes = len(df[df.Outcome == 0])
countHaveDiabetes = len(df[df.Outcome == 1])
print("Percentage of Patients Have Diabetes: {:.2f}%".format((countNoDiabetes / (len(df.Outcome))*100)))
print("Percentage of Patients Havent Diabetes: {:.2f}%".format((countHaveDiabetes / (len(df.Outcome))*100)))


# In[15]:


df.groupby('Outcome').mean()


# In[16]:


pd.crosstab(df.Pregnancies,df.Outcome).plot(kind="bar",figsize=(20,6))
plt.title('Diabetes Frequency for pregnancies')
plt.xlabel('Pregnancies')
plt.ylabel('Frequency')
plt.savefig('Diabetesfrequencybypregnancies.png')
plt.show()


# * Tampilan histogram dari data x = pregnancies.
# * Warna biru menunjukkan bahwa tidak terkena diabetes
# * Warna orange menunjukkan terkena diabetes

# In[19]:


pd.crosstab(df.Age,df.Outcome).plot(kind="bar",figsize=(20,6))
plt.title('Diabetes Frequency for Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('DiabetesfrequencybyAge.png')
plt.show()


# * Histogram dengan melihat umur
# * Orang terkena diabetes paling banyak pada umur 41 tahun dengan frekuensi 10 orang

# In[21]:


plt.scatter(x=df.Age[df.Outcome==1], y=df.Glucose[(df.Outcome==1)], c="red")
plt.scatter(x=df.Age[df.Outcome==0], y=df.Glucose[(df.Outcome==0)])
plt.legend(["Diabetes", "Not Diabetes"])
plt.xlabel("Age")
plt.ylabel("Max Diabetes Rate")
plt.show()


# * Sebaran data dari orang yang terkena diabetes dengan melihat dari usia, penyebaran data cukup merata dengan melihat orang yang terkena diabetes lebih sedikit daripada yang terkena diabetes.
# * Usia paling rawan terkena diabetes adalah 40 - 50 tahun
# 

# In[23]:


plt.scatter(x=df.Age[df.Outcome==1], y=df.BloodPressure[(df.Outcome==1)], c="red")
plt.scatter(x=df.Age[df.Outcome==0], y=df.BloodPressure[(df.Outcome==0)])
plt.legend(["Diabetes", "Not Diabetes"])
plt.xlabel("Age")
plt.ylabel("Max Diabetes Rate")
plt.show()


# * Orang yang terkena diabetes dan memiliki tekanan darah tidak terlalu banyak, karena dari gambar lebih dominan yang tidak terkena yaitu warna biru
# * Tekanan darah dari orang yang terkena diabetes paling banyak antara 80 - 100 dan pada usia 50 - 60 tahun
# 

# In[59]:


df.describe()


# * Dari analisis deskriptif diatas bisa dilihat masing - masing variabel independen memiliki nilai
# * Untuk nilai maksimal dari pregnancies adalah 17
# * Untuk nilai maksimal dari Glucose adalah 199
# * Dan jumlah data yang ada sebanyak 768

# In[24]:


y = df.Outcome.values
x_data = df.drop(['Outcome'], axis = 1)


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ### Normalize Data
# <br>
# <br>
# <img src="https://1.bp.blogspot.com/-n06ZvyeI2HM/WrMW4KervoI/AAAAAAAAC7k/rktq848B-3IZdYrQzsEVaQe3agq1GzruwCLcBGAs/s640/min%2Bmax.jpg" width="400px"/>

# In[30]:


# Normalize
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values


# In[31]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)


# In[32]:


#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


# * Normalisasi data ditunjukkan untuk meminimalisir hasil analisis yang tidak relevan atau tidak valid. Sehingga normalisasi data sangat diperlukan dalam analisis regresi

# In[33]:


#initialize
def initialize(dimension):
    
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias


# In[34]:


def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head


# * Karena variabel dependen ini bersifat kategori ( diabetes dan Non diabetes) 
# * Scatter plot yang di dapatkan tidak menunjukan sebaran data yang mengumpul dan mendekati garis
# 

# ### Forward and Backward 
# <br>
# <img src="http://umardanny.com/wp-content/uploads/2014/03/depthfirstsearch.jpg" width="500px"/>

# In[35]:


def forwardBackward(weight,bias,x_train,y_train):
    # Forward
    
    y_head = sigmoid(np.dot(weight.T,x_train) + bias)
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    
    # Backward
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}
    
    return cost,gradients


# * Metode backward dan forward digunakan untuk melihat fakta - fakta yang sama dari masing - masing variabel terdekat.
# * Metode backward dan forward ini dipengaruhi oleh tiga macam teknik penelusuran yaitu : Deep -First Search, Breadth firsh search, Best first search
# 

# ### Cost Function
# <br>
# <img src="https://i.stack.imgur.com/XbU4S.png" width="500px"/>

# ### Gradient Descent
# <br>
# <img src="https://i.stack.imgur.com/pYVzl.png" width="500px"/>

# In[36]:


def update(weight,bias,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    
    #for each iteration, update weight and bias values
    for i in range(iteration):
        cost,gradients = forwardBackward(weight,bias,x_train,y_train)
        weight = weight - learningRate * gradients["Derivative Weight"]
        bias = bias - learningRate * gradients["Derivative Bias"]
        
        costList.append(cost)
        index.append(i)

    parameters = {"weight": weight,"bias": bias}
    
    print("iteration:",iteration)
    print("cost:",cost)

    plt.plot(index,costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients


# * Metode ini bekerja dengan memulai dari sebuah tebakan awal (boleh acak) dan secara iteratif tebakan ini diperbaiki berdasarkan suatu aturan yang melibatkan gradien/turunan pertama dari fungsi yang ingin diminimumkan.

# In[37]:


def predict(weight,bias,x_test):
    z = np.dot(weight.T,x_test) + bias
    y_head = sigmoid(z)

    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction


# * Karena untuk mengalisa data dengan nilai y lebih dari 1 maka menggunakan metode regresi logistik dan salah satu syaratnya adalah data harus normal.

# In[38]:


def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
    dimension = x_train.shape[0]
    weight,bias = initialize(dimension)
    
    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)

    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))


# * Regresi logistik adalah salah satu regresi non linear karena tidak membutuhkan asumsi - asumsi dari regresi. 
# * diatas adalah bentuk dari persamaan regresi logistik

# In[39]:


logistic_regression(x_train,y_train,x_test,y_test,1,100)


# In[41]:


accuracies = {}

lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
acc = lr.score(x_test.T,y_test.T)*100

accuracies['Logistic Regression'] = acc
print("Test Accuracy {:.2f}%".format(acc))


# * Akurasi dari model yang di dapat mendapatkan nilai dari 80.52% 

# **KNN Algorithm**
# <br>
# <img src="https://www.python-course.eu/images/k_NN.png"/>

# In[42]:


# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)

print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))


# * Algoritma KNN diklasifikan berdasarkan mayoritas dari kategori pada KNN. Tujuan dari algoritma ini adalah mengklasifikasikan obyek baru bedasarkan atribut dan training sample. Classifier tidak menggunakan model apapun untuk dicocokkan dan hanya berdasarkan pada memori
# * Algoritma metode KNN sangatlah sederhana, bekerja berdasarkan jarak terpendek dari query instance ke training sample untuk menentukan KNN-nya
# 

# In[43]:


# try ro find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)*100
accuracies['KNN'] = acc
print("Maximum KNN Score is {:.2f}%".format(acc))


# **Support Vector Machine Algorithm**
# <br>
# <img src="data:image/png;base64,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" width="500px"/>

# * teknik SVM digunakan untuk menemukan fungsi pemisah(klasifier) yang optimal yang bisa memisahkan dua set data dari dua kelas yang berbeda. Penggunaan teknik machine learning tersebut, karena performansinya yang meyakinkan dalam memprediksi kelas suatu data baru.
# 

# In[44]:


from sklearn.svm import SVC


# In[45]:


svm = SVC(random_state = 1)
svm.fit(x_train.T, y_train.T)

acc = svm.score(x_test.T,y_test.T)*100
accuracies['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))


# **Naive Bayes Algorithm**
# <br>
# <img src="https://s3.ap-south-1.amazonaws.com/techleer/204.png" width="500px"/>

# * asumsi yg sangat kuat (naïf) akan independensi dari masing-masing kondisi / kejadian.
# * Naive Bayes Classifier bekerja sangat baik dibanding dengan model classifier lainnya
# * Keuntungan penggunan adalah bahwa metoda ini hanya membutuhkan jumlah data pelatihan (training data) yang kecil untuk menentukan estimasi parameter yg diperlukan dalam proses pengklasifikasian

# In[46]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)

acc = nb.score(x_test.T,y_test.T)*100
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))


# ** Decision Tree Algoritm**
# <br>
# <img src="http://dataaspirant.com/wp-content/uploads/2017/01/B03905_05_01-compressor.png" width="500px"/>

# * model prediksi menggunakan struktur pohon atau struktur berhirarki
# * menemukan hubungan tersembunyi antara sejumlah calon variabel input dengan sebuah variabel target. Decision tree memadukan antara eksplorasi data dan pemodelan, sehingga sangat bagus sebagai langkah awal dalam proses pemodelan bahkan ketika dijadikan sebagai model akhir dari beberapa teknik lain.

# In[47]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train.T, y_train.T)

acc = dtc.score(x_test.T, y_test.T)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))


# In[48]:


# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train.T, y_train.T)

acc = rf.score(x_test.T,y_test.T)*100
accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))


# In[51]:


colors = ["red", "green", "orange", "pink","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()


# * Nilai  yang di  dapatkan dari beberapa uji dengan menggunakan logistic, knn, svm, naive bayes, decision tree, dan random forest mendapatkan hasil akurasi diatas 50 %, Bisa dikatakan rata - ratanya sekitar 80%. 
# 

# In[52]:


# Predicted values
y_head_lr = lr.predict(x_test.T)
knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(x_train.T, y_train.T)
y_head_knn = knn3.predict(x_test.T)
y_head_svm = svm.predict(x_test.T)
y_head_nb = nb.predict(x_test.T)
y_head_dtc = dtc.predict(x_test.T)
y_head_rf = rf.predict(x_test.T)


# In[53]:


from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test,y_head_lr)
cm_knn = confusion_matrix(y_test,y_head_knn)
cm_svm = confusion_matrix(y_test,y_head_svm)
cm_nb = confusion_matrix(y_test,y_head_nb)
cm_dtc = confusion_matrix(y_test,y_head_dtc)
cm_rf = confusion_matrix(y_test,y_head_rf)


# * Confusion matrix adalah suatu metode yang biasanya digunakan untuk melakukan perhitungan akurasi pada konsep data mining atau Sistem Pendukung Keputusan. Pada pengukuran kinerja menggunakan confusion matrix, terdapat 4 (empat) istilah sebagai representasi hasil proses klasifikasi
# * True Positive (TP), True Negative (TN), False Positive (FP) dan False Negative (FN). Nilai True Negative (TN) merupakan jumlah data negatif yang terdeteksi dengan benar, sedangkan False Positive (FP) merupakan data negatif namun terdeteksi sebagai data positif. Sementara itu, True Positive (TP) merupakan data positif yang terdeteksi benar. False Negative (FN) merupakan kebalikan dari True Positive, sehingga data posifit, namun terdeteksi sebagai data negatif.
# 
# 

# In[54]:


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,6)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()

