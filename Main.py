#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import streamlit as st
from scipy.stats import chi2_contingency


# # DATA UNDERSTANDING

# In[2]:


df = pd.read_csv('F:/Coapps/Medimapper/Medimapper_dataset.csv')


# In[3]:


df.head()


# In[5]:


#df.shape


# In[6]:


# df.info


# In[7]:


#  print("Columns in the dataset:")
# for column in df.columns:
#     print(column)


# In[6]:


df.describe()


# In[7]:


df.describe(include='O')


# In[10]:


# for column in df.columns:
#     # unique_values = df[column].unique()
#     print(f"Unique values in {column}: {unique_values}")


# ## DATA CLEANING

# In[8]:


df.drop(['Name'],axis=1,inplace = True)


# In[9]:


df.head()


# In[13]:


# df.isnull().sum()


# In[14]:


# df.duplicated().sum()


# In[10]:


df.drop_duplicates(inplace=True)


# In[16]:


# df.shape


# In[17]:


# # Function to detect outliers using IQR method
# def detect_outliers_iqr(column):
#     Q1 = column.quantile(0.25)
#     Q3 = column.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = column[(column < lower_bound) | (column > upper_bound)]
#     return outliers

# outliers_age = detect_outliers_iqr(df['Age'])

# if not outliers_age.empty:
#     print("Outliers in the 'Age' column:")
#     print(outliers_age)
# else:
#     print("No outliers found in the 'Age' column.")


# ## EXPLORATORY DATA ANALYSIS

# In[18]:


# for col in df.columns:
#     most_frequent_values = df[col].value_counts().head(5)
#     print(f"Most frequent values in {col}:")
#     print(most_frequent_values)
#     print()


# In[19]:


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Calculate correlation matrix
corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

# Calculate correlation for numerical columns
numerical_columns = df.select_dtypes(include=['int', 'float']).columns
for col1 in numerical_columns:
    for col2 in numerical_columns:
        corr_matrix.loc[col1, col2] = df[col1].corr(df[col2])

# Calculate correlation for categorical columns using Cramer's V
categorical_columns = df.select_dtypes(include=['object']).columns
for col1 in categorical_columns:
    for col2 in categorical_columns:
        corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix.astype(float), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[20]:


plt.hist(df['Age'],bins=30,edgecolor="black")
plt.xlabel('AGE')
plt.ylabel('FREQUENCY')
plt.title('VISUALIZATION OF THE AGE GROUPS MOST FREQUENTLY AFFECTED BY ILLNESS')
plt.show()


# In[21]:


custom_palette = {'Male': 'red', 'Female': 'purple'}
plt.figure(figsize=(6, 2))
sns.countplot(y='Gender', data=df, order=df['Gender'].value_counts().index, palette=custom_palette)
plt.title('GENDER DISTRIBUTION')
plt.xlabel('COUNT')
plt.ylabel('GENDER')
plt.show()


# In[22]:


plt.figure(figsize=(8, 6))
df['Blood Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['red', 'blue', 'green', 'pink','purple','yellow','lightblue','lightgreen'])
plt.title('BLOOD TYPE DISTRIBUTION')
plt.ylabel('')
plt.show()


# In[23]:


plt.figure(figsize=(8, 6))
df['Test Result'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['red', 'blue', 'green'], wedgeprops=dict(width=0.3))
plt.title('TEST RESULT DISTRIBUTION')
plt.ylabel('')
plt.show()


# In[24]:


plt.figure(figsize=(8, 6))
df['Disease'].value_counts().sort_values().plot(kind='barh', color='red')
plt.title('THE AGGREGATE COUNT OF INDIVIDUALS AFFLICTED BY A SPECIFIC ILLNESS.')
plt.xlabel('COUNT OF PEPOPLE')
plt.ylabel('LIST OF DISEASE')
plt.show()


# In[25]:


from wordcloud import WordCloud
plt.figure(figsize=(10, 8))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Medication']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('MEDICATION WORD CLOUD')
plt.show()


# In[26]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Test Result', hue='Test Result', data=df, palette='pastel')
plt.title(' COUNT OF TEST RESULT')
plt.xlabel('TEST RESULT')
plt.ylabel('COUNT')
plt.show()


# In[27]:


custom_palette = {'Male': 'blue', 'Female': 'purple'}
plt.figure(figsize=(8, 6))
sns.violinplot(x='Gender', y='Age', data=df, palette=custom_palette)
plt.title('AGE DISTRIBUTION BY GENDER')
plt.xlabel('GENDER')
plt.ylabel('AGE')
plt.show()


# In[28]:


test_medication = df.groupby(['Test Result', 'Medication']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 8))
test_medication.plot(kind='bar', stacked=False, cmap='YlGnBu')
plt.title('Test Result vs. Medication')
plt.xlabel('Test Result')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Medication', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ## DATA PREPROCESSING

# In[11]:


categorical_columns = ['Disease','Test Result']
numerical_columns = ['Age']

X = df[categorical_columns + numerical_columns]
y = df['Medication']


# In[12]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns,)
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 22)


# ## DATA MODELLING

# In[15]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=22)
classifier.fit(X_train, y_train)


# In[16]:


y_pred=classifier.predict(X_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)*100
print("Decision Tree Classifier")
print("Test Accuracy: {:.2f}".format(accuracy))


# In[25]:


# # Compute aggregated confusion matrix
# aggregated_cm = np.sum(cm, axis=0)

# # Print aggregated confusion matrix as an array
# # print(aggregated_cm)


# In[32]:


# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
# print("Cross-validation Scores:", cv_scores)
# print("Mean Cross-validation Score:", np.mean(cv_scores))
# print("Standard Deviation of Cross-validation Scores:", np.std(cv_scores))


# In[55]:


from sklearn.feature_selection import chi2
X = df[['Age', 'Gender', 'Blood Type', 'Disease', 'Test Result']]
X_encoded = pd.get_dummies(X)
y = df['Medication']
chi_scores, p_values = chi2(X_encoded, y)
feature_scores = list(zip(X_encoded.columns, chi_scores, p_values))
feature_scores.sort(key=lambda x: x[1], reverse=True)
for feature, chi_score, p_value in feature_scores:
    print(f"Feature: {feature}, Chi-square Score: {chi_score}, P-value: {p_value}")


# In[35]:


from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(random_state=22)
classifier2.fit(X_train, y_train)


# In[36]:


y_pred_train =classifier2.predict(X_train)
accuracy = metrics.accuracy_score(y_test, y_pred)*100
print("Random Forest Classifier")
print("Test Accuracy: {:.2f}".format(accuracy))


# In[38]:


from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors=5) 
classifier4.fit(X_train, y_train)


# In[39]:


y_pred=classifier4.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)*100
print("KNN")
print("Test Accuracy: {:.2f}".format(accuracy))


# In[40]:


from sklearn.svm import SVC

classifier6 = SVC(random_state=22)
classifier6.fit(X_train, y_train)


# In[42]:


y_pred=classifier6.predict(X_test)
print("Support Vector Machines")
print("Test Accuracy: {:.2f}".format(accuracy))


# In[55]:


st.set_page_config(layout="wide")
with st.sidebar:
    st.title("Medication Recommendation System")  
    st.markdown(
        """
        ## About the Project
        
        This is a Medication Recommendation System designed to provide personalized medication recommendations 
        based on user inputs such as age, gender, blood type, disease, and test results. Simply fill in the 
        required information and click the button to get your medication recommendation.
        """
    )
with st.container():
    st.subheader("User Inputs")
    with st.form(key='my_form'):
        age = st.slider("Age", min_value=1, max_value=100, value=30)
        gender = st.radio("Gender", ["Male", "Female"])
        blood_type = st.selectbox("Blood Type", ["A", "A-", "B", "B-", "AB", "AB-", "O", "O-"])
        disease = st.selectbox("Disease", ["Acne", "Osteoarthritis", "Bronchial Asthma", "Alcoholic hepatitis", "Impetigo",
                                           "Tonsillitis", "(Vertigo) Paroxysmal Positional Vertigo",
                                           "Dimorphic hemorrhoids (piles)", "Tuberculosis", "Pneumonia", "Varicose veins",
                                           "Hypothyroidism", "Heart attack", "Hypoglycemia", "Cervical spondylosis",
                                           "Diabetes", "Common Cold", "Arthritis", "Hypertension", "Chronic cholestasis",
                                           "Migraine", "Urinary tract infection", "Hyperthyroidism",
                                           "GERD (Gastroesophageal Reflux Disease)", "Allergy", "Chickenpox", "Dengue",
                                           "Psoriasis", "Malaria", "Fungal infection", "Jaundice", "Hepatitis A",
                                           "Paralysis (brain hemorrhage)", "Peptic ulcer disease",
                                           "Vertigo (Paroxysmal Positional Vertigo)", "Hepatitis B", "Gastroenteritis",
                                           "Typhoid", "AIDS", "Hepatitis E", "Drug Reaction", "Hepatitis C", "Hepatitis D"])
        test_result = st.selectbox("Test Result", ["Normal", "Abnormal", "Inconclusive"])
        submit_button = st.form_submit_button(label='Get Recommendation', help='Click to get medication recommendation')
    if submit_button:
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Blood Type': [blood_type],
            'Disease': [disease],
            'Test Result': [test_result]
        })
        input_data_encoded = preprocessor.transform(input_data)
        medication = classifier.predict(input_data_encoded)
        st.subheader("Recommended Medication")
        st.write(medication[0])


# In[ ]:




