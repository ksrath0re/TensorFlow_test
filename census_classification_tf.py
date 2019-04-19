import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report

def label_fix(label):
    if(label =='<=50'):
        return 0
    else:
        return 1
tf.reset_default_graph()
census_data = pd.read_csv('census_data.csv')
#print(census_data.head(5))

print(census_data['income_bracket'].unique())

census_data['income_bracket'] = census_data['income_bracket'].apply(label_fix)

#perform train test split using sklearn
x_data = census_data.drop('income_bracket', axis=1)
y_labels = census_data['income_bracket']

X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=101)

#TF offers an API called estimator

#print('Columns in data ', census_data.columns)
data_columns = ['age', 'workclass', 'education', 'education_num', 'marital_status',
       'occupation', 'relationship', 'race', 'gender', 'capital_gain',
       'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']

#create feature column for categorical values
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"]) #list for limited known values
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000) #hashbucket for unknown unlimited values
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

#create feature column for continuous values
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

feature_cols = [gender, occupation, marital_status, relationship,
                   education, workclass, native_country,
                   age, education_num, capital_gain,
                   capital_loss, hours_per_week]

#We need these two things.. (1) Feature Colums (2) Input Function ( for creating model)

input_function = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=100,num_epochs=None,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feature_cols)
print('Model Details : ',model)
model.train(input_fn=input_function, steps=500)
#training done

#Now for prediction

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test),shuffle=False)

predictions = list(model.predict(input_function))

print(predictions[0])

final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

#Use sklearn for classification report

print(classification_report(y_test, final_preds))