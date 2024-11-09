import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import shap
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten

df = pd.read_excel(r'C:\Users\noasc\PycharmProjects\pythonProject\project\final project\all_data_long.xlsx')
df.columns = df.columns.str.replace("'", "")


simple_df = df[['year', 'מין','סוג פיקוח', 'שם מגזר',
       ' פסיכומטרי רב תחומי', 'פסיכומטרי כמותי', 'פסיכומטרי מילולי',
       'שפת פסיכומטרי', 'מצב בגרות', 'בגרות', 'ציון מכינה', 'שנת מכינה',
       'מסלול מכינה', 'ציון אנגלית', 'הצהיר רל"ק',
       'ניקוד רל"ק', 'שנת לידה', 'סוג תושבות', 'עתודאי', 'פקולטה', 'חוג', 'מסלול',
      'משקולל קיים', 'שנת עליה', 'רמת אנגלית', 'ציון עברית_x', 'סה"כ יחידות',
      'מתקבל לפי משוקלל שנתי', 'מתקבל לפי משוקלל 2019',
       'מתקבל לפי משוקלל 2024','מתקבל לפי בגרות שנתי',
       'מתקבל לפי בגרות 2019',
       'מתקבל לפי בגרות 2024',
      'מתקבל לפי פסיכומטרי',
       'מתקבל לפי פסיכומטרי 2019',
       'מתקבל לפי פסיכומטרי 2024', 'מתקבל לפי מכינה',
       'מתקבל לפי מכינה 2019',
       'מתקבל לפי מכינה 2024', 'נשר חיצונית']]

small_df = df[['year', 'מין', 'סוג פיקוח', ' פסיכומטרי רב תחומי', 'פסיכומטרי כמותי', 'פסיכומטרי מילולי',
               'שפת פסיכומטרי', 'בגרות', 'ציון מכינה', 'ציון אנגלית', 'ניקוד רל"ק', 'שנת לידה', 'פקולטה', 'חוג', 'מסלול', 'משקולל קיים', 'נשר חיצונית']]

def remove_rer_hugs(df):
    hug_cout_df = df[['חוג']].groupby(['חוג']).size().reset_index(name='counts')
    hug_cout_df = hug_cout_df[hug_cout_df['counts'] > 70]
    df = df[df['חוג'].isin(hug_cout_df['חוג'])]
    return df


def clean_data(df):
    """
    :param df: sample size = 44868
    :return:
    """
    df = remove_rer_hugs(df) # remove 811 samples
    df['מין'] = np.where(df['מין'] == 'ז', 0, 1)
    df['מסלול'] = np.where(df['מסלול'] == 'חד חוגי', 0, 1)
    df = df[df['בגרות'] != 0]# remove 1252 samples
    df['סוג פיקוח'] = np.where(df['סוג פיקוח'].isna(), 'חסר', df['סוג פיקוח'])
    df['רלק'] = np.where(df['ניקוד רל"ק'] >= 30, 1, 0)
    df.pop('ניקוד רל"ק')
    df['חוג_פקולטה'] = df['חוג'].astype(str) + '_' +df['פקולטה']
    df.pop('חוג')
    df.pop('פקולטה')
    return df


# create embeding for hug_fac
label_encoder = LabelEncoder()
df['hug_fac_encoded'] = label_encoder.fit_transform(df['חוג_פקולטה'])

# Define the embedding layer
n_unique_categories = X['school_course_encoded'].nunique()
embedding_dim = 4  # Choose a reasonable dimension for the embedding

# Input layer for the encoded school_course feature
input_school_course = tf.keras.Input(shape=(1,))
embedding_layer = layers.Embedding(input_dim=n_unique_categories, output_dim=embedding_dim)(input_school_course)
flattened_embedding = layers.Flatten()(embedding_layer)