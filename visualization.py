import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import colors

# Load the preprocessed data
with open('preprocessed_data.pickle', 'rb') as f:
    X, y, X_test, y_test, df_train, df_test = pickle.load(f)
# Define color function for word cloud
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # Create a blue gradient color scheme for negative sentiment word cloud
    gradient = colors.LinearSegmentedColormap.from_list('blue_gradient', ['#0000FF', '#ADD8E6'])
    return tuple(int(x * 255) for x in gradient(random_state.randint(0, 255)))

# Generate word cloud for negative sentiment
data_neg = df_train[df_train.label == 0].text
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=1000, width=1600, height=800, collocations=False, color_func=color_func).generate(" ".join(" ".join(i) for i in data_neg))
plt.imshow(wc)
plt.axis('off')
plt.show()

# Define color function for word cloud
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # Create a red gradient color scheme for positive sentiment word cloud
    gradient = colors.LinearSegmentedColormap.from_list('red_gradient', ['#FF0000', '#FFA07A'])
    return tuple(int(x * 255) for x in gradient(random_state.randint(0, 255)))

# Generate word cloud for positive sentiment
data_pos = df_train[df_train.label == 1].text
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=1000, width=1600, height=800, collocations=False, color_func=color_func).generate(" ".join(" ".join(i) for i in data_pos))
plt.imshow(wc)
plt.axis('off')
plt.show()
