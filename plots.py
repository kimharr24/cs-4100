import matplotlib.pyplot as plt
import seaborn as sns

# MLP vs CNN main result

mlp_scores = [0.72, 0.73, 0.73, 0.74, 0.73, 0.74, 0.75, 0.75, 0.73]
cnn_scores = [0.79, 0.78, 0.79, 0.79, 0.78, 0.79, 0.77, 0.78, 0.78]

sns.kdeplot(mlp_scores, label="MLP", color="blue", fill=True)
sns.kdeplot(cnn_scores, label="CNN", color="red", fill=True)
plt.title("MLP vs CNN Performance On Test Set")
plt.xlabel("Weighted Average F1-Score")
plt.ylabel("Density")
plt.legend()

plt.show()

# Reversed sentences results

mlp_score_differences = [0.13, 0.14, 0.13, 0.12, 0.13, 0.13, 0.14, 0.14, 0.14]
cnn_score_differences = [0.10, 0.11, 0.10, 0.12, 0.10, 0.09, 0.10, 0.10, 0.11]

sns.kdeplot(mlp_score_differences, label="MLP", color="blue", fill=True)
sns.kdeplot(cnn_score_differences, label="CNN", color="red", fill=True)
plt.title("MLP vs CNN Performance Drop On Reversed Test Set")
plt.xlabel("Drop in Weighted Average F1-Score")
plt.ylabel("Density")
plt.legend()

plt.show()

# Domain-specific word2vec results

mlp_score_differences = [0, 0, 0.01, 0, 0.01, 0.02, 0, 0, 0, 0.01]
cnn_score_differences = [0.03, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03]
sns.kdeplot(mlp_score_differences, label="MLP", color="blue", fill=True)
sns.kdeplot(cnn_score_differences, label="CNN", color="red", fill=True)
plt.title("MLP vs CNN Performance Boost On Twitter Word2Vec")
plt.xlabel("Boost In Weighted Average F1-Score")
plt.ylabel("Density")
plt.legend()

plt.show()
