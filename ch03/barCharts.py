from matplotlib import pyplot as plt

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]
# bars are by default width 0.8, so we'll add 0.1 to the left coordinates
# so that each bar is centered
# plot bars with left x-coordinates [xs], heights [num_oscars]
plt.bar(range(len(movies)), num_oscars)

plt.ylabel("# of Academy Awards")
plt.title("My Favorite Movies")

# label x-axis with movie names at bar centers
plt.xticks([i + 0.01 for i, _ in enumerate(movies)], movies)
plt.show()

plt.savefig('im/viz_movies.png')
plt.gca().clear()