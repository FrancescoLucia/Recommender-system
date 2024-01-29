import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotY(Y):
    plt.figure("Y")
    plt.imshow(Y, cmap='viridis', aspect='auto')
    plt.ylabel('Movies')
    plt.xlabel('Users')
    plt.title('Movie Ratings Matrix')
    plt.colorbar(label='Rating')
    plt.show(block=False)


def normalizeRatings(Y, R):
    [m, n] = Y.shape
    Ymean = pd.Series(np.zeros(m))
    Ynorm = pd.DataFrame(np.zeros(Y.shape))
    for i in range(m):
        idx = np.where(R.iloc[i, :] == 1)[0]
        Ymean.iloc[i] = Y.iloc[i, idx].mean()
        Ynorm.iloc[i, idx] = Y.iloc[i, idx] - Ymean.iloc[i]
    return {"Ynorm": Ynorm, "Ymean": Ymean}


def costFunction(X, Theta, Y, R, usersNumber, moviesNumber, featuresNumber, lambda_):
    errors = pd.DataFrame((( np.dot(X.to_numpy(), Theta.T.to_numpy()) - Y.to_numpy()) * R.to_numpy()))
    squared_errors = errors ** 2
    J = ((1 / 2) * np.sum(squared_errors.to_numpy()) +
         (lambda_ / 2) * np.sum(Theta.to_numpy().flatten() ** 2) +
         (lambda_ / 2) * np.sum(X.to_numpy().flatten() ** 2))
    X_grad = np.dot(errors, Theta) + (lambda_ * X)
    Theta_grad = np.dot(errors.T, X) + (lambda_ * Theta)
    return [J, X_grad, Theta_grad]

def gradientDescent(X, Theta, Y, R, usersNumber, moviesNumber, featuresNumber, lambda_, alpha, iterations ):
    J_history = np.ones(iterations)
    for iter in range(iterations):
        [J, X_grad, Theta_grad] = costFunction(X, Theta, Y, R, usersNumber, moviesNumber, featuresNumber, lambda_)
        print(f"Iteration {iter}, cost: {J}")
        X -= alpha * X_grad
        Theta -= alpha * Theta_grad
        J_history[iter] = J
    return {"J_history": J_history, "X": X, "Theta": Theta}

def main():
    Y = pd.read_csv('Y.csv', sep=';', header=None)
    moviesNumber, usersNumber = Y.shape
    R = pd.read_csv('R.csv', sep=';', header=None)
    movieList = pd.read_csv("movie_ids.csv", sep=";")
    print(f"Loaded {moviesNumber} movies and {usersNumber} users")

    movieId = 0
    movieName = movieList.iloc[movieId]["name"]
    movieRatings = Y.loc[movieId, R.iloc[movieId] == 1]
    movieMean = np.mean(movieRatings)
    print(f"Average rating for movie '{movieName}': {movieMean} / 5")

    plt.imshow(Y, cmap='viridis', aspect='auto')
    plt.ylabel('Movies')
    plt.xlabel('Users')
    plt.title('Movie Ratings Matrix')
    plt.colorbar(label='Rating')
    plt.show(block=False)

    newRatings = np.zeros(moviesNumber)
    newRatings[0] = 4
    newRatings[97] = 2
    newRatings[49] = 5
    newRatings[55] = 4
    newRatings[66] = 4
    newRatings[71] = 4
    newRatings[81] = 5
    newRatings[95] = 4
    newRatings[120] = 4
    newRatings[126] = 5
    newRatings[143] = 5
    newRatings[173] = 5
    newRatings[179] = 4
    newRatings[187] = 5
    newRatings[209] = 5
    print("New user ratings:")
    for index in range(moviesNumber):
        if newRatings[index] > 0:
            movieName = movieList.iloc[index]["name"]
            print(f"Rate {newRatings[index]} for movie {movieName}")
    newRatingsSeries = pd.Series(newRatings)
    Y = pd.concat([newRatingsSeries, Y], axis=1)
    [moviesNumber, usersNumber] = Y.shape
    print(f"Now dataset contains {usersNumber} users")
    newRatingsBool = (newRatings != 0).astype(int)
    R = pd.concat([pd.Series(newRatingsBool), R], axis=1)

    normalizeResults = normalizeRatings(Y, R)
    Ynorm = normalizeResults["Ynorm"]
    Ymean = normalizeResults["Ymean"]

    featuresNumber = 20
    epsilon_init = 1
    X = pd.DataFrame(np.random.randn(moviesNumber, featuresNumber) * epsilon_init)
    Theta = pd.DataFrame(np.random.randn(usersNumber, featuresNumber) * epsilon_init)
    lambda_ = 2
    alpha = 0.002
    iterations = 200
    trainingResults = gradientDescent(X, Theta, Ynorm, R, usersNumber, moviesNumber, featuresNumber, lambda_, alpha, iterations)
    J_history = trainingResults["J_history"]
    X = trainingResults["X"]
    Theta = trainingResults["Theta"]
    plt.figure("j")
    plt.plot(J_history)
    plt.title('Cost Function History')
    plt.ylabel('Cost Function')
    plt.xlabel('Iterations')
    plt.show()
    print("Learning completed")

    print("Starting predictions")
    predictions = np.dot(X, Theta.T)
    newUserPrediction = pd.Series(predictions[:, 0] + Ymean)
    newUserPrediction = newUserPrediction.sort_values(ascending=False)
    for i in range(20):
        movieRate = newUserPrediction.iloc[i]
        movieId = newUserPrediction.index[i]
        movieName = movieList.iloc[movieId]["name"]
        print(f"Predicting rating {movieRate} for movie {movieName}")

    print("Finding similarities")
    movieId = 50
    print(f"movie: {movieList.iloc[movieId - 1]['name']}")
    similaritySeries = pd.Series(np.zeros(moviesNumber))
    for i in range(moviesNumber):
        similaritySeries.iloc[i] = np.linalg.norm(X.iloc[i, :] - X.iloc[movieId, :])
    similaritySeries.sort_values(ascending=True)
    max = 11
    preferedMoviesRate = similaritySeries.sort_values(ascending=True).iloc[1:max]
    for index in preferedMoviesRate.index:
        movieName = movieList.iloc[index]["name"]
        movieSimilarity = preferedMoviesRate.loc[index]
        print(f"Movie {movieName} --- with similarity {movieSimilarity}")

if __name__ == "__main__":
    main()
